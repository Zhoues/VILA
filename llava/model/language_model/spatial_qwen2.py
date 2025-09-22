import torch
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM
from llava.constants import IGNORE_INDEX
from llava.model.language_model.llava_llama import LlavaLlamaModel
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from llava.model.loss import metric_scale_factor_loss_function
from llava.model.multimodal_spatialencoder.MoGe.moge.model.modules import MLP
from llava.model.multimodal_spatialencoder.MoGe.moge.model.v2 import MoGeModel
from llava.train.utils import mprint


class SpatialQwen2Config(Qwen2Config):
    model_type = "spatial_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "spatial_qwen2"


# NOTE(Zhouenshen): SpatialQwen2Model is a Qwen2 LLM model with additional special projector and decoder
class SpatialQwen2Model(Qwen2Model):
    config_class = SpatialQwen2Config

    def __init__(self, config):
        super(SpatialQwen2Model, self).__init__(config)
        self.config = config
        self.metric_scale_factor_projector = None

        if 'out_dim' not in config:
            config.out_dim = 1024

    def build_metric_scale_factor_projector(self, config):            
        # Projection layer for metric scale factor projector
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        ]

        # Projection layer for metric scale factor, its parameters are not frozen
        self.metric_scale_factor_projector = nn.Sequential(*text_fc)
        self.metric_scale_factor_projector.train()
        for param in self.metric_scale_factor_projector.parameters():
            param.requires_grad = True

    def build_metric_scale_factor_decoder(self, config):        
        spatial_tower, moge_own_config = MoGeModel.from_pretrained(config.spatial_tower_weights_path) 
        self.metric_scale_factor_decoder = spatial_tower.scale_head.to(torch.device("cuda"))
        for param in self.metric_scale_factor_decoder.parameters():
            param.requires_grad = False
        del spatial_tower, moge_own_config

class SpatialQwen2ForCausalLM(Qwen2ForCausalLM):

    config_class = SpatialQwen2Config

    def __init__(self, config):
        super(SpatialQwen2ForCausalLM, self).__init__(config)

        self.model = SpatialQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)        
    

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model

    def get_lm_head(self):
        return self.lm_head

    def get_metric_scale_factor_projector(self):
        return getattr(self.model, "metric_scale_factor_projector", None)
    
    def get_metric_scale_factor_decoder(self):
        return getattr(self.model, "metric_scale_factor_decoder", None)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        metric_scale_factors: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        torch.cuda.empty_cache()

        if self.training:
            self.config.metric_scale_factor_projector_and_decoder = True

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # import ipdb; ipdb.set_trace()
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # NOTE(Zhouenshen): Cross-Entropy loss for Next Token predion
            cross_entropy_loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            metric_scale_factor_loss = 0

            if metric_scale_factors is not None and len(metric_scale_factors) > 0:

                # NOTE(Zhouenshen): Metric Scale Factor loss
                # # (batch_size, seq_len, hidden_size) -> (batch_size, seq_len, out_dim) 
                # # hidden_size: 1536, out_dim: 1024

                hidden_states_for_metric_scale_factor = self.model.metric_scale_factor_projector(hidden_states)
                pred_metric_scale_factors_embeddings = []

                # 1. 首先先判断 batch_size 数据中，哪些数据是包含 geo 的，然后使用 geo 的 embedding 进行预测
                # labels: (batch_size, seq_len)， 判断哪条数据是包含 geo_token_ids 的，返回一个 bool 值的 tensor，形状为 (batch_size,)
                geo_token_mask = (labels == self.config.geo_token_ids).any(dim=1)   # like tensor([False, False, False, False, False], device='cuda:0')
                geo_num = geo_token_mask.sum().item()
                
                if geo_num > 0:

                    # 2. 将之前变量进行 shift， (geo_num, seq_len - 1, out_dim)
                    shift_logits_have_gt = logits[geo_token_mask][:, :-1, :]
                    shift_labels_have_gt = labels[geo_token_mask][..., 1:]
                    shift_hidden_states_for_metric_scale_factor_have_gt = hidden_states_for_metric_scale_factor[geo_token_mask][:, :-1, :]
                    metric_scale_factors_have_gt = metric_scale_factors[geo_token_mask]
                    
                    # 3. 获取 shift_logits_have_gt 中预测所有词的 ID，形状为 (len(geo_token_mask), seq_len - 1)
                    shift_vocal_ids_have_gt = torch.argmax(shift_logits_have_gt, dim=-1)

                    # 4. 遍历 shift_vocal_ids_have_gt 中每一个 sample，如果这个 sample 有 geo_token_ids，则将这个 sample 的 embedding 添加到 pred_metric_scale_factors_embeddings 中
                    for idx in range(len(shift_vocal_ids_have_gt)):
                        # 获取到当前 sample 的所有真实词的 ID
                        sample_shift_labels_have_gt = shift_labels_have_gt[idx]
                        # 获取第一个非 IGNORE_INDEX 的索引
                        non_ignore_idx = torch.where(sample_shift_labels_have_gt != IGNORE_INDEX)[0]
                        if len(non_ignore_idx) > 0:
                            first_non_ignore_idx = non_ignore_idx[0]
                        else:
                            geo_num -= 1
                            continue
                        # 获取到当前 sample 的所有预测词的 ID
                        sample_pred_vocal_ids = shift_vocal_ids_have_gt[idx]
                        # 屏蔽 label 无效的 token， 只考虑输出部分
                        sample_pred_vocal_ids[:first_non_ignore_idx] = IGNORE_INDEX
                        # 找到数字和 geo_token_ids 一致的索引
                        sample_pred_geo_token_ids = torch.where(sample_pred_vocal_ids == self.config.geo_token_ids)[0]

                        if len(sample_pred_geo_token_ids) > 0:
                            # 取第一个预测的 [GEO]
                            pred_geo_token_id = sample_pred_geo_token_ids[0]
                        else:
                            # 如果没有预测 [GEO]，预测的第一个token为有效 token
                            pred_geo_token_id = first_non_ignore_idx

                        pred_geo_token_embedding = shift_hidden_states_for_metric_scale_factor_have_gt[idx][pred_geo_token_id]
                        pred_metric_scale_factors_embeddings.append(pred_geo_token_embedding)

                    # 5. 将 pred_metric_scale_factors_embeddings 转换为 tensor，形状为 (geo_num, out_dim), out_dim: 1024
                    pred_metric_scale_factors_embeddings = torch.stack(pred_metric_scale_factors_embeddings)

                    # 6. 通过 Decoder 获得对应的 metric scale factor, 形状为 (geo_num, 1)
                    pred_metric_scale_factors = self.model.metric_scale_factor_decoder(pred_metric_scale_factors_embeddings)

                    # 7. 计算 metric_scale_factor_loss
                    assert pred_metric_scale_factors.shape == metric_scale_factors_have_gt.shape, "pred_metric_scale_factors and metric_scale_factors must have the same shape"
                    assert (metric_scale_factors_have_gt > 0).all(), "metric_scale_factors_have_gt must be positive"
                    metric_scale_factor_loss = metric_scale_factor_loss_function(pred_metric_scale_factors, metric_scale_factors_have_gt)
                    
            mprint(f"crossentropy_loss: {cross_entropy_loss.item():.4f}, metric_scale_factor_loss: {metric_scale_factor_loss.item():.4f}")
            loss = cross_entropy_loss + 0.5 * metric_scale_factor_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoConfig.register("spatial_qwen2", SpatialQwen2Config)
AutoModelForCausalLM.register(SpatialQwen2Config, SpatialQwen2ForCausalLM)