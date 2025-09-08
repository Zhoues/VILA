import torch
from typing import Optional, Union, Tuple, List
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn as nn
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, AutoConfig, AutoModelForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaModel
from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from llava.model.multimodal_spatialencoder.MoGe.moge.model.modules import MLP
from llava.model.multimodal_spatialencoder.MoGe.moge.model.v2 import MoGeModel


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
        self.metric_scale_factor_decoder.eval()
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
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

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