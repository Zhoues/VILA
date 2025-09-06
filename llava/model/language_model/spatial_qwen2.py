# from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
# import torch.nn as nn
# import torch.nn.functional as F


# class SpatialQwen2Config(Qwen2Config):
#     model_type = "spatial_qwen2"

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.model_type = "spatial_qwen2"



# class SpatialQwen2Model(Qwen2Model):
#     config_class = SpatialQwen2Config

#     def __init__(self, config: SpatialQwen2Config):
#         super(SpatialQwen2Model, self).__init__(config)

#         if hasattr(config, "metric_scale_factor_decoder"): # inference
#             self.build_metric_scale_factor_decoder(config)
#         else: # training
#             if 'metric_scale_factor_out_dim' not in config:
#                 config.metric_scale_factor_dim = 1024        


#     def build_metric_scale_factor_decoder(self, config):
            
#         # Projection layer for moge metric scale factor prediction
#         metric_scale_factor_in_dim = config.hidden_size
#         metric_scale_factor_out_dim = config.metric_scale_factor_out_dim
#         metric_scale_factor_fc = [
#             nn.Linear(metric_scale_factor_in_dim, metric_scale_factor_in_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(metric_scale_factor_in_dim, metric_scale_factor_out_dim),
#             nn.Dropout(0.0),
#         ]
#         self.metric_scale_factor_hidden_fcs = nn.ModuleList([nn.Sequential(*metric_scale_factor_fc)])
#         self.metric_scale_factor_hidden_fcs.train()
#         for param in self.metric_scale_factor_hidden_fcs.parameters():
#             param.requires_grad = True  



# class SpatialQwen2ForCausalLM(Qwen2ForCausalLM, SpatialQwen2Model, ):
