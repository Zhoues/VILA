# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/

from hmac import new
import os

import torch
from transformers import PretrainedConfig, PreTrainedModel

from .base_projector import MultimodalProjector, MultimodalProjectorConfig


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        if os.path.exists(model_type_or_path) and (os.path.exists(os.path.join(model_type_or_path, "model.pt")) or os.path.exists(os.path.join(model_type_or_path, "model.safetensors"))):
            return MultimodalProjector.from_pretrained(model_type_or_path, config, torch_dtype=eval(config.model_dtype))
        else:
            # assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
            print(f"Resume mm projector path {model_type_or_path} does not exist!")
            print(f"Building mm projector from scratch!")

            if "spatial_projector" in model_type_or_path:
                # NOTE(Zhouenshen): Build spatial projector from scratch for MoGe2
                new_projector_type = config.spatial_projector_cfg.get("mm_projector_type", "mlp_downsample_3x3_fix")
                new_projector_use_cls_token = config.spatial_projector_cfg.get("spatial_tower_vision_select_feature", "cls_patch")
                new_projector_num_tokens = config.spatial_projector_cfg.get("spatial_tower_vision_num_tokens", 3600)
            else:
                raise ValueError(f"Unknown projector: {model_type_or_path}")
            
            new_projector_cfg = MultimodalProjectorConfig(new_projector_type, spatial_tower_vision_select_feature=new_projector_use_cls_token, spatial_tower_vision_num_tokens=new_projector_num_tokens)
            return MultimodalProjector(new_projector_cfg, config).to(eval(config.model_dtype))

    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config).to(eval(config.model_dtype))
        return mm_projector


