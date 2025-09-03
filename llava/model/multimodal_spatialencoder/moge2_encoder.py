import os
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from llava.model.multimodal_spatialencoder.MoGe.moge.model.v2 import MoGeModel



class MogeEncoderConfig(PretrainedConfig):
    model_type = "moge_encoder"

    def __init__(self, spatial_tower_vision_select_feature: str = "cls_patch", spatial_tower_vision_num_tokens: int = 3600, hidden_size: int = 1024, moge_own_config: dict = None, **kwargs):
        super().__init__()
        self.spatial_tower_vision_select_feature = spatial_tower_vision_select_feature
        self.spatial_tower_vision_num_tokens = spatial_tower_vision_num_tokens
        self.hidden_size = hidden_size
        self.moge_own_config = moge_own_config


class MogeEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__()

        model_weights_path = os.path.join(model_name_or_path, "model.pt")
        if not os.path.exists(model_weights_path):
            model_weights_path = os.path.join(model_name_or_path, "model.safetensors")

        self.spatial_tower, moge_own_config = MoGeModel.from_pretrained(model_weights_path)
        self.spatial_tower = self.spatial_tower.to(torch.device("cuda"))  

        # NOTE(Zhouenshen): The args in config is more priority than the args in spatial_tower_cfg
        if hasattr(config, "spatial_tower_vision_select_feature"):
            self.spatial_tower_vision_select_feature = config.spatial_tower_vision_select_feature
        else:
            self.spatial_tower_vision_select_feature = "cls_patch"

        if hasattr(config, "spatial_tower_vision_num_tokens"):
            self.spatial_tower_vision_num_tokens = config.spatial_tower_vision_num_tokens
        else:
            self.spatial_tower_vision_num_tokens = 3600

        self.config = MogeEncoderConfig(spatial_tower_vision_select_feature=self.spatial_tower_vision_select_feature, spatial_tower_vision_num_tokens=self.spatial_tower_vision_num_tokens, hidden_size=getattr(config, "spatial_hidden_size", 1024), moge_own_config=moge_own_config)

        # Update the spatial tower config
        config.spatial_tower_cfg["spatial_tower_vision_select_feature"] = self.spatial_tower_vision_select_feature
        config.spatial_tower_cfg["spatial_tower_vision_num_tokens"] = self.spatial_tower_vision_num_tokens
        config.spatial_tower_cfg["hidden_size"] = self.hidden_size

    def forward(self, images: torch.Tensor):

        # 获取输入图像的高和宽
        _, _, img_h, img_w = images.shape

        # 计算图像的宽高比
        aspect_ratio = img_w / img_h

        # 根据token数量和宽高比，计算特征图的高和宽
        base_h = int((self.spatial_tower_vision_num_tokens / aspect_ratio) ** 0.5)
        base_w = int((self.spatial_tower_vision_num_tokens * aspect_ratio) ** 0.5)

        # 使用MoGe2的encoder进行特征提取
        # features: (B, hidden_size, base_h, base_w)
        # cls_token: (B, hidden_size)

        features, cls_token = self.spatial_tower.encoder(images, base_h, base_w, return_class_token=True)

        # 将cls_token形状从 (B, hidden_size) 变为 (B, 1, hidden_size)
        cls_token = cls_token.unsqueeze(1)

        # 将features从 (B, hidden_size, base_h, base_w) 变为 (B, base_h * base_w, hidden_size)
        B, hidden_size, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, hidden_size)

        if self.spatial_tower_vision_select_feature == "cls_patch":
        # 拼接cls_token到features前面，得到 (B, base_h * base_w + 1, hidden_size)
            output = torch.cat([cls_token, features], dim=1)
        else:
            output = features

        return output

    def save_pretrained(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.state_dict()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # dict_keys(['model_config', 'model'])
        # save model_config to model.pt['model_config'], save model to model.pt['model']
        torch.save({"model_config": self.config.moge_own_config, "model": state_dict}, os.path.join(output_dir, "model.pt"))

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.spatial_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.spatial_tower.parameters():
            return p.device

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size