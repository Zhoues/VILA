from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from llava.model.multimodal_spatialencoder.moge2_encoder import MogeEncoder


# NOTE(Zhouenshen): Build spatial tower, such as moge2
def build_spatial_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_name_or_path is None:
        return None

    spatial_tower_name = model_name_or_path

    if "moge" in spatial_tower_name:
        spatial_tower = MogeEncoder(model_name_or_path, config=config)
    else:
        raise ValueError(f"Unknown spatial tower: {spatial_tower_name}")

    return spatial_tower