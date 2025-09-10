from functools import partial
from typing import Any, Dict, List, Optional

import torch

from llava.model.encoders.base import BaseEncoder

__all__ = ["BasicImageEncoder"]


class BasicImageEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm.model.embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(self, images: List[torch.Tensor], config: Dict[str, Any], is_spatial: bool = False, enable_spatial: bool = True) -> List[torch.Tensor]:
        images = torch.stack(images, dim=0)
        features = self.parent.encode_images(images, block_sizes=config.get("block_sizes"), is_spatial=is_spatial, enable_spatial=enable_spatial)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]

    # def forward(self, images: List[torch.Tensor], config: Dict[str, Any], is_spatial: bool = False, enable_spatial: bool = True) -> List[torch.Tensor]:
    #     process_features = partial(
    #         self._process_features,
    #         start_token_embeds=self.embed_tokens(self.start_tokens),
    #         end_token_embeds=self.embed_tokens(self.end_tokens),
    #     )
    #     # images: [batch_size, num_images, channels, height, width]
        
    #     if is_spatial and enable_spatial:
    #         # NOTE(Zhouenshen): Process each image separately, as MoGe-2 does not support patch processing
    #         process_features_list = []
    #         for image in images:
    #             image_features = self.parent.encode_images(image.unsqueeze(0), block_sizes=config.get("block_sizes"), is_spatial=is_spatial, enable_spatial=enable_spatial)
    #             process_features_list.append(process_features(image_features.squeeze(0)))
    #         return process_features_list
    #     else:
    #         images = torch.stack(images, dim=0)
    #         features = self.parent.encode_images(images, block_sizes=config.get("block_sizes"), is_spatial=is_spatial, enable_spatial=enable_spatial) 
    #         return [process_features(f) for f in features]


