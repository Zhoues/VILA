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

import warnings
from dataclasses import dataclass, field


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str = field(default="torch")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    # NOTE(Zhouenshen): Add the depth path for spatialdataset
    depth_path: str = field(default=None, metadata={"help": "Path to the training depth data."})
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed desciption of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )
    test_script: str = (None,)
    maintainer: str = (None,)
    ############## ############## ############## ############## ############## ##############
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})


DATASETS_LEGACY = {}


def add_dataset(dataset):
    if dataset.dataset_name in DATASETS_LEGACY:
        # make sure the data_name is unique
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})


def register_datasets_mixtures():

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="torch",
        data_path="/home/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/home/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)


    # NOTE(Zhouenshen): Add the spatialvlm dataset for stage 1
    openspaces_spacellava_9k = Dataset(
        dataset_name="openspaces_spacellava_9k",
        dataset_type="spatialdataset",
        data_path="/home/zhouenshen/dataset/vlm/openspaces/train/metadata.json",
        image_path="/home/zhouenshen/dataset/vlm/openspaces/train/images",
        depth_path="/home/zhouenshen/dataset/vlm/openspaces/train/depths",
        description="9.2K SFT data by SpatialVLM w/ depth (template) from the Cauldron Dataset."
    )
    add_dataset(openspaces_spacellava_9k)


    # NOTE(Zhouenshen): Add the spatialvlm dataset for stage 1
    vqasynth_spacellava_25k = Dataset(
        dataset_name="vqasynth_spacellava_25k",
        dataset_type="spatialdataset",
        data_path="/home/zhouenshen/dataset/vlm/vqasynth/train/metadata.json",
        image_path="/home/zhouenshen/dataset/vlm/vqasynth/train/images",
        depth_path="/home/zhouenshen/dataset/vlm/vqasynth/train/depths",
        description="25.2K SFT data by SpatialVLM w/ depth (template) focusing on warehouse scenes."
    )
    add_dataset(vqasynth_spacellava_25k)
