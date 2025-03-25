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

    # llava_1_5_mm_align = Dataset(
    #     dataset_name="llava_1_5_mm_align",
    #     dataset_type="torch",
    #     data_path="/home_sfs/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/chat.json",
    #     image_path="/home_sfs/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/images",
    # )
    # add_dataset(llava_1_5_mm_align)


    # NOTE(Zhouenshen): Add the spatialvlm dataset for stage 1
    # openspaces_spacellava_9k = Dataset(
    #     dataset_name="openspaces_spacellava_9k",
    #     dataset_type="spatialdataset",
    #     data_path="/home_sfs/zhouenshen/dataset/vlm/openspaces/train/metadata.json",
    #     image_path="/home_sfs/zhouenshen/dataset/vlm/openspaces/train/images",
    #     depth_path="/home_sfs/zhouenshen/dataset/vlm/openspaces/train/depths",
    #     description="9.2K SFT data by SpatialVLM w/ depth (template) from the Cauldron Dataset."
    # )
    # add_dataset(openspaces_spacellava_9k)


    # NOTE(Zhouenshen): Add the spatialvlm dataset for stage 1
    # vqasynth_spacellava_25k = Dataset(
    #     dataset_name="vqasynth_spacellava_25k",
    #     dataset_type="spatialdataset",
    #     data_path="/home_sfs/zhouenshen/dataset/vlm/vqasynth/train/metadata.json",
    #     image_path="/home_sfs/zhouenshen/dataset/vlm/vqasynth/train/images",
    #     depth_path="/home_sfs/zhouenshen/dataset/vlm/vqasynth/train/depths",
    #     description="25.2K SFT data by SpatialVLM w/ depth (template) focusing on warehouse scenes."
    # )
    # add_dataset(vqasynth_spacellava_25k)


    # llava_1_5_lrv_mix1008k = Dataset(
    #     dataset_name="llava_1_5_lrv_mix1008k",
    #     dataset_type="torch",
    #     data_path="/home/vlm/finetune_json/llava_v1_5_lrv_mix1008k.json",
    #     image_path="/home/vlm/finetune_json/images",
    # )
    # add_dataset(llava_1_5_lrv_mix1008k)


    # NOTE(Zhouenshen): Add the spatialvlm dataset for stage 1
    template_qa_4_7M = Dataset(
        dataset_name="template_qa_4_7M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_template_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="4.7 M SFT data w/ depth from OpenImage, 2.1 M data is related to point."
    )
    add_dataset(template_qa_4_7M)


    choice_qa_920k = Dataset(
        dataset_name="choice_qa_920k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="920k SFT data w/ depth from OpenImage, 460k data is related to point."
    )
    add_dataset(choice_qa_920k)


    reason_qa_1_2M = Dataset(
        dataset_name="reason_qa_1_2M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="1.2 M SFT data w/ depth from OpenImage, 380k data is related to point."
    )
    add_dataset(reason_qa_1_2M)