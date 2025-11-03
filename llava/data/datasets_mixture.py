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
    # NOTE(Zhouenshen): Add the enable_spatial flag for whether to use spatial encoder.
    spatial_feature_path: str = field(default=None, metadata={"help": "Path to the training spatial feature data."})

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

    ### OpenImage (2D Dataset)

    reason_template_qa = Dataset(
        dataset_name="reason_template_qa",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/2D_OpenImage/spatial_feature_only_image",
    )
    add_dataset(reason_template_qa)

    reason_template_qa_RGB = Dataset(
        dataset_name="reason_template_qa_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
    )
    add_dataset(reason_template_qa_RGB)


    choice_qa = Dataset(
        dataset_name="choice_qa",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/2D_OpenImage/spatial_feature_only_image",
        description="4 M SFT data w/ depth from OpenImage."
    )
    add_dataset(choice_qa)

    choice_qa_RGB = Dataset(
        dataset_name="choice_qa_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(choice_qa_RGB)

    # ### CA-1M (3D Dataset)

    ca1m_reasoning_template_qa_split = Dataset(
        dataset_name="ca1m_reasoning_template_qa_split",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_reasoning_template_qa_split)

    ca1m_reasoning_template_qa_split_RGB = Dataset(
        dataset_name="ca1m_reasoning_template_qa_split_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_reasoning_template_qa_split_RGB)


    ca1m_reasoning_template_qa_split_w_intrinsics = Dataset(
        dataset_name="ca1m_reasoning_template_qa_split_w_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_reasoning_template_qa_split_w_intrinsics)


    ca1m_reasoning_template_qa_split_w_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_reasoning_template_qa_split_w_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_reasoning_template_qa_split_w_intrinsics_and_depth)





    # ca1m_template_qa_split = Dataset(
    #     dataset_name="ca1m_template_qa_split",
    #     dataset_type="geometricdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_template_qa_normalized_1000_split_spatial_feature.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
    #     spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    # )
    # add_dataset(ca1m_template_qa_split)

    # ca1m_template_qa_split_RGB = Dataset(
    #     dataset_name="ca1m_template_qa_split_RGB",
    #     dataset_type="geometricdataset",   
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_template_qa_normalized_1000_split_spatial_feature.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    # )
    # add_dataset(ca1m_template_qa_split_RGB)


    # ca1m_template_qa_split_w_intrinsics = Dataset(
    #     dataset_name="ca1m_template_qa_split_w_intrinsics",
    #     dataset_type="geometricdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_template_qa_normalized_1000_split_spatial_feature.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
    #     spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    # )
    # add_dataset(ca1m_template_qa_split_w_intrinsics)

    # ca1m_template_qa_split_w_intrinsics_and_depth = Dataset(
    #     dataset_name="ca1m_template_qa_split_w_intrinsics_and_depth",
    #     dataset_type="geometricdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_template_qa_normalized_1000_split_spatial_feature.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
    #     spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    # )
    # add_dataset(ca1m_template_qa_split_w_intrinsics_and_depth)
    
    

    ca1m_choice_qa_split = Dataset(
        dataset_name="ca1m_choice_qa_split",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_choice_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_choice_qa_split)


    ca1m_choice_qa_split_RGB = Dataset(
        dataset_name="ca1m_choice_qa_split_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_choice_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_choice_qa_split_RGB)


    ca1m_choice_qa_split_w_intrinsics = Dataset(
        dataset_name="ca1m_choice_qa_split_w_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_choice_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_choice_qa_split_w_intrinsics)

    ca1m_choice_qa_split_w_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_choice_qa_split_w_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_choice_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_choice_qa_split_w_intrinsics_and_depth)


    ca1m_distance_qa_split = Dataset(
        dataset_name="ca1m_distance_qa_split",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_distance_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_distance_qa_split)


    ca1m_distance_qa_split_RGB = Dataset(
        dataset_name="ca1m_distance_qa_split_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_distance_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_distance_qa_split_RGB)


    ca1m_distance_qa_split_w_intrinsics = Dataset(
        dataset_name="ca1m_distance_qa_split_w_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_distance_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_distance_qa_split_w_intrinsics)

    ca1m_distance_qa_split_w_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_distance_qa_split_w_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_distance_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_distance_qa_split_w_intrinsics_and_depth)



    ca1m_visual_choice_qa = Dataset(
        dataset_name="ca1m_visual_choice_qa",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_visual_choice_qa)

    ca1m_visual_choice_qa_RGB = Dataset(
        dataset_name="ca1m_visual_choice_qa_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images",
    )
    add_dataset(ca1m_visual_choice_qa_RGB)


    ca1m_vacant_qa = Dataset(
        dataset_name="ca1m_vacant_qa",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_vacant_qa)

    ca1m_vacant_qa_RGB = Dataset(
        dataset_name="ca1m_vacant_qa_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_vacant_qa_RGB)

    ca1m_vacant_qa_intrinsics = Dataset(
        dataset_name="ca1m_vacant_qa_intrinsics",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_vacant_qa_intrinsics)

    ca1m_vacant_qa_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_vacant_qa_intrinsics_and_depth",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_vacant_qa_intrinsics_and_depth)


    ca1m_vacant_qa_3d = Dataset(
        dataset_name="ca1m_vacant_qa_3d",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_3d_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_vacant_qa_3d)

    ca1m_vacant_qa_3d_RGB = Dataset(
        dataset_name="ca1m_vacant_qa_3d_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_3d_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_vacant_qa_3d_RGB)

    ca1m_vacant_qa_3d_intrinsics = Dataset(
        dataset_name="ca1m_vacant_qa_3d_intrinsics",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_3d_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_vacant_qa_3d_intrinsics)

    ca1m_vacant_qa_3d_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_vacant_qa_3d_intrinsics_and_depth",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_3d_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_vacant_qa_3d_intrinsics_and_depth)








    ca1m_multi_view_qa = Dataset(
        dataset_name="ca1m_multi_view_qa",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_multi_view_qa_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images_multi_view",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_multi_view_only_image",
    )
    add_dataset(ca1m_multi_view_qa)

    ca1m_multi_view_qa_RGB = Dataset(
        dataset_name="ca1m_multi_view_qa_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_multi_view_qa_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images_multi_view"
    )
    add_dataset(ca1m_multi_view_qa_RGB)

    # ### Simulator


    simulator_blender = Dataset(
        dataset_name="simulator_blender",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/Sim_Blender/spatial_feature_only_image",
    )
    add_dataset(simulator_blender)

    simulator_blender_RGB = Dataset(
        dataset_name="simulator_blender_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images",
    )
    add_dataset(simulator_blender_RGB)



    ### RefCOCO (2D Dataset)

    refcoco = Dataset(
        dataset_name="refcoco",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcoco/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/COCO/spatial_feature_only_image",
    )
    add_dataset(refcoco)

    refcoco_RGB = Dataset(
        dataset_name="refcoco_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcoco/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcoco_RGB)

    refcocop = Dataset(
        dataset_name="refcocop",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocop/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/COCO/spatial_feature_only_image",
    )
    add_dataset(refcocop)

    refcocop_RGB = Dataset(
        dataset_name="refcocop_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocop/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocop_RGB)  

    refcocog = Dataset(
        dataset_name="refcocog",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014", 
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/COCO/spatial_feature_only_image",
    )
    add_dataset(refcocog)

    refcocog_RGB = Dataset(
        dataset_name="refcocog_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata_normalized_1000_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocog_RGB)


    ## SAT

    sat = Dataset(
        dataset_name="sat",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/SAT/spatial_feature_multi_view_only_image",
    )
    add_dataset(sat)

    sat_RGB = Dataset(
        dataset_name="sat_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images"
    )
    add_dataset(sat_RGB)
  

    # ### EmbSpatial (Static 2D Dataset)

    embspatial = Dataset(
        dataset_name="embspatial",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/EmbSpatial/spatial_feature_only_image",
    )
    add_dataset(embspatial)

    embspatial_RGB = Dataset(
        dataset_name="embspatial_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images"
    )
    add_dataset(embspatial_RGB)    


    embspatial_random = Dataset(
        dataset_name="embspatial_random",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_random_90_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images_random_90",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/EmbSpatial/spatial_feature_only_image_random_90",
    )
    add_dataset(embspatial_random)

    embspatial_random_RGB = Dataset(
        dataset_name="embspatial_random_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_random_90_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images_random_90"
    )
    add_dataset(embspatial_random_RGB)


    blink_spatial_relation = Dataset(
        dataset_name="blink_spatial_relation",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/BLINK/spatial_feature_only_image",
    )
    add_dataset(blink_spatial_relation)

    blink_spatial_relation_RGB = Dataset(
        dataset_name="blink_spatial_relation_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_spatial_relation_RGB)



    llava_1_5_lrv_mix_965k = Dataset(
        dataset_name="llava_1_5_lrv_mix_965k",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/llava_v1_5_lrv_mix965k.json",
        image_path="/share/project/emllm_mnt.1d/hpfs/baaiei/vlm/robobrain_train_images",
    )
    add_dataset(llava_1_5_lrv_mix_965k)

    # sat_metric_factor_test = Dataset(
    #     dataset_name="sat_metric_factor_test",
    #     dataset_type="geometricdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/metric_factor_test/metadata.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
    #     enable_spatial=True,
    # )
    # add_dataset(sat_metric_factor_test)

    # sat_metric_factor_test_RGB = Dataset(
    #     dataset_name="sat_metric_factor_test_RGB",
    #     dataset_type="geometricdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/metric_factor_test/metadata.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images"
    # )
    # add_dataset(sat_metric_factor_test_RGB)

    ### DROID
    DROID_w_image_RGB = Dataset(
        dataset_name="DROID_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/droid_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/images_640x360"
    )
    add_dataset(DROID_w_image_RGB)

    DROID_w_image = Dataset(
        dataset_name="DROID_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/droid_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/images_640x360",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/DROID/spatial_feature_only_image",
        description="70k SFT data from DROID."
    )
    add_dataset(DROID_w_image)

    DROID_w_image_intrinsics = Dataset(
        dataset_name="DROID_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/droid_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/images_640x360",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/DROID/spatial_feature_w_intrinsics",
        description="70k SFT data w/ intrinsics from DROID."
    )
    add_dataset(DROID_w_image_intrinsics)

    DROID_w_image_intrinsics_and_depth = Dataset(
        dataset_name="DROID_w_image_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/droid_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/images_640x360",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/DROID/spatial_feature_w_intrinsics_and_depth",
        description="70k SFT data w/ depth and intrinsics from DROID."
    )
    add_dataset(DROID_w_image_intrinsics_and_depth)


    ### ShareRobot
    ShareRobot_w_image_RGB = Dataset(
        dataset_name="ShareRobot_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/ShareRobot/sharerobot_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/ShareRobot/images"
    )
    add_dataset(ShareRobot_w_image_RGB)

    ShareRobot_w_image = Dataset(
        dataset_name="ShareRobot_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/ShareRobot/sharerobot_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/ShareRobot/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/ShareRobot/spatial_feature_only_image",
        description="16k SFT data from ShareRobot."
    )
    add_dataset(ShareRobot_w_image)


    ### CA-1M Traj
    ca1m_traj_w_image_RGB = Dataset(
        dataset_name="ca1m_traj_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_trajectory_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
    )
    add_dataset(ca1m_traj_w_image_RGB)

    ca1m_traj_w_image = Dataset(
        dataset_name="ca1m_traj_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_trajectory_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_only_image",
    )
    add_dataset(ca1m_traj_w_image)

    ca1m_traj_w_image_intrinsics = Dataset(
        dataset_name="ca1m_traj_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_trajectory_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics",
    )
    add_dataset(ca1m_traj_w_image_intrinsics)

    ca1m_traj_w_image_intrinsics_and_depth = Dataset(
        dataset_name="ca1m_traj_w_image_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_trajectory_qa_normalized_1000_split_spatial_feature.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_CA1M/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ca1m_traj_w_image_intrinsics_and_depth)


    ### AGIBOT Traj
    agibot_traj_w_image_RGB = Dataset(
        dataset_name="agibot_traj_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/agibot_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/images",
    )
    add_dataset(agibot_traj_w_image_RGB)

    agibot_traj_w_image = Dataset(
        dataset_name="agibot_traj_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/agibot_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/AGIBOT/spatial_feature_only_image",
    )
    add_dataset(agibot_traj_w_image)

    agibot_traj_w_image_intrinsics = Dataset(
        dataset_name="agibot_traj_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/agibot_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/AGIBOT/spatial_feature_w_intrinsics",
    )
    add_dataset(agibot_traj_w_image_intrinsics)



    robotwin_w_image_RGB = Dataset(
        dataset_name="robotwin_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/RoboTwin/robotwin_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/RoboTwin/images",
    )
    add_dataset(robotwin_w_image_RGB)

    robotwin_w_image = Dataset(
        dataset_name="robotwin_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/RoboTwin/robotwin_trajectory_normalized_1000.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/RoboTwin/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/RoboTwin/spatial_feature_only_image",
    )
    add_dataset(robotwin_w_image)

    ### Scannet
    ScanNet_reasoning_template_qa_split_RGB = Dataset(
        dataset_name="ScanNet_reasoning_template_qa_split_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_reasoning_template_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images"
    )
    add_dataset(ScanNet_reasoning_template_qa_split_RGB)

    ScanNet_reasoning_template_qa_split = Dataset(
        dataset_name="ScanNet_reasoning_template_qa_split",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_reasoning_template_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_only_image",
    )
    add_dataset(ScanNet_reasoning_template_qa_split)

    ScanNet_reasoning_template_qa_split_w_image_intrinsics = Dataset(
        dataset_name="ScanNet_reasoning_template_qa_split_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_reasoning_template_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics",
    )
    add_dataset(ScanNet_reasoning_template_qa_split_w_image_intrinsics)

    ScanNet_reasoning_template_qa_split_w_image_intrinsics_and_depth = Dataset(
        dataset_name="ScanNet_reasoning_template_qa_split_w_image_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_reasoning_template_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ScanNet_reasoning_template_qa_split_w_image_intrinsics_and_depth)




    ScanNet_choice_qa_w_image_RGB = Dataset(
        dataset_name="ScanNet_choice_qa_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_choice_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images"
    )
    add_dataset(ScanNet_choice_qa_w_image_RGB)

    ScanNet_choice_qa_w_image = Dataset(
        dataset_name="ScanNet_choice_qa_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_choice_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_only_image",
    )
    add_dataset(ScanNet_choice_qa_w_image)

    ScanNet_choice_qa_w_image_intrinsics = Dataset(
        dataset_name="ScanNet_choice_qa_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_choice_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics",
    )
    add_dataset(ScanNet_choice_qa_w_image_intrinsics)

    ScanNet_choice_qa_w_image_intrinsics_and_depth = Dataset(
        dataset_name="ScanNet_choice_qa_w_image_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_choice_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ScanNet_choice_qa_w_image_intrinsics_and_depth)


    ScanNet_traj_w_image_RGB = Dataset(
        dataset_name="ScanNet_traj_w_image_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_trajectory_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
    )
    add_dataset(ScanNet_traj_w_image_RGB)

    ScanNet_traj_w_image = Dataset(
        dataset_name="ScanNet_traj_w_image",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_trajectory_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_only_image",
    )
    add_dataset(ScanNet_traj_w_image)

    ScanNet_traj_w_image_intrinsics = Dataset(
        dataset_name="ScanNet_traj_w_image_intrinsics",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_trajectory_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics",
    )
    add_dataset(ScanNet_traj_w_image_intrinsics)

    ScanNet_traj_w_image_intrinsics_and_depth = Dataset(
        dataset_name="ScanNet_traj_w_image_intrinsics_and_depth",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_trajectory_qa_normalized_1000_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/images",
        spatial_feature_path="/share/project/emllm_mnt.1d/sfs/baaiei/exact_spatial_feature/3D_ScanNet/spatial_feature_w_intrinsics_and_depth",
    )
    add_dataset(ScanNet_traj_w_image_intrinsics_and_depth)
