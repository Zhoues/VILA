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
    enable_spatial: bool = field(default=False, metadata={"help": "Whether to use spatial encoder."})

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
    choice_qa_4M = Dataset(
        dataset_name="choice_qa_4M",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/train_depth",
        description="4 M SFT data w/ depth from OpenImage."
    )
    add_dataset(choice_qa_4M)

    choice_qa_4M_RGB = Dataset(
        dataset_name="choice_qa_4M_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(choice_qa_4M_RGB)


    reason_template_qa_5_9M = Dataset(
        dataset_name="reason_template_qa_5_9M",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        # depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/train_depth",
        enable_spatial=True,
        description="5.9 M SFT data w/ depth from OpenImage."
    )
    add_dataset(reason_template_qa_5_9M)

    reason_template_qa_5_9M_RGB = Dataset(
        dataset_name="reason_template_qa_5_9M_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(reason_template_qa_5_9M_RGB)



    ### CA-1M (3D Dataset)

    ca1m_reasoning_template_qa_3_2M_split = Dataset(
        dataset_name="ca1m_reasoning_template_qa_3_2M_split",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="3.3 M SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_reasoning_template_qa_3_2M_split)

    ca1m_reasoning_template_qa_3_2M_split_RGB = Dataset(
        dataset_name="ca1m_reasoning_template_qa_3_2M_split_RGB",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_reasoning_template_qa_3_2M_split_RGB)
    
    

    ca1m_choice_qa_2_1M_split = Dataset(
        dataset_name="ca1m_choice_qa_2_1M_split",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_choice_qa_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="2.1 M SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_choice_qa_2_1M_split)


    ca1m_choice_qa_2_1M_split_RGB = Dataset(
        dataset_name="ca1m_choice_qa_2_1M_split_RGB",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_choice_qa_split.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_choice_qa_2_1M_split_RGB)


    ca1m_visual_choice_qa_341k = Dataset(
        dataset_name="ca1m_visual_choice_qa_341k",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/depths",
        description="341k SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_visual_choice_qa_341k)

    ca1m_visual_choice_qa_341k_RGB = Dataset(
        dataset_name="ca1m_visual_choice_qa_341k_RGB",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images"
    )
    add_dataset(ca1m_visual_choice_qa_341k_RGB)

    ca1m_vacant_qa_121k = Dataset(
        dataset_name="ca1m_vacant_qa_121k",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="121k SFT data w/ depth from CA-1M."    
    )
    add_dataset(ca1m_vacant_qa_121k)

    ca1m_vacant_qa_121k_RGB = Dataset(
        dataset_name="ca1m_vacant_qa_121k_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_vacant_qa_121k_RGB)


    ca1m_vacant_qa_231k = Dataset(
        dataset_name="ca1m_vacant_qa_231k",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_v2.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images",
        enable_spatial=True,
        description="231k SFT data w/ depth from CA-1M."    
    )
    add_dataset(ca1m_vacant_qa_231k)

    ca1m_vacant_qa_231k_RGB = Dataset(
        dataset_name="ca1m_vacant_qa_231k_RGB",
        dataset_type="geometricdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_v2.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_vacant_qa_231k_RGB)

    ca1m_multi_view_qa_77k = Dataset(
        dataset_name="ca1m_multi_view_qa_77k",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_multi_view_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images_multi_view",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/depths_multi_view",    
        description="77k SFT data w/ depth from CA-1M."    
    )
    add_dataset(ca1m_multi_view_qa_77k)

    ca1m_multi_view_qa_77k_RGB = Dataset(
        dataset_name="ca1m_multi_view_qa_77k_RGB",
        dataset_type="spatialdataset",   
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_multi_view_qa.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/images_multi_view"
    )
    add_dataset(ca1m_multi_view_qa_77k_RGB)

    ### Simulator (2D Dataset)
    simulator_216k = Dataset(
        dataset_name="simulator_216k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_split_10.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/depths",
        description="216k SFT data w/ depth from Simulator."
    )
    add_dataset(simulator_216k)

    simulator_216k_RGB = Dataset(
        dataset_name="simulator_216k_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_split_10.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images"
    )
    add_dataset(simulator_216k_RGB)

    simulator_246k = Dataset(
        dataset_name="simulator_246k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_new_split_10.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/depths",
        description="246k SFT data w/ depth from Simulator."
    )
    add_dataset(simulator_246k)

    simulator_246k_RGB = Dataset(
        dataset_name="simulator_246k_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_new_split_10.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/images"
    )
    add_dataset(simulator_246k_RGB)



    ### RefCOCO (2D Dataset)

    refcoco_1_2M = Dataset(
        dataset_name="refcoco_1_2M",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcoco/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="1.2 M SFT data w/ depth from RefCOCO."
    )
    add_dataset(refcoco_1_2M)

    refcoco_1_2M_RGB = Dataset(
        dataset_name="refcoco_1_2M_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcoco/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcoco_1_2M_RGB)

    refcocop_1_2M = Dataset(
        dataset_name="refcocop_1_2M",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocop/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="1.2 M SFT data w/ depth from RefCOCOp."
    )
    add_dataset(refcocop_1_2M)

    refcocop_1_2M_RGB = Dataset(
        dataset_name="refcocop_1_2M_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocop/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocop_1_2M_RGB)  

    refcocog_80k = Dataset(
        dataset_name="refcocog_80k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014", 
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="80k SFT data w/ depth from RefCOCOg."
    )
    add_dataset(refcocog_80k)

    refcocog_80k_RGB = Dataset(
        dataset_name="refcocog_80k_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocog_80k_RGB)

    refcocog_80k = Dataset(
        dataset_name="refcocog_80k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014", 
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="80k SFT data w/ depth from RefCOCOg."
    )
    add_dataset(refcocog_80k)

    refcocog_80k_RGB = Dataset(
        dataset_name="refcocog_80k_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocog_80k_RGB)


    ### SAT (Dynamic 2D Dataset)

    # sat_176k = Dataset(
    #     dataset_name="sat_176k",
    #     dataset_type="spatialdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
    #     depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/depths",
    #     description="176k SFT data w/ depth from SAT."
    # )
    # add_dataset(sat_176k)

    # sat_176k_RGB = Dataset(
    #     dataset_name="sat_176k_RGB",
    #     dataset_type="spatialdataset",
    #     data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata.json",
    #     image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images"
    # )
    # add_dataset(sat_176k_RGB)
  



    ### EmbSpatial (Static 2D Dataset)

    embspatial_127k = Dataset(
        dataset_name="embspatial_127k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/depths",
        description="127k SFT data w/ depth from EmbSpatial."
    )
    add_dataset(embspatial_127k)

    embspatial_127k_RGB = Dataset(
        dataset_name="embspatial_127k_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images"
    )
    add_dataset(embspatial_127k_RGB)    


    embspatial_12k_random = Dataset(
        dataset_name="embspatial_12k_random",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_random.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images_random",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/depths_random",
        description="12k SFT data w/ depth from EmbSpatial."
    )
    add_dataset(embspatial_12k_random)

    embspatial_12k_random_RGB = Dataset(
        dataset_name="embspatial_12k_random_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_random.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images_random"
    )
    add_dataset(embspatial_12k_random_RGB)

    blink_spatial_relation = Dataset(
        dataset_name="blink_spatial_relation",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/depths",
        description="572 SFT data w/ depth from BLINK."
    )
    add_dataset(blink_spatial_relation)

    blink_spatial_relation_RGB = Dataset(
        dataset_name="blink_spatial_relation_RGB",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_spatial_relation_RGB)



    ### RefSpatial
    refSpatial = Dataset(
        dataset_name="refSpatial",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RefSpatial/refspatial_new_sim.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RefSpatial",
        description="SFT data from RefSpatial."
    )
    add_dataset(refSpatial)

    ### General QA (2D Dataset)

    llava_1_5_lrv_mix_965k = Dataset(
        dataset_name="llava_1_5_lrv_mix_965k",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/llava_v1_5_lrv_mix965k.json",
        image_path="/share/project/emllm_mnt.1d/hpfs/baaiei/vlm/robobrain_train_images",
    )
    add_dataset(llava_1_5_lrv_mix_965k)


    ### final dataset
    refspatial_old_placemennt_new_sim = Dataset(
        dataset_name="refspatial_old_placemennt_new_sim",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RoboRefer_train_data/refspatial_old_placement_new_sim.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RoboRefer_train_data",
    )
    add_dataset(refspatial_old_placemennt_new_sim)


    refspatial_old_placemennt_old_sim = Dataset(
        dataset_name="refspatial_old_placemennt_old_sim",
        dataset_type="spatialdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RoboRefer_train_data/refspatial_old_placement_old_sim.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/RoboRefer_train_data",
    )
    add_dataset(refspatial_old_placemennt_old_sim)


    human_qa_4k_RGB = Dataset(
        dataset_name="human_qa_4k_RGB",
        dataset_type="spatialdataset",   
        data_path="/share/project/hanyi/dataset/synthetic_images_with_human/metadata.json",
        image_path="/share/project/hanyi/dataset/synthetic_images_with_human/images_with_human"
    )
    add_dataset(human_qa_4k_RGB)


    sat_176k = Dataset(
        dataset_name="sat_176k",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
        enable_spatial=True,
        description="176k SFT data w/ depth from SAT."
    )
    add_dataset(sat_176k)

    sat_176k_RGB = Dataset(
        dataset_name="sat_176k_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
        enable_spatial=True,
    )
    add_dataset(sat_176k_RGB)

    sat_metric_factor_test = Dataset(
        dataset_name="sat_metric_factor_test",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/metric_factor_test/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images",
        enable_spatial=True,
    )
    add_dataset(sat_metric_factor_test)

    sat_metric_factor_test_RGB = Dataset(
        dataset_name="sat_metric_factor_test_RGB",
        dataset_type="geometricdataset",
        data_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/metric_factor_test/metadata.json",
        image_path="/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/train/images"
    )
    add_dataset(sat_metric_factor_test_RGB)