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

    ### OpenImage (2D Dataset)

    template_qa_4_7M = Dataset(
        dataset_name="template_qa_4_7M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_template_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="4.7 M SFT data w/ depth from OpenImage."
    )
    add_dataset(template_qa_4_7M)

    template_qa_4_7M_RGB = Dataset(
        dataset_name="template_qa_4_7M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_template_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(template_qa_4_7M_RGB)




    choice_qa_4M = Dataset(
        dataset_name="choice_qa_4M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="4 M SFT data w/ depth from OpenImage."
    )
    add_dataset(choice_qa_4M)

    choice_qa_4M_RGB = Dataset(
        dataset_name="choice_qa_4M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(choice_qa_4M_RGB)




    reason_qa_1_2M = Dataset(
        dataset_name="reason_qa_1_2M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="1.2 M SFT data w/ depth from OpenImage."
    )
    add_dataset(reason_qa_1_2M)

    reason_qa_1_2M_RGB = Dataset(
        dataset_name="reason_qa_1_2M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(reason_qa_1_2M_RGB)




    reason_template_qa_5_9M = Dataset(
        dataset_name="reason_template_qa_5_9M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="5.9 M SFT data w/ depth from OpenImage."
    )
    add_dataset(reason_template_qa_5_9M)

    reason_template_qa_5_9M_RGB = Dataset(
        dataset_name="reason_template_qa_5_9M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(reason_template_qa_5_9M_RGB)


    reason_template_qa_5_9M_split = Dataset(
        dataset_name="reason_template_qa_5_9M_split",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive",
        depth_path="/home_sfs//zhouenshen/dataset/OpenImage/train_depth",
        description="5.9 M SFT data w/ depth from OpenImage."
    )
    add_dataset(reason_template_qa_5_9M_split)

    reason_template_qa_5_9M_split_RGB = Dataset(
        dataset_name="reason_template_qa_5_9M_split_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/positive"
    )
    add_dataset(reason_template_qa_5_9M_split_RGB)





    ### CA-1M (3D Dataset)

    ca1m_reasoning_template_qa_3_2M_split = Dataset(
        dataset_name="ca1m_reasoning_template_qa_3_2M_split",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="3.3 M SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_reasoning_template_qa_3_2M_split)

    ca1m_reasoning_template_qa_3_2M_split_RGB = Dataset(
        dataset_name="ca1m_reasoning_template_qa_3_2M_split_RGB",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_reasoning_template_qa_3_2M_split_RGB)
    
    

    ca1m_choice_qa_2_1M_split = Dataset(
        dataset_name="ca1m_choice_qa_2_1M_split",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_choice_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="2.1 M SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_choice_qa_2_1M_split)


    ca1m_choice_qa_2_1M_split_RGB = Dataset(
        dataset_name="ca1m_choice_qa_2_1M_split_RGB",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_choice_qa_split.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_choice_qa_2_1M_split_RGB)


    ca1m_visual_choice_qa_341k = Dataset(
        dataset_name="ca1m_visual_choice_qa_341k",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images",
        depth_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/depths",
        description="341k SFT data w/ depth from CA-1M."
    )
    add_dataset(ca1m_visual_choice_qa_341k)

    ca1m_visual_choice_qa_341k_RGB = Dataset(
        dataset_name="ca1m_visual_choice_qa_341k_RGB",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/visual_choice_qa_images"
    )
    add_dataset(ca1m_visual_choice_qa_341k_RGB)

    ca1m_vacant_qa_121k = Dataset(
        dataset_name="ca1m_vacant_qa_121k",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images",
        depth_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/depths",    
        description="121k SFT data w/ depth from CA-1M."    
    )
    add_dataset(ca1m_vacant_qa_121k)

    ca1m_vacant_qa_121k_RGB = Dataset(
        dataset_name="ca1m_vacant_qa_121k_RGB",
        dataset_type="spatialdataset",   
        data_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa.json",
        image_path="/home_sfs/zhouenshen/dataset/3D/cubifyanything/images"
    )
    add_dataset(ca1m_vacant_qa_121k_RGB)


    ### Simulator (2D Dataset)
    simulator_216k = Dataset(
        dataset_name="simulator_216k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Simulator/metadata_split_10.json",
        image_path="/home_sfs/zhouenshen/dataset/Simulator/images",
        depth_path="/home_sfs/zhouenshen/dataset/Simulator/depths",
        description="216k SFT data w/ depth from Simulator."
    )
    add_dataset(simulator_216k)

    simulator_216k_RGB = Dataset(
        dataset_name="simulator_216k_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Simulator/metadata_split_10.json",
        image_path="/home_sfs/zhouenshen/dataset/Simulator/images"
    )
    add_dataset(simulator_216k_RGB)


    ### RefCOCO (2D Dataset)

    refcoco_1_2M = Dataset(
        dataset_name="refcoco_1_2M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcoco/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014",
        depth_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="1.2 M SFT data w/ depth from RefCOCO."
    )
    add_dataset(refcoco_1_2M)

    refcoco_1_2M_RGB = Dataset(
        dataset_name="refcoco_1_2M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcoco/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcoco_1_2M_RGB)

    refcocop_1_2M = Dataset(
        dataset_name="refcocop_1_2M",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcocop/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014",
        depth_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="1.2 M SFT data w/ depth from RefCOCOp."
    )
    add_dataset(refcocop_1_2M)

    refcocop_1_2M_RGB = Dataset(
        dataset_name="refcocop_1_2M_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcocop/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocop_1_2M_RGB)  

    refcocog_80k = Dataset(
        dataset_name="refcocog_80k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014", 
        depth_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014_depths",
        description="80k SFT data w/ depth from RefCOCOg."
    )
    add_dataset(refcocog_80k)

    refcocog_80k_RGB = Dataset(
        dataset_name="refcocog_80k_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/Detection/refcocog/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/Detection/coco/train2014"
    )
    add_dataset(refcocog_80k_RGB)
    


    ### SAT (Dynamic 2D Dataset)

    sat_176k = Dataset(
        dataset_name="sat_176k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/SAT/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/SAT/train/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/SAT/train/depths",
        description="176k SFT data w/ depth from SAT."
    )
    add_dataset(sat_176k)

    sat_176k_RGB = Dataset(
        dataset_name="sat_176k_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/SAT/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/SAT/train/images"
    )
    add_dataset(sat_176k_RGB)





    ### EmbSpatial (Static 2D Dataset)

    embspatial_127k = Dataset(
        dataset_name="embspatial_127k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/depths",
        description="127k SFT data w/ depth from EmbSpatial."
    )
    add_dataset(embspatial_127k)

    embspatial_127k_RGB = Dataset(
        dataset_name="embspatial_127k_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/images"
    )
    add_dataset(embspatial_127k_RGB)    


    embspatial_12k_random = Dataset(
        dataset_name="embspatial_12k_random",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/metadata_random.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/images_random",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/depths_random",
        description="12k SFT data w/ depth from EmbSpatial."
    )
    add_dataset(embspatial_12k_random)

    embspatial_12k_random_RGB = Dataset(
        dataset_name="embspatial_12k_random_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/metadata_random.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/EmbSpatial/images_random"
    )
    add_dataset(embspatial_12k_random_RGB)

    ### BLINK (Static 2D Dataset)
    blink_all = Dataset(
        dataset_name="blink_all",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/depths",
        description="SFT data w/ depth from BLINK."
    )
    add_dataset(blink_all)

    blink_all_RGB = Dataset(
        dataset_name="blink_all_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_all_RGB)

    blink_spatial_relation = Dataset(
        dataset_name="blink_spatial_relation",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/depths",
        description="572 SFT data w/ depth from BLINK."
    )
    add_dataset(blink_spatial_relation)

    blink_spatial_relation_RGB = Dataset(
        dataset_name="blink_spatial_relation_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_spatial_relation_RGB)


    blink_relative_depth = Dataset(
        dataset_name="blink_relative_depth",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Relative_Depth.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/depths",
        description="247 SFT data w/ depth from BLINK."
    )
    add_dataset(blink_relative_depth)

    blink_relative_depth_RGB = Dataset(
        dataset_name="blink_relative_depth_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Relative_Depth.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_relative_depth_RGB)


    blink_Object_Localization = Dataset(
        dataset_name="blink_Object_Localization",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Object_Localization.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/depths",
        description="247 SFT data w/ depth from BLINK."
    )
    add_dataset(blink_Object_Localization)

    blink_Object_Localization_RGB = Dataset(
        dataset_name="blink_Object_Localization_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Object_Localization.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_Object_Localization_RGB)


    blink_Multi_view_Reasoning = Dataset(
        dataset_name="blink_Multi_view_Reasoning",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Multi-view_Reasoning.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/depths",
        description="1064 SFT data w/ depth from BLINK."
    )
    add_dataset(blink_Multi_view_Reasoning)

    blink_Multi_view_Reasoning_RGB = Dataset(
        dataset_name="blink_Multi_view_Reasoning_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/metadata_Multi-view_Reasoning.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/BLINK/images"
    )
    add_dataset(blink_Multi_view_Reasoning_RGB)




    ### CV-Bench (Static 2D Dataset)
    cv_bench_all = Dataset(
        dataset_name="cv_bench_all",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/depths",
        description="SFT data w/ depth from CV-Bench."
    )
    add_dataset(cv_bench_all)

    cv_bench_all_RGB = Dataset(
        dataset_name="cv_bench_all_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images"
    )
    add_dataset(cv_bench_all_RGB)

    cv_bench_relation = Dataset(
        dataset_name="cv_bench_relation",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Relation.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/depths",
        description="1300 SFT data w/ depth from CV-Bench."
    )
    add_dataset(cv_bench_relation)

    cv_bench_relation_RGB = Dataset(
        dataset_name="cv_bench_relation_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Relation.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images"
    )
    add_dataset(cv_bench_relation_RGB)


    cv_bench_depth = Dataset(
        dataset_name="cv_bench_depth",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Depth.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/depths",
        description="1200 SFT data w/ depth from CV-Bench."
    )
    add_dataset(cv_bench_depth)

    cv_bench_depth_RGB = Dataset(
        dataset_name="cv_bench_depth_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Depth.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images"
    )
    add_dataset(cv_bench_depth_RGB)


    cv_bench_distance = Dataset(
        dataset_name="cv_bench_distance",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Distance.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images",
        depth_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/depths",
        description="1200 SFT data w/ depth from CV-Bench."
    )
    add_dataset(cv_bench_distance)

    cv_bench_distance_RGB = Dataset(
        dataset_name="cv_bench_distance_RGB",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/metadata_Distance.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/CV-Bench/images"
    )
    add_dataset(cv_bench_distance_RGB)



    ### LLaVA (Static 2D Dataset)

    llava_1_5_mm_align = Dataset(
        dataset_name="llava_1_5_mm_align",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/chat.json",
        image_path="/home_sfs/zhouenshen/dataset/vlm/LLaVA-CC3M-Pretrain-595K/images",
    )
    add_dataset(llava_1_5_mm_align)


    ### PixMol (Static 2D Dataset)

    pixmol_151k = Dataset(
        dataset_name="pixmol_151k",
        dataset_type="spatialdataset",
        data_path="/home_sfs/zhouenshen/dataset/vlm/Pixmo/pixmo_0_10_points_w_counting.json",
        image_path="/home/tangyingbo/dataset/pixmo/pixmo-points/pointing",
        description="151k SFT pointing and counting data from PixMol. Total 1.48M QA pairs."
    )
    add_dataset(pixmol_151k)


    ### General QA (2D Dataset)

    llava_1_5_lrv_mix_965k = Dataset(
        dataset_name="llava_1_5_lrv_mix_965k",
        dataset_type="spatialdataset",
        data_path="/home/vlm/finetune_json/llava_v1_5_lrv_mix965k.json",
        image_path="/home/vlm/train_images",
    )
    add_dataset(llava_1_5_lrv_mix_965k)