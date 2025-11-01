#!/bin/bash

DEFAULT_RUN_NAME="mapanything-reason-template-qa-geo-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=48
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

# STAGE_PATH=${1:-"/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything-reason-template-qa-geo-align/model"}
STAGE_PATH=${1:-"/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything"}
# STAGE_PATH=${1:-"/home/zhouenshen/code/VILA/runs/train/NVILA-8B-depth-align/model"}
# DATA_MIXTURE=${2:-"sat_176k+pixmol_151k"}
# export DATA_MIXTURE="choice_qa_4M+reason_template_qa_5_9M+ca1m_reasoning_template_qa_3_2M_split+ca1m_choice_qa_2_1M_split"

# export DATA_MIXTURE="choice_qa_4M_RGB+reason_template_qa_5_9M_RGB+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M_RGB+refcocop_1_2M_RGB+refcocog_80k_RGB+blink_spatial_relation_RGB+blink_relative_depth_RGB+blink_Object_Localization_RGB+blink_Multi_view_Reasoning_RGB+cv_bench_relation_RGB+cv_bench_depth_RGB+cv_bench_distance_RGB+llava_1_5_lrv_mix_965k"

# export DATA_MIXTURE="simulator_216k+reason_template_qa_5_9M+ca1m_reasoning_template_qa_3_2M_split"

# export DATA_MIXTURE="sat_176k+sat_176k_RGB"
# export DATA_MIXTURE="ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB"
# export DATA_MIXTURE="DROID_w_image+DROID_w_image_RGB+DROID_w_image_intrinsics+DROID_w_image_intrinsics_and_depth+ShareRobot_w_image+ShareRobot_w_image_RGB"
# export DATA_MIXTURE="DROID_w_image+DROID_w_image_RGB+DROID_w_image_intrinsics+DROID_w_image_intrinsics_and_depth"

# export DATA_MIXTURE="ScanNet_w_image+ScanNet_w_image_RGB+ScanNet_w_image_intrinsics+ScanNet_w_image_intrinsics_and_depth"
# export DATA_MIXTURE="ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_w_intrinsics+ca1m_reasoning_template_qa_3_2M_split_w_intrinsics_and_depth"
# export DATA_MIXTURE="ca1m_template_qa_split_RGB+ca1m_template_qa_split+ca1m_template_qa_split_w_intrinsics+ca1m_template_qa_split_w_intrinsics_and_depth"
export DATA_MIXTURE="robotwin_w_image"


export WANDB_MODE=offline

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR=${3:-"runs/train/NVILA-Lite-2B-MapAnything-reason-template-qa-geo-sft"}

source scripts/setups/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --chat_template qwen2 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower /share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448 \
        --spatial_tower /share/project/zhouenshen/hpfs/ckpt/mapanything/map-anything \
        --mm_vision_select_feature cls_patch \
        --spatial_tower_vision_select_feature scale_token_patch \
        --spatial_tower_vision_num_tokens 1369 \
        --mm_projector mlp_downsample_3x3_fix \
        --spatial_projector mlp_downsample_3x3_fix \
        --enable_spatial True \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --tune_spatial_projector True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --report_to wandb
