#!/bin/bash

DEFAULT_RUN_NAME="stage1_9tile"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=8
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

STAGE_PATH=${1:-"/home/zhouenshen/ckpt/vlm/NVILA-Lite-2B-depth"}
# STAGE_PATH=${1:-"/home/zhouenshen/code/VILA/runs/train/NVILA-2B-Lite-depth-align-20250405_161452/model"}
DATA_MIXTURE=${2:-"sat_176k"}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OUTPUT_DIR=${3:-"runs/train/NVILA-2B-Lite-depth-align-${TIMESTAMP}"}

source scripts/setups/train.sh

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --chat_template qwen2 \
        --data_mixture $DATA_MIXTURE \
        --vision_tower /home/zhouenshen/code/VILA/ckpt/pretrain_weights/paligemma-siglip-so400m-patch14-448 \
        --depth_tower /home/zhouenshen/code/VILA/ckpt/pretrain_weights/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --depth_projector mlp_downsample_3x3_fix \
        --enable_depth True \
        --use_depth_tower True \
        --tune_vision_tower False \
        --tune_mm_projector False \
        --tune_language_model False \
        --tune_depth_tower False \
        --tune_depth_projector True \
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
        --save_steps 10 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 8 \
        --report_to wandb
