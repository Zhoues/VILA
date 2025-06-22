#!/bin/bash
. /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate /share/project/zhouenshen/miniconda3/envs/vila

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

export BASE_RUN_NAME="depth-sft-old_placement+new_simulator_gas1"

export STAGE_PATH=/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-depth-align-old_placement+new_simulator/model
export VISION_TOWER=/share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448
export DEPTH_TOWER=/share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448

# export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_216k+simulator_216k_RGB+simulator_216k+simulator_216k_RGB+blink_spatial_relation+blink_spatial_relation_RGB+embspatial_12k_random+embspatial_12k_random_RGB"
# export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_216k+simulator_216k_RGB+simulator_216k+simulator_216k_RGB+blink_spatial_relation+blink_spatial_relation_RGB+embspatial_12k_random+embspatial_12k_random_RGB"


export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_246k+simulator_246k_RGB+simulator_246k+simulator_246k_RGB+blink_spatial_relation+blink_spatial_relation_RGB"
# export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_246k+simulator_246k_RGB+simulator_246k+simulator_246k_RGB+blink_spatial_relation+blink_spatial_relation_RGB"


export OUTPUT_DIR=/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-${BASE_RUN_NAME}
mkdir -p $OUTPUT_DIR
touch $OUTPUT_DIR/exp.log

# training config
export GPUS_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=6
export GRADIENT_ACCUMULATION_STEPS=1
export MASTER_PORT=25290

# network config
export NCCL_P2P_LEVEL=NVL
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export OMP_NUM_THREADS=4
export ACCELERATE_CPU_AFFINITY=1
export NCCL_IB_HCA=mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107,mlx5_108
ulimit -n 1048576  #增大可打开文件描述符


torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower $VISION_TOWER \
    --depth_tower $DEPTH_TOWER \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample_3x3_fix \
    --depth_projector mlp_downsample_3x3_fix \
    --enable_depth True \
    --use_depth_tower True \
    --tune_vision_tower True \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_depth_tower True \
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
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 >>$OUTPUT_DIR/exp.log 2>&1

    
    
#    --report_to wandb >>$OUTPUT_DIR/exp.log 2>&1
    
