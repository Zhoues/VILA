#!/bin/bash
. /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate /share/project/zhouenshen/miniconda3/envs/vila

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

export BASE_RUN_NAME="depth-align-new_placement+new_simulator-8-nodes"
export STAGE_PATH=/share/project/zhouenshen/hpfs/ckpt/vlm/NVILA-8B-depth
export VISION_TOWER=/share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448
export DEPTH_TOWER=/share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448

# export DATA_MIXTURE="choice_qa_4M+reason_template_qa_5_9M+sat_176k+ca1m_reasoning_template_qa_3_2M_split+ca1m_choice_qa_2_1M_split+ca1m_visual_choice_qa_341k+refcoco_1_2M+refcocop_1_2M+refcocog_80k+sat_176k+blink_spatial_relation+blink_spatial_relation+blink_spatial_relation+blink_relative_depth+blink_Object_Localization+blink_Multi_view_Reasoning+cv_bench_relation+cv_bench_depth+cv_bench_depth+cv_bench_depth+cv_bench_distance+cv_bench_distance+cv_bench_distance"
# export DATA_MIXTURE="choice_qa_4M+reason_template_qa_5_9M+sat_176k+ca1m_reasoning_template_qa_3_2M_split+ca1m_choice_qa_2_1M_split+ca1m_visual_choice_qa_341k+refcoco_1_2M+refcocop_1_2M+refcocog_80k+sat_176k+blink_spatial_relation+blink_relative_depth+blink_Object_Localization+blink_Multi_view_Reasoning+blink_all+cv_bench_relation+cv_bench_depth+cv_bench_distance+cv_bench_all+embspatial_127k+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k+simulator_216k+simulator_216k"

# export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_246k+simulator_246k_RGB+simulator_246k+simulator_246k_RGB+blink_spatial_relation+blink_spatial_relation_RGB+embspatial_12k_random+embspatial_12k_random_RGB"
export DATA_MIXTURE="choice_qa_4M+choice_qa_4M_RGB+reason_template_qa_5_9M+reason_template_qa_5_9M_RGB+sat_176k+sat_176k_RGB+ca1m_reasoning_template_qa_3_2M_split+ca1m_reasoning_template_qa_3_2M_split_RGB+ca1m_choice_qa_2_1M_split+ca1m_choice_qa_2_1M_split_RGB+ca1m_visual_choice_qa_341k+ca1m_visual_choice_qa_341k_RGB+refcoco_1_2M+refcoco_1_2M_RGB+refcocop_1_2M+refcocop_1_2M_RGB+refcocog_80k+refcocog_80k_RGB+embspatial_127k+embspatial_127k_RGB+llava_1_5_lrv_mix_965k+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_vacant_qa_231k+ca1m_vacant_qa_231k_RGB+ca1m_multi_view_qa_77k+ca1m_multi_view_qa_77k_RGB+simulator_246k+simulator_246k_RGB+simulator_246k+simulator_246k_RGB+blink_spatial_relation+blink_spatial_relation_RGB+embspatial_12k_random+embspatial_12k_random_RGB"

export OUTPUT_DIR=/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-8B-${BASE_RUN_NAME}
mkdir -p $OUTPUT_DIR
touch $OUTPUT_DIR/exp.log

# training config
export GPUS_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=2
export MASTER_PORT=25190


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

# export NCCL_IB_CUDA_SUPPORT=1
# export NCCL_IB_GID_INDEX=3
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_BLOCKING_WAIT=1
# export NCCL_PORT=25191
# source scripts/setups/train.sh



torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower $VISION_TOWER \
    --depth_tower $DEPTH_TOWER \
    --dynamic_s2 True \
    --s2_scales "448,896,1344" \
    --s2_max_split_size 448 \
    --s2_resize_output_to_scale_idx -1 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --depth_projector mlp_downsample \
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
    --image_aspect_ratio dynamic_s2 \
    --bf16 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to wandb >>$OUTPUT_DIR/exp.log 2>&1
    
