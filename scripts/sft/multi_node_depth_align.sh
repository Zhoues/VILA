#!/bin/bash
source /home/tanhuajie/miniconda3/bin/activate vila_zhoues

export PYTHONPATH=$(pwd)
export BASE_RUN_NAME="depth-align-mlp-2d+3d"

export STAGE_PATH=/home/zhouenshen/code/VILA/ckpt/pretrain_weights/NVILA-8B-depth

export VISION_TOWER=/home/zhouenshen/code/VILA/ckpt/pretrain_weights/paligemma-siglip-so400m-patch14-448
export DEPTH_TOWER=/home/zhouenshen/code/VILA/ckpt/pretrain_weights/paligemma-siglip-so400m-patch14-448
# export DATA_MIXTURE="template_qa_4_7M+choice_qa_4M+reason_qa_1_2M+sat_176k"
export DATA_MIXTURE="choice_qa_4M+reason_template_qa_5_9M+sat_176k+ca1m_reasoning_template_qa_3_2M_split+ca1m_choice_qa_2_1M_split+ca1m_visual_choice_qa_341k+refcoco_1_2M+refcocop_1_2M+refcocog_80k+sat_176k+blink_spatial_relation+blink_relative_depth+blink_Object_Localization+blink_Multi_view_Reasoning+blink_all+cv_bench_relation+cv_bench_depth+cv_bench_distance+cv_bench_all+embspatial_127k+ca1m_vacant_qa_121k+ca1m_vacant_qa_121k+simulator_216k+simulator_216k"
export OUTPUT_DIR=/home/zhouenshen/code/VILA/runs/train/NVILA-8B-${BASE_RUN_NAME}


export HOSTFILE=/home/zhouenshen/code/VILA/hostfile/hostfile_ours

# training config
export GPUS_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=2
export MASTER_PORT=25990


# network config
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
export OMP_NUM_THREADS=4
export NCCL_IB_HCA=mlx5_0,mlx5_1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_BLOCKING_WAIT=1
export NCCL_PORT=25991
# source scripts/setups/train.sh


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir "$OUTPUT_DIR"
fi

NNODES=`wc -l ${HOSTFILE} | cut -d " " -f1`
MASTER_ADDR=`head -n 1 ${HOSTFILE} | cut -d " " -f1`
echo "Master node: ${MASTER_ADDR}"
i=0
for ip in `cat ${HOSTFILE} | cut -d " " -f1`
do
  echo "Starting node ${i}/${NNODES}: ${ip}"
      ssh $ip \
      "cd ${PWD} && \
      export WANDB_MODE=offline && \
      export ACCELERATE_CPU_AFFINITY=1 && \
      export PYTHONPATH=/home/zhouenshen/code/VILA:$PYTHONPATH && \
      /home/tanhuajie/miniconda3/envs/vila_zhoues/bin/torchrun \
    --nnodes=${NNODES} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${i} \
    --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    llava/train/train_mem.py \
        --deepspeed scripts/zero2.json \
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
        --num_train_epochs 2 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 4000 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 16384 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --report_to wandb >>$OUTPUT_DIR/exp.$ip 2>&1" &
    i=`expr $i + 1`
done
