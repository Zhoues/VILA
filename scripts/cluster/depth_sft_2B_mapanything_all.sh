#!/bin/bash
. /share/project/zhouenshen/hpfs/cache.sh
. /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate /share/project/zhouenshen/miniconda3/envs/vila

export PYTHONPATH=$(pwd)
export WANDB_MODE=offline

export BASE_RUN_NAME="MapAnything-sft-v3"
export STAGE_PATH=/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything-align-v3/model
export VISION_TOWER=/share/project/zhouenshen/hpfs/ckpt/vlm/paligemma-siglip-so400m-patch14-448
export SPATIAL_TOWER=/share/project/zhouenshen/hpfs/ckpt/mapanything/map-anything
export OUTPUT_DIR=/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-${BASE_RUN_NAME}
mkdir -p $OUTPUT_DIR
touch $OUTPUT_DIR/exp.log


# datasets=(
#     reason_template_qa
#     # reason_template_qa_RGB
#     choice_qa
#     # choice_qa_RGB

#     ca1m_reasoning_template_qa_split
#     ca1m_reasoning_template_qa_split_RGB
#     # ca1m_reasoning_template_qa_split_w_intrinsics
#     ca1m_reasoning_template_qa_split_w_intrinsics_and_depth
#     ca1m_choice_qa_split
#     # ca1m_choice_qa_split_RGB
#     # ca1m_choice_qa_split_w_intrinsics
#     ca1m_choice_qa_split_w_intrinsics_and_depth

#     ScanNet_reasoning_template_qa_split
#     ScanNet_reasoning_template_qa_split_RGB
#     # ScanNet_reasoning_template_qa_split_w_image_intrinsics
#     ScanNet_reasoning_template_qa_split_w_image_intrinsics_and_depth

#     ScanNet_choice_qa_w_image
#     # ScanNet_choice_qa_w_image_RGB
#     # ScanNet_choice_qa_w_image_intrinsics
#     ScanNet_choice_qa_w_image_intrinsics_and_depth

#     ca1m_visual_choice_qa
#     ca1m_visual_choice_qa_RGB
#     ca1m_multi_view_qa
#     ca1m_multi_view_qa_RGB

#     ca1m_vacant_qa
#     # ca1m_vacant_qa_RGB
#     ca1m_vacant_qa_intrinsics
#     ca1m_vacant_qa_intrinsics_and_depth
#     ca1m_vacant_qa_3d
#     # ca1m_vacant_qa_3d_RGB
#     ca1m_vacant_qa_3d_intrinsics
#     ca1m_vacant_qa_3d_intrinsics_and_depth


#     simulator_blender
#     # simulator_blender
#     simulator_blender_RGB
#     # simulator_blender_RGB

#     refcoco
#     # refcoco_RGB
#     refcocop
#     # refcocop_RGB
#     refcocog
#     # refcocog_RGB

#     sat
#     # sat_RGB
#     embspatial
#     # embspatial_RGB
#     embspatial_random
#     embspatial_random_RGB
#     blink_spatial_relation
#     blink_spatial_relation_RGB
#     llava_1_5_lrv_mix_965k

#     DROID_w_image_RGB
#     DROID_w_image
#     DROID_w_image_intrinsics
#     DROID_w_image_intrinsics_and_depth
#     ShareRobot_w_image
#     ShareRobot_w_image_RGB
#     ShareRobot_w_image
#     ShareRobot_w_image_RGB
#     ca1m_traj_w_image
#     ca1m_traj_w_image_RGB
#     ca1m_traj_w_image_intrinsics
#     ca1m_traj_w_image_intrinsics_and_depth
#     agibot_traj_w_image
#     agibot_traj_w_image_RGB
#     agibot_traj_w_image_intrinsics
#     robotwin_w_image
#     robotwin_w_image_RGB
#     ScanNet_traj_w_image
#     ScanNet_traj_w_image_RGB
#     ScanNet_traj_w_image_intrinsics
#     ScanNet_traj_w_image_intrinsics_and_depth
# )



datasets=(
    reason_template_qa
    choice_qa

    ca1m_reasoning_template_qa_split
    ca1m_reasoning_template_qa_split_RGB
    ca1m_reasoning_template_qa_split_w_intrinsics_and_depth
    ca1m_choice_qa_split
    ca1m_choice_qa_split_RGB
    ca1m_choice_qa_split_w_intrinsics_and_depth

    ca1m_distance_qa_split
    ca1m_distance_qa_split_RGB
    ca1m_distance_qa_split_w_intrinsics
    ca1m_distance_qa_split_w_intrinsics_and_depth

    ScanNet_reasoning_template_qa_split
    ScanNet_reasoning_template_qa_split_RGB
    ScanNet_reasoning_template_qa_split_w_image_intrinsics_and_depth
    ScanNet_choice_qa_w_image
    ScanNet_choice_qa_w_image_RGB
    ScanNet_choice_qa_w_image_intrinsics_and_depth

    ca1m_visual_choice_qa
    ca1m_visual_choice_qa_RGB
    ca1m_multi_view_qa
    ca1m_multi_view_qa_RGB

    ca1m_vacant_qa
    ca1m_vacant_qa_RGB
    ca1m_vacant_qa_intrinsics
    ca1m_vacant_qa_intrinsics_and_depth
    ca1m_vacant_qa_3d
    ca1m_vacant_qa_3d_RGB
    ca1m_vacant_qa_3d_intrinsics
    ca1m_vacant_qa_3d_intrinsics_and_depth

    simulator_blender
    simulator_blender_RGB

    refcoco
    refcocop
    refcocog

    sat
    sat_RGB
    embspatial
    embspatial_random
    embspatial_random_RGB
    embspatial_random
    embspatial_random_RGB
    blink_spatial_relation
    blink_spatial_relation_RGB
    blink_spatial_relation
    blink_spatial_relation_RGB
    llava_1_5_lrv_mix_965k

    DROID_w_image
    DROID_w_image_RGB
    DROID_w_image_intrinsics_and_depth
    ShareRobot_w_image
    ShareRobot_w_image_RGB
    ShareRobot_w_image
    ShareRobot_w_image_RGB
    ca1m_traj_w_image
    ca1m_traj_w_image_RGB
    ca1m_traj_w_image_intrinsics_and_depth
    agibot_traj_w_image
    agibot_traj_w_image_RGB
    robotwin_w_image
    robotwin_w_image_RGB
    ScanNet_traj_w_image
    ScanNet_traj_w_image_RGB
    ScanNet_traj_w_image_intrinsics_and_depth
)


filtered=()
for x in "${datasets[@]}"; do
  [[ -n "$x" ]] && filtered+=("$x")
done
export DATA_MIXTURE=$(IFS=+; echo "${filtered[*]}")


# training config
export GPUS_PER_NODE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=6
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


torchrun --nnodes=${WORLD_SIZE} --nproc_per_node=${GPUS_PER_NODE} --node_rank=${RANK} \
--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --chat_template qwen2 \
    --data_mixture $DATA_MIXTURE \
    --vision_tower $VISION_TOWER \
    --spatial_tower $SPATIAL_TOWER \
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
    --dataloader_num_workers 16 >>$OUTPUT_DIR/exp.log 2>&1

    
    
#    --report_to wandb >>$OUTPUT_DIR/exp.log 2>&1
    
