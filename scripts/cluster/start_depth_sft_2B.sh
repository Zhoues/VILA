#!/bin/bash
. /share/project/zhouenshen/hpfs/cache.sh
. /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate /share/project/zhouenshen/miniconda3/envs/vila
cd /share/project/zhouenshen/hpfs/code/VILA
# bash scripts/cluster/depth_sft_2B_traj.sh
bash scripts/cluster/depth_sft_2B_mapanything_all.sh