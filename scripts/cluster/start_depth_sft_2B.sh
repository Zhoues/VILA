#!/bin/bash
. /share/project/zhouenshen/miniconda3/etc/profile.d/conda.sh
conda activate /share/project/zhouenshen/miniconda3/envs/vila
cd /share/project/zhouenshen/hpfs/code/VILA
bash scripts/cluster/depth_sft_2B.sh