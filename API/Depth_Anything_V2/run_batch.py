import os
import glob
import numpy as np
import multiprocessing

# 1. 读取所有图片路径
IMG_DIR = "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/images_random"  # 你的图片文件夹
all_images = glob.glob(os.path.join(IMG_DIR, "**/*"), recursive=True)
all_images = [img for img in all_images if img.lower().endswith((".jpg", ".png", ".jpeg"))]

print(len(all_images))

# 2. 确保任务均匀分配给 32 个进程
# NUM_PROCESSES = 24
NUM_PROCESSES = 3
image_chunks = np.array_split(all_images, NUM_PROCESSES)

# 3. 生成任务列表，每个进程一个图片子集
tasks = [(i, chunk.tolist()) for i, chunk in enumerate(image_chunks)]

# 处理图片
OUTDIR = f"/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/depths_random"
os.makedirs(OUTDIR, exist_ok=True)

import argparse
import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2 # 确保你的模型类正确导入

def process_images(gpu_id, process_id, filenames):
    """ 每个进程运行的推理任务 """
    DEVICE = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    # 选择对应的模型
    encoder_type = "vitl"  # 你可以改成 vits/vitb/vitg
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 加载模型
    model = DepthAnythingV2(**model_configs[encoder_type])
    model.load_state_dict(torch.load(f'/share/project/zhouenshen/hpfs/ckpt/depthanything/depth_anything_v2_{encoder_type}.pth', map_location='cpu'))
    model.to(DEVICE).eval()



    for filename in tqdm(filenames, desc=f"GPU {gpu_id} Process {process_id}", unit="image"):
        raw_image = cv2.imread(filename)
        depth = model.infer_image(raw_image, input_size=518, device=DEVICE)

        # 归一化
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        
        # 保存
        cv2.imwrite(os.path.join(OUTDIR, os.path.splitext(os.path.basename(filename))[0] + '.png'), depth)
    
    print(f"Process {process_id} on GPU {gpu_id} finished.")

if __name__ == "__main__":
    # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    processes = []
    # NUM_GPUS = 8
    NUM_GPUS = 1
    PROCESSES_PER_GPU = 3

    for i, (process_id, image_list) in enumerate(tasks):
        gpu_id = i % NUM_GPUS  # 轮流分配 GPU
        p = multiprocessing.Process(target=process_images, args=(gpu_id, process_id, image_list))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()  # 等待所有进程结束