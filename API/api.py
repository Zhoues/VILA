import argparse
import json
import os
import torch
import base64
import uuid
import cv2
import numpy as np
import imageio
from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Spatial, Video
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat


######################## Flask
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/share/project/zhouenshen/hpfs/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
######################## Flask

######################## MapAnything
from llava.model.multimodal_spatialencoder.mapanything.mapanything.models.mapanything import MapAnything
from llava.model.multimodal_spatialencoder.mapanything.mapanything.utils.image import preprocess_inputs
# DEVICE = 'cuda:7' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
mapanything_model_path = '/share/project/zhouenshen/hpfs/ckpt/mapanything/map-anything'
mapanything_model = MapAnything.from_pretrained(mapanything_model_path)
# mapanything_model = mapanything_model.to(torch.device(DEVICE)).eval()
mapanything_model = mapanything_model.to(device).eval()
######################## MapAnything

######################## VLM
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-8B-depth-sft-new_placement+new_simulator-8-nodes/model'
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/RoboRefer-8B-SFT'
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything-geo-sft/model'
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-8B-MapAnything-scannet-geo-sft-small/model'
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything-partial-geo-sft/model'
vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/runs/train/NVILA-Lite-2B-MapAnything-sft-v2/model' # ajk 10/20 modify to test full data for msmu
# vlm_model_path = '/share/project/zhouenshen/hpfs/code/VILA/ckpt/pretrain_weights/NVILA-8B'

vlm_conv_mode = 'auto'
# 加载模型
vlm_model = llava.load(vlm_model_path)
# 设置会话模式
clib.default_conversation = clib.conv_templates[vlm_conv_mode].copy()
########################



# 定义一个辅助函数，用于将 base64 字符串解码并保存成本地文件
def decode_base64_to_file(base64_str, prefix="image"):
    filename = f"{UPLOAD_FOLDER}/{prefix}_{uuid.uuid4().hex}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(base64_str))
    return filename

@app.route('/', methods=['GET'])
def index():
    return "Hello, World!"


@app.route('/query', methods=['POST'])
def query():
    """
    该接口接受 JSON 数据，数据示例:
    {
        "image_url": [...],   # base64 字符串列表
        "intrinsics": [...],  # 内参矩阵列表 (3, 3)
        "depth_z": [...],  # 深度图列表 (H, W)
        "enable_spatial": 0/1,  # 0 表示不需要空间信息, 1 表示需要空间信息
        "text": ""            # 文本字符串，表示问题或提示
    }
    """

    data = request.get_json()

    image_paths = data.get("image_paths", [])
    depth_z_paths = data.get("depth_z_paths", None)
    intrinsics = data.get("intrinsics", None)
    enable_spatial = data.get("enable_spatial", 0)
    text = data.get("text", "")

    prompt = []
    for img_f in image_paths:
        if any(img_f.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            prompt.append(Image(img_f))
        elif any(img_f.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
            prompt.append(Video(img_f))
        else:
            raise ValueError(f"Unsupported media type: {img_f}")

    if enable_spatial == 1:
        if depth_z_paths is not None:
            assert len(depth_z_paths) == len(image_paths), "Depth z paths length must be equal to image paths length"

        image_list = []
        for image_path in image_paths:
            image = imageio.v2.imread(image_path)
            image_list.append(image)
        
        if depth_z_paths is not None:
            depth_z_list = []
            for depth_z_path in depth_z_paths:
                depth_z = imageio.v2.imread(depth_z_path).astype(np.float32) if os.path.exists(depth_z_path) else None
                depth_z_list.append(depth_z)
        
        if intrinsics is not None:
            intrinsics = np.array(intrinsics)

        views = []
        for idx, image in enumerate(image_list):
            view = {
                "img": image[:, :, :3], # (H, W, 4) -> (H, W, 3)
                "is_metric_scale": torch.tensor([True]),
            }
            if intrinsics is not None:
                view["intrinsics"] = torch.tensor(intrinsics, dtype=torch.float32)
            if depth_z_paths is not None:
                view["depth_z"] = torch.tensor(depth_z_list[idx], dtype=torch.float32)
            views.append(view)

        views = preprocess_inputs(views, resize_mode="square", size=518)

        predictions = mapanything_model.infer(
            views,
            memory_efficient_inference=False,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=False,
            confidence_percentile=10,
        )


        for prediction in predictions:
            spatial = prediction["spatial_features"] # (1, hidden, base_h * base_w + 1)
            scale = prediction["scale_token"] # (1, hidden, 1)
            concat = torch.cat([spatial, scale], dim=-1) # (1, hidden, base_h * base_w + 1)
            prompt.append(Spatial(spatial_feature=concat))
        
        print(f"Generate Spatial Features Successfully!")

    if text:
        prompt.append(text)

    answer = vlm_model.generate_content(prompt)

    print(colored(answer, "cyan", attrs=["bold"]))

    # for img_f in image_files:
    #     os.remove(img_f)
    # for dp_f in depth_files:
    #     os.remove(dp_f)

    response = jsonify({'result': 1, 'answer': answer})

    response.headers.set('Content-Type', 'application/json')

    return response


# 如果仅在脚本方式运行，则启动 Flask
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=25554)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)