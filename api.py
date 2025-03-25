


import argparse
import json
import os
import torch
import base64
import uuid
import cv2
import numpy as np

from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video, Depth
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat


######################## Flask
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = '/home/zhouenshen/tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
######################## Flask

######################## Depth Anything
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
depth_encoder = 'vitl'
depth_input_size = 518
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
depth_anything = DepthAnythingV2(**model_configs[depth_encoder])
depth_anything.load_state_dict(torch.load(
    f'/home/zhouenshen/ckpt/depthanything/depth_anything_v2_{depth_encoder}.pth',
    map_location='cpu'
))
depth_anything = depth_anything.to(DEVICE).eval()
########################

######################## VLM
# vlm_model_path = '/home/zhouenshen/code/VILA/ckpt/pretrain_weights/NVILA-15B'
vlm_model_path = '/home/zhouenshen/code/VILA/runs/train/NVILA-8B-depth-align-tower/model'
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
        "depth_url": [...],   # base64 字符串列表, 可选
        "enable_depth": 0/1,  # 0 表示不需要深度图, 1 表示需要深度图
        "text": ""            # 文本字符串，表示问题或提示
    }
    """

    # 从请求中获取 JSON 参数
    data = request.get_json()

    # 按照协议获取对应字段
    image_urls = data.get("image_url", [])
    depth_urls = data.get("depth_url", [])
    enable_depth = data.get("enable_depth", 0)
    text = data.get("text", "")


    # 1. 解码或保存图像
    image_files = [decode_base64_to_file(img_b64, prefix="image") for img_b64 in image_urls]

    # 2. 根据 enable_depth 决定是否读取或生成深度图
    depth_files = []
    if enable_depth == 1:
        if len(depth_urls) > 0:
            # 如果已经提供了 depth_url，则断言长度匹配
            assert len(depth_urls) == len(image_urls), "Depth URL数量与Image URL数量不匹配"
            depth_files = [decode_base64_to_file(dp_b64, prefix="depth") for dp_b64 in depth_urls]
        else:
            # 未提供 depth_url，需要根据图片自动生成深度图
            for img_f in image_files:
                raw_image = cv2.imread(img_f)
                # 使用深度模型推理
                depth = depth_anything.infer_image(raw_image, input_size=depth_input_size, device=DEVICE)

                # 归一化并转为 8 bit
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                # 转为三通道，方便后续保存
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

                # 保存深度图
                depth_file = f"depth_{uuid.uuid4().hex}.png"
                cv2.imwrite(depth_file, depth)
                depth_files.append(depth_file)
                print(f"Depth file saved to {depth_file}")

    # 3. 组织 prompt，依次将图片、深度图及文本放入
    prompt = []
    # 将所有图片放入 prompt
    for img_f in image_files:
        if any(img_f.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            prompt.append(Image(img_f))
        elif any(img_f.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
            prompt.append(Video(img_f))
        else:
            raise ValueError(f"Unsupported media type: {img_f}")

    # 如果需要深度图，则将深度图也放到 prompt
    if enable_depth == 1 and depth_files:
        for dp_f in depth_files:
            prompt.append(Depth(dp_f))

    # 如果有文本，就把文本也加到 prompt
    if text:
        prompt.append(text)

    # 4. 调用模型生成内容 
    answer = vlm_model.generate_content(prompt)

    # 可以在控制台打印也可以直接返回
    print(colored(answer, "cyan", attrs=["bold"]))


    # 5. 清理临时文件
    for img_f in image_files:
        os.remove(img_f)
    for dp_f in depth_files:
        os.remove(dp_f)

    response = jsonify({'result': 1, 'answer': answer})

    response.headers.set('Content-Type', 'application/json')

    return response


# 如果仅在脚本方式运行，则启动 Flask
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=25547)
    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port)