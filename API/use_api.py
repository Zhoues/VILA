from query_model import query_server, query_gemini_2_5_pro
import cv2
import numpy as np
import re, json, os
from PIL import Image
from datetime import datetime  # <--- 1. 导入datetime模块

# --- 日志文件处理函数 ---
JSON_LOG_FILE = "results_log.json"

def load_log_file(filepath):
    """安全地加载JSON日志文件，如果文件不存在则返回一个空列表。"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"警告: 无法读取或解析 '{filepath}'。将创建一个新的日志。错误: {e}")
            return []
    return []

def save_log_file(filepath, data):
    """将数据以格式化的方式保存到JSON日志文件中。"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"结果已成功更新到: {filepath}")
    except IOError as e:
        print(f"错误: 无法写入文件 '{filepath}'. 错误: {e}")

# --- 您已有的函数 (保持不变) ---
def json2pts(json_text):
    match = re.search(r"```(?:\w+)?\n(.*?)```", json_text, re.DOTALL)
    if not match:
        print("在Gemini结果中未找到有效的代码块。")
        return np.empty((0, 2), dtype=int)
    json_cleaned = match.group(1).strip()
    try:
        data = json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return np.empty((0, 2), dtype=int)  
    points = []
    for item in data:
        if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
            y_norm, x_norm = item["point"]
            x = x_norm / 1000.0
            y = y_norm / 1000.0
            points.append((x, y))
    return np.array(points)

def resize_and_save_image(image_path):
    if not os.path.isabs(image_path):
        print(f"错误: 输入的路径 '{image_path}' 不是一个绝对路径。")
        return None
    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在。")
        return None
    try:
        img = Image.open(image_path)
        original_width, original_height = img.size
        new_width = original_width // 2
        new_height = original_height // 2
        resized_img = img.resize((new_width, new_height))
        current_directory = os.getcwd()
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        new_file_name = f"resize_image/{name}_resized{ext}"
        save_path = os.path.join(current_directory, new_file_name)
        resized_img.save(save_path)
        img.close()
        resized_img.close()
        print(f"图片已成功缩放并保存到: {save_path}")
        return save_path
    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None
        
def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg", color=(0, 0, 255), radius=10):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    height, width = image.shape[:2]
    for nx, ny in normalized_points:
        x = int(nx * width)
        y = int(ny * height)
        cv2.circle(image, (x, y), radius, color, thickness=-1)
    cv2.imwrite(output_path, image)
    print(f"已保存标注图像到: {output_path}")

# --- 主代码 ---

# 1. 加载现有日志
all_results = load_log_file(JSON_LOG_FILE)

# 2. 设置本次运行的参数
test_image_path = "/share/project/zhouenshen/hpfs/code/NIPS-Rebuttal-Benchmark/sampled_images_1000_20250728_005024/0293.jpg"
target = "the building on the right which is the third building from the front to back"
test_prompt = f"Please point to {target}"
suffix = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
# --- 主要改动在这里 ---
# 3. 为本次运行生成唯一标识符
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # <--- 2. 生成唯一时间戳
print(f"本次运行的唯一标识符 (时间戳): {run_timestamp}")

# 3. 准备图片和数据结构
new_test_image_path = resize_and_save_image(test_image_path)
if not new_test_image_path:
    print("图片处理失败，程序退出。")
    exit()

test_image_paths = [str(new_test_image_path)]
image_basename = os.path.splitext(os.path.basename(new_test_image_path))[0]

# 创建当前运行的记录字典
current_run_data = {
    "run_id": run_timestamp, # <--- 新增一个ID字段
    "image_path": new_test_image_path,
    "target_description": target,
    "our_model_output_image": f"./test_output/{image_basename}_{run_timestamp}_our_result.jpg",
    "gemini_model_output_image": f"./test_output/{image_basename}_{run_timestamp}_gemini_result.jpg",
    "our_model_correct": None,
    "gemini_model_correct": None,
    "steps": None
}

print("\n--- 开始查询'Our Model' ---")
our_answer = query_server(
    test_image_paths,
    test_prompt + " " + suffix,
    url="http://127.0.0.1:25547",
    enable_depth=1
)
our_normalized_points = eval(our_answer.strip())
denormalize_and_mark(new_test_image_path, our_normalized_points, output_path=current_run_data["our_model_output_image"])

print("\n--- 开始查询'Gemini Model' ---")
gemini_prompt = f"Locate the point of {target}"
gemini_answer = query_gemini_2_5_pro(
    test_image_paths,
    gemini_prompt,
)
gemini_normalized_points = json2pts(gemini_answer.strip())
denormalize_and_mark(new_test_image_path, gemini_normalized_points, output_path=current_run_data["gemini_model_output_image"])

# 4. 获取用户输入进行评估
print("\n--- 请对结果进行评估 ---")
print(f"Target: {target}")
print(f"请查看图片: {current_run_data['our_model_output_image']} 和 {current_run_data['gemini_model_output_image']}")

# 获取'our'模型评估
while True:
    our_correct = input("'Our'模型的输出是否正确? (y/n): ").lower()
    if our_correct in ['y', 'yes']:
        current_run_data['our_model_correct'] = True
        break
    elif our_correct in ['n', 'no']:
        current_run_data['our_model_correct'] = False
        break
    else:
        print("输入无效，请输入 'y' 或 'n'.")

# 获取'gemini'模型评估
while True:
    gemini_correct = input("'Gemini'模型的输出是否正确? (y/n): ").lower()
    if gemini_correct in ['y', 'yes']:
        current_run_data['gemini_model_correct'] = True
        break
    elif gemini_correct in ['n', 'no']:
        current_run_data['gemini_model_correct'] = False
        break
    else:
        print("输入无效，请输入 'y' 或 'n'.")

# 获取识别步骤数
while True:
    try:
        steps = int(input("确定这个target需要多少步(step)? "))
        current_run_data['steps'] = steps
        break
    except ValueError:
        print("输入无效，请输入一个整数。")

# 5. 更新并保存日志文件
all_results.append(current_run_data)
save_log_file(JSON_LOG_FILE, all_results)

print("\n--- 本次运行记录完成 ---")