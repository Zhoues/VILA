from query_model import query_server, query_gemini_2_5_pro
import cv2
import numpy as np
import re, json, os
from PIL import Image
def json2pts(json_text):
    # 去除 markdown 代码块标记（```json ... ```)
    json_cleaned = re.sub(r"^```json\n|\n```$", "", json_text.strip())

    try:
        data = json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return np.empty((0, 2), dtype=int)

    points = []
    for item in data:
        if "point" in item and isinstance(item["point"], list) and len(item["point"]) == 2:
            y_norm, x_norm = item["point"]  # 输入是 (y, x)
            # 归一化范围是 0~1000
            x = x_norm / 1000.0
            y = y_norm / 1000.0
            points.append((x, y))  # 返回 (x, y)
    return np.array(points)

def resize_and_save_image(image_path):
    """
    读取指定绝对路径的图片，将其尺寸缩放为原来的一半，
    保存在当前工作目录下，并返回新图片的绝对路径。
    """
    if not os.path.isabs(image_path):
        print(f"错误: 输入的路径 '{image_path}' 不是一个绝对路径。")
        return None

    if not os.path.exists(image_path):
        print(f"错误: 文件 '{image_path}' 不存在。")
        return None

    try:
        # 1. 读取绝对路径的图片
        img = Image.open(image_path)
        print(f"成功读取图片: {image_path}")
        print(f"原始尺寸: {img.size} (width, height)")

        # 获取原始尺寸
        original_width, original_height = img.size

        # 计算新尺寸（原来的一半）
        new_width = original_width // 2
        new_height = original_height // 2
        print(f"新尺寸 (原来的一半): ({new_width}, {new_height})")

        # 2. resize 为原来的二分之一
        resized_img = img.resize((new_width, new_height))

        # 3. 保存在当前文件夹下面
        # 获取当前工作目录的绝对路径
        current_directory = os.getcwd()

        # 获取原文件名和扩展名
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)

        # 构建新的文件名 (例如: original_image_resized.jpg)
        new_file_name = f"{name}_resized{ext}"

        # 构建新图片的完整保存路径 (当前目录 + 新文件名)
        save_path = os.path.join(current_directory, new_file_name)

        # 检查是否已存在同名文件并提供反馈 (可选)
        if os.path.exists(save_path):
            print(f"注意: 文件 '{new_file_name}' 在当前目录已存在，将被覆盖。")

        # 保存图片
        resized_img.save(save_path)
        print(f"图片已成功保存到: {save_path}")

        # 关闭图片对象释放资源
        img.close()
        resized_img.close()

        # 4. 读取一下这个新图片的绝对路径 (就是上面的 save_path)
        # save_path 已经是绝对路径了，因为 current_directory 是绝对路径
        new_image_absolute_path = save_path
        print(f"新图片的绝对路径是: {new_image_absolute_path}")

        return new_image_absolute_path

    except FileNotFoundError:
        print(f"错误: 文件 '{image_path}' 未找到。请检查路径是否正确。")
        return None
    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None
        
def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg", color=(0, 0, 255), radius=4):
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

# 主代码
test_image_path = "image.jpg"
test_prompt = "Please point to the leftmost mug"
suffix = "Your answer should be formatted as a list of tuples, i.e. [(x1, y1)], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
new_test_image_path = resize_and_save_image(test_image_path)
test_image_paths = [str(new_test_image_path)]

answer = query_server(
    test_image_paths,
    test_prompt + suffix,
    url="http://127.0.0.1:25547",
    enable_depth=1
)

# 转换字符串为坐标点
normalized_points = eval(answer.strip())

# 绘制点并保存图像
denormalize_and_mark(new_test_image_path, normalized_points, output_path="our_result.jpg")

gemini_prompt = "Locate the point of the white object which is behind the alarm clock."
answer = query_gemini_2_5_pro(
    test_image_paths,
    gemini_prompt,
)

# 转换字符串为坐标点
normalized_points = json2pts(answer.strip())

# 绘制点并保存图像
denormalize_and_mark(new_test_image_path, normalized_points, output_path="gemini_result.jpg")