from query_model import query_server
import cv2
import numpy as np

def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg", color=(0, 0, 255), radius=8):
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

test_image_paths = [test_image_path]

answer = query_server(
    test_image_paths,
    test_prompt + suffix,
    url="http://127.0.0.1:25543",
    enable_depth=1
)

# 转换字符串为坐标点
normalized_points = eval(answer.strip())

# 绘制点并保存图像
denormalize_and_mark(test_image_path, normalized_points)