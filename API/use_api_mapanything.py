import argparse
import re
import json
import cv2
import numpy as np
from query_model import query_server
from PIL import Image
from PIL import Image, ImageDraw
import os
import cv2

def draw_diamond(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x + size, center_y),
        (center_x, center_y + size),
        (center_x - size, center_y)
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)

def draw_triangle(draw, center_x, center_y, size, fill_color, outline_color=(255, 255, 255)):
    points = [
        (center_x, center_y - size),
        (center_x - size, center_y + size),
        (center_x + size, center_y + size)
    ]
    draw.polygon(points, fill=fill_color, outline=outline_color, width=2)


def interpolate_color(start_color, end_color, ratio):
    r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
    g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
    b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
    return (r, g, b)

def denormalize_and_mark(image_path, normalized_points, output_path="output.jpg",
                         color=(244, 133, 66), radius=12, border_color=(255, 255, 255), border_thickness=2):
    """
    Denormalizes normalized points and marks them on the image with a colored circle and white border.
    
    Args:
        image_path (str): Path to the input image.
        normalized_points (list of tuple): List of (x, y) in normalized coordinates [0, 1].
        output_path (str): Where to save the annotated image.
        color (tuple): BGR color of the inner circle.
        radius (int): Radius of the inner circle.
        border_color (tuple): BGR color of the circle's white border.
        border_thickness (int): Thickness of the border around the circle.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = image.shape[:2]

    for nx, ny in normalized_points:
        x = int(nx * width)
        y = int(ny * height)
        # Draw outer white border
        cv2.circle(image, (x, y), radius + border_thickness, border_color, thickness=-1)
        # Draw inner colored circle
        cv2.circle(image, (x, y), radius, color, thickness=-1)

    cv2.imwrite(output_path, image)
    print(f"Saved annotated image to: {output_path}")


def visualize_trajectory_and_points(image_path, pred_trajectory, save_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # 注释掉绘制ans_trajectory（绿色线）的代码
    # if len(ans_trajectory) > 1:
    #     for i in range(len(ans_trajectory) - 1):
    #         start_point = tuple(ans_trajectory[i])
    #         end_point = tuple(ans_trajectory[i + 1])
    #         draw.line([start_point, end_point], fill=(0, 255, 0), width=2)
    #     
    #     start_x, start_y = ans_trajectory[0]
    #     draw.ellipse([(start_x-6, start_y-6), (start_x+6, start_y+6)], fill=(0, 255, 0), outline=(255, 255, 255), width=1)
    #     
    #     end_x, end_y = ans_trajectory[-1]
    #     draw.ellipse([(end_x-6, end_y-6), (end_x+6, end_y+6)], fill=(0, 255, 0), outline=(255, 255, 255), width=1)

    absolute_points = []
    for point in pred_trajectory:
        absolute_points.append((int(point[0] * image.width / 1000), int(point[1] * image.height / 1000)))
    
    if len(absolute_points) > 1:
        start_color = (255, 0, 0)
        end_color = (0, 0, 255)
        
        for i in range(len(absolute_points) - 1):
            ratio = i / (len(absolute_points) - 1)
            line_color = interpolate_color(start_color, end_color, ratio)
            
            start_point = tuple(absolute_points[i])
            end_point = tuple(absolute_points[i + 1])
            draw.line([start_point, end_point], fill=line_color, width=3)
        
        start_x, start_y = absolute_points[0]
        draw_diamond(draw, int(start_x), int(start_y), 8, (255, 0, 0))
        
        end_x, end_y = absolute_points[-1]
        draw_triangle(draw, int(end_x), int(end_y), 8, (0, 0, 255))
    
    image.save(save_path)
    return save_path

"""
python use_api_mapanything.py \
    --image_path "/share/project/zhouenshen/hpfs/code/RoboRefer/assets/tabletop.jpg" \
    --prompt "Move the leftmost cup around the nearest apple to the robot, then place it to the right of the rightmost hamburger." \
    --enable_spatial 1 \
    --output_path our_result.jpg \
    --url http://127.0.0.1:25548


python use_api_mapanything.py \
    --image_path "/share/project/zhouenshen/sfs/dataset/3D/scannet_filtered/scene0000_00/00000/00000.jpg" \
    --prompt "Place the black controler at right which is on the white table to the left of the corner of the white table." \
    --enable_spatial 1 \
    --output_path our_result.jpg \
    --url http://127.0.0.1:25548

python use_api_mapanything.py \
    --image_path "/share/project/zhouenshen/sfs/dataset/3D/cubifyanything/filter_step_20/47204858/1618826467583/wide/image.png" \
    --prompt "Place the left bottle to the right of the faucet." \
    --enable_spatial 1 \
    --output_path our_result.jpg \
    --url http://127.0.0.1:25548

python use_api_mapanything.py \
    --image_path "/share/project/zhouenshen/hpfs/code/benchmark/result/VisualTraceBench/saved_images/TraceSpatial_images/50.png" \
    --prompt "Please predict 2D object-centric waypoints to complete the task successfully. The task is \"Pick up the the lid of the red pot to the far front side of the the rightmost red object\"." \
    --enable_spatial 1 \
    --output_path our_result.jpg \
    --url http://127.0.0.1:25547

    42445697/62691302989875


Please predict 2D object-centric waypoints to complete the task successfully. The task is \"pick up the lid of the red pot, and move it to the front of the rightmost red object\".

pick up the black controler at right which is on the white table and place it into the corner of the white table which is closest to the black chair on the left.

Place the black controler at right which is on the white table to the left of the corner of the white table which is closest to the black chair.

pick up the lid of the red pot, and move it to the front of the rightmost red object.

Pick up the the lid of the red pot to the front side of the the rightmost red object.
"""



def main():
    parser = argparse.ArgumentParser(description="Query and annotate image with points.")
    parser.add_argument("--image_path", type=str, default="image.jpg", help="Path to input image")
    parser.add_argument("--intrinsics", type=str, default=None, help="Path to intrinsics file")
    parser.add_argument("--depth_z", type=str, default=None, help="Path to depth_z file")
    parser.add_argument("--prompt", type=str, default="Please point to the leftmost mug", help="Prompt for the model")
    parser.add_argument("--enable_spatial", type=int, default=1, help="Whether to enable spatial")
    parser.add_argument("--output_path", type=str, default="our_result.jpg", help="Path to save output image")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:25547", help="Server URL for query")

    args = parser.parse_args()

    # 3D 尾缀
    # suffix = " Your answer should be formatted as a list of tuples, i.e., [(x1, y1, d1), (x2, y2, d2), ...], where each tuple contains the x and y coordinates and the depth of the point."
    # 2D 尾缀
    suffix = " Your answer should be formatted as a list of tuples, i.e., [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point."
    
    test_image_paths = [args.image_path]

    if args.intrinsics:
        intrinsics = json.load(open(args.intrinsics))
    else:
        intrinsics = None

    if args.depth_z:
        test_depth_z_paths = [args.depth_z]
    else:
        test_depth_z_paths = None

    answer = query_server(
        test_image_paths,        
        args.prompt + suffix,
        # args.prompt,
        intrinsics=intrinsics,
        depth_z_paths=test_depth_z_paths,
        url=args.url,
        enable_spatial=int(args.enable_spatial)
    )

    print(answer)
    normalized_points = eval(answer.strip())
    # denormalize_and_mark(args.image_path, normalized_points, output_path=args.output_path)
    visualize_trajectory_and_points(args.image_path, normalized_points, args.output_path)
if __name__ == "__main__":
    main()