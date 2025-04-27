
export TORCH_USE_CUDA_DSA=1  # Linux/macOS
export TORCH_USE_CUDA_DSA=1  # Linux/macOS

# image description
vila-infer \
    --model-path /home/zhouenshen/code/VILA/ckpt/pretrain_weights/NVILA-8B-depth\
    --conv-mode auto \
    --text "Can you point to the knife? Please provide its 2D coordinates." \
    --media "/home/zhouenshen/code/VILA/demo_images/test_Spatial_Relation_9_1.jpg" \
    --depth "/home/zhouenshen/code/VILA/demo_images/test_Spatial_Relation_9_1.png"
# --model-path /home/zhouenshen/code/VILA/runs/train/NVILA-8B-depth-align-osd+sat-tower-9M/model \
    # --text "Can you point to the brown box at left? Please provide its 2D coordinates." \
# vila-infer \
#     --model-path /home/zhouenshen/code/VILA/runs/train/NVILA-Lite-2B-depth-align-MLP-9M/model \
#     --conv-mode auto \
#     --text "Please describe the image" \
#     --media /home/zhouenshen/code/VILA/demo_images/test_Spatial_Relation_9_1.jpg

# {
#     "id": "68a41c60-095a-4e44-824b-f40657a668f1",
#     "image": "68a41c60-095a-4e44-824b-f40657a668f1.jpg",
#     "depth": "68a41c60-095a-4e44-824b-f40657a668f1.png",
#     "conversations": [
#         {
#         "from": "human",
#         "value": "<image> <depth>\nWhich is below, the woman in white shirt standing in parking lot or the woman in yellow vest carrying box? "
#         },
#         {
#         "from": "gpt",
#         "value": "Positioned lower is woman in white shirt standing in parking lot."
#         },
#         {
#         "from": "human",
#         "value": "Between the woman in white shirt standing in parking lot and the woman in yellow vest carrying box, which one has less height? "
#         },
#         {
#         "from": "gpt",
#         "value": "With less height is woman in yellow vest carrying box."
#         }
#     ]
# }

# 0: 2B
# 1: 2B SFT
# 2: 2B depth align
# 3: 8B SFT all
# 4: 8B depth align
# 5: 8B SFT wo tower
# 6: 8B



# Identify several spots within the vacant space that's between the two mugs. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image.
# In the image, there is a cup. Pinpoint several points within the vacant space situated to the right of the cup. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points.