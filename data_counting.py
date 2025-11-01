import json

data_path_list = [


    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_choice_qa_normalized_1000_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/OpenImage/filter/train_20250307_211637_015_573_filter/osd_reasoning_template_qa_normalized_1000.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_reasoning_template_qa_normalized_1000_split_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/cubifyanything/ca1m_choice_qa_normalized_1000_split_spatial_feature.json"
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_vacant_qa_normalized_1000_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_visual_choice_qa_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_multi_view_qa_spatial_feature.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_reasoning_template_qa_normalized_1000_split.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_choice_qa_normalized_1000_split.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/cubifyanything/ca1m_trajectory_qa_normalized_1000_split_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/DROID/droid_trajectory_normalized_1000.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/AGIBOT/agibot_trajectory_normalized_1000.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/RoboTwin/robotwin_trajectory_normalized_1000.json",
        # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Traj/ShareRobot/sharerobot_trajectory_normalized_1000.json",

    "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/3D/ScanNet_v2/scannet_trajectory_qa_normalized_1000_split.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Simulator/metadata_spatial_feature.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/SAT/metadata_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcoco/metadata_normalized_1000_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocog/metadata_normalized_1000_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/Detection/refcocop/metadata_normalized_1000_spatial_feature.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/llava_v1_5_lrv_mix965k.json",

    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/BLINK/metadata_Spatial_Relation_spatial_feature.json",
    # "/share/project/emllm_mnt.1d/sfs/baaiei/zhouenshen/dataset/vlm/EmbSpatial/metadata_random_90_spatial_feature.json",


]

# 全局统计变量
global_total_metadata = 0
global_total_qa = 0

# 遍历每个文件
for file_path in data_path_list:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        total_metadata = len(data)
        qa_counts = []

        for item in data:
            conversations = item.get("conversations", [])
            qa_count = len(conversations) // 2  # 每两个构成一个 QA 对
            qa_counts.append(qa_count)

        total_qa = sum(qa_counts)
        average_qa = total_qa / total_metadata if total_metadata > 0 else 0
        max_qa = max(qa_counts) if qa_counts else 0
        min_qa = min(qa_counts) if qa_counts else 0

        # 更新全局统计
        global_total_metadata += total_metadata
        global_total_qa += total_qa

        # 输出每个文件的统计信息
        print(f"\n📂 文件名: {file_path}")
        print(f"✅ 总共有 {total_metadata} 条 metadata")
        print(f"💬 总共有 {total_qa} 个 QA 对")
        print(f"📊 每条 metadata 平均有 {average_qa:.2f} 个 QA 对")
        print(f"🔺 最多 QA 数量: {max_qa}")
        print(f"🔻 最少 QA 数量: {min_qa}")

    except Exception as e:
        print(f"\n❌ 处理文件失败: {file_path}")
        print(f"   错误信息: {e}")

# 输出全局统计信息
global_average_qa = global_total_qa / global_total_metadata if global_total_metadata > 0 else 0

print("\n📊📈 全部文件统计汇总")
print(f"    ✅ 总 metadata 数量: {global_total_metadata}")
print(f"    💬 总 QA 对数量: {global_total_qa}")
print(f"    📉 平均每条 metadata 的 QA 数: {global_average_qa:.2f}")