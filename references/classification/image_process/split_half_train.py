import os
import random
import shutil
from tqdm import tqdm


def split_dataset_half(train_dir, unused_dir, seed=42):
    """
    将 train 目录下的每个类别图片随机分成两半，把其中一半移走。
    """
    # 设置随机种子，保证每次运行划分的结果是一致的
    random.seed(seed)

    # 确保备用目录存在
    os.makedirs(unused_dir, exist_ok=True)

    # 获取所有的类别文件夹 (CIFAR-100 有 100 个类)
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    total_moved = 0
    total_kept = 0

    print("正在随机平分数据...")
    for cls in tqdm(classes):
        cls_dir_train = os.path.join(train_dir, cls)
        cls_dir_unused = os.path.join(unused_dir, cls)

        os.makedirs(cls_dir_unused, exist_ok=True)

        # 获取该类别下所有的图片文件
        images = [img for img in os.listdir(cls_dir_train) if img.endswith('.png')]

        # 打乱图片顺序
        random.shuffle(images)

        # 计算一半的数量（CIFAR-100 每个类原来有 500 张，一半就是 250 张）
        half_idx = len(images) // 2

        # 切片，分出要移走的一半
        images_to_move = images[:half_idx]
        images_to_keep = images[half_idx:]

        total_moved += len(images_to_move)
        total_kept += len(images_to_keep)

        # 将选出的那一半移动到备用文件夹 (B份)
        for img in images_to_move:
            src_path = os.path.join(cls_dir_train, img)
            dst_path = os.path.join(cls_dir_unused, img)
            shutil.move(src_path, dst_path)

    print(f"\n划分完成！")
    print(f"保留在原 train 文件夹的图片数 (A份): {total_kept}")
    print(f"移走到 train_B 文件夹的图片数 (B份): {total_moved}")


# --- 执行划分 ---
# 指向你刚刚转换好的图片目录
original_train_dir = '/root/autodl-tmp/cifar-100-images/train'
# B份存放的位置（备用，不会参与训练）
b_split_dir = '/root/autodl-tmp/cifar-100-images/train_B'

split_dataset_half(original_train_dir, b_split_dir)