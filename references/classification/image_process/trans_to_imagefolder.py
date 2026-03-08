import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm  # 用于显示进度条


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def convert_cifar100_to_imagefolder(source_dir, target_dir):
    """
    将 CIFAR-100 python 版本转换为 ImageFolder 可读的目录结构
    """
    # 1. 读取类别名称
    meta_path = os.path.join(source_dir, 'meta')
    meta_data = unpickle(meta_path)
    class_names = [name.decode('utf-8') for name in meta_data[b'fine_label_names']]

    # 2. 依次处理训练集(train)和测试集(test)
    for split in ['train', 'test']:
        print(f"正在处理 {split} 集...")
        data_dict = unpickle(os.path.join(source_dir, split))

        images = data_dict[b'data']
        labels = data_dict[b'fine_labels']
        filenames = data_dict[b'filenames']

        # 官方 train.py 中使用的是 'train' 和 'val' 文件夹
        target_split = 'val' if split == 'test' else 'train'

        for i in tqdm(range(len(images))):
            img_flat = images[i]
            label_idx = labels[i]
            class_name = class_names[label_idx]
            filename = filenames[i].decode('utf-8')

            # 将 1D 数组转换为 32x32x3 的图像格式
            img_reshaped = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
            img = Image.fromarray(img_reshaped)

            # 创建目标类别的文件夹
            class_dir = os.path.join(target_dir, target_split, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # 保存图片
            img.save(os.path.join(class_dir, filename))


# --- 执行转换 ---
# 假设你的 cifar-100-python 文件夹在这个路径
source_cifar_dir = '/root/autodl-tmp/cifar-100-python'
# 我们将图片保存到新的 cifar-100-images 文件夹中
target_imagefolder_dir = '/root/autodl-tmp/cifar-100-images'

convert_cifar100_to_imagefolder(source_cifar_dir, target_imagefolder_dir)
print("转换完成！可以开始训练了。")