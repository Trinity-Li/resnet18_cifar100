import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

# 1. 配置参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = '/root/autodl-tmp/resnet18_cifar100_output/checkpoint.pth'  # 模型权重
output_base_path = '/root/autodl-tmp/'  # 特征文件保存的基础目录

# 定义要遍历的数据集目录
dataset_dirs = [
    '/root/autodl-tmp/cifar-100-images/train_B',
    '/root/autodl-tmp/cifar-100-images/train_B_20',
    '/root/autodl-tmp/cifar-100-images/train_B_40',
    '/root/autodl-tmp/cifar-100-images/train_B_60',
    '/root/autodl-tmp/cifar-100-images/train_B_80',
    '/root/autodl-tmp/cifar-100-images/train_B_100'
]


# 2. 定义模型 (保持不变)
def get_feature_extractor(checkpoint_file, num_classes=100):
    model = torchvision.models.resnet18(num_classes=num_classes)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    print(f"正在加载权重: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.fc = nn.Identity()

    model.to(device)
    model.eval()
    return model


# 3. 数据预处理 (保持不变)
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 4. 提取核心逻辑 (修改为接收路径参数)
def extract_features(model, data_dir, save_path):
    if not os.path.exists(data_dir):
        print(f"⚠️ 警告: 找不到路径 {data_dir}，跳过该数据集。")
        return

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    all_features = []
    all_labels = []

    print(f"\n>>> 正在提取 {os.path.basename(data_dir)} 的特征...")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=os.path.basename(data_dir)):
            images = images.to(device)
            features = model(images)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    final_features = np.concatenate(all_features, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    torch.save({
        'features': final_features,
        'labels': final_labels,
        'class_to_idx': dataset.class_to_idx
    }, save_path)

    print(f"✅ 提取完成！特征形状: {final_features.shape}")
    print(f"✅ 结果已保存至: {save_path}")


def main():
    # 只需要在最开始加载一次模型
    print("初始化并加载特征提取器...")
    model = get_feature_extractor(checkpoint_path)

    # 循环遍历每一个数据集目录
    for data_dir in dataset_dirs:
        # 根据文件夹名字自动生成保存的文件名
        # 例如 train_B 生成 train_B_features.pt
        # train_B_20 生成 train_B_20_features.pt
        folder_name = os.path.basename(data_dir)
        save_filename = f"{folder_name}_features.pt"
        save_path = os.path.join(output_base_path, save_filename)

        # 执行提取
        extract_features(model, data_dir, save_path)

    print("\n🎉 所有数据集特征提取完毕！")


if __name__ == '__main__':
    main()