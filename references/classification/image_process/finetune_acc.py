import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================= 1. 全局配置参数 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_data_path = "/root/autodl-tmp/cifar-100-images"
val_dir = os.path.join(base_data_path, "val")  # 统一的测试集路径
checkpoint_path = "/root/autodl-tmp/resnet18_cifar100_output/checkpoint.pth"
csv_save_path = "/root/autodl-tmp/finetune_acc_results.csv"

# 包含 60，与您实际的文件夹结构完全对应
ratios = ['B', '20', '40', '60', '80', '100']

# 微调超参数
EPOCHS = 10
BATCH_SIZE = 128
LR = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 8
# =================================================

# ================= 2. 数据预处理 =================
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ================= 3. 模型定义与加载 =================
def get_modified_resnet18(checkpoint_file, num_classes=100):
    model = torchvision.models.resnet18(num_classes=num_classes)

    # 适配 CIFAR 的修改
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 加载预训练权重
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    return model


# ================= 4. 核心训练与验证逻辑 =================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ================= 5. 主循环调度 =================
def main():
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------
    # 【新增】: 在微调开始前，先记录并打印原始 Checkpoint 的准确率
    # -----------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print("🔍 正在评估原始 Checkpoint 在测试集上的初始准确率...")
    base_model = get_modified_resnet18(checkpoint_path)
    _, initial_val_acc = evaluate(base_model, val_loader, criterion)
    print(f"🌟 原始 Checkpoint 初始测试准确率 (0 Epoch): {initial_val_acc:.2f}%")
    print(f"{'=' * 50}\n")

    # 释放一下显存，避免占用
    del base_model
    torch.cuda.empty_cache()
    # -----------------------------------------------------------------

    results = []

    for ratio in ratios:
        dataset_name = f"train_B_{ratio}" if ratio != 'B' else "train_B"
        train_dir = os.path.join(base_data_path, dataset_name)

        if not os.path.exists(train_dir):
            print(f"\n⚠️ 找不到路径 {train_dir}，跳过该比例。")
            continue

        print(f"\n{'=' * 50}")
        print(f"🚀 开始在 {dataset_name} 上进行微调 (Finetune)...")

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

        # 每次从头加载权重
        model = get_modified_resnet18(checkpoint_path)

        optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc = 0.0
        final_val_acc = 0.0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, val_loader, criterion)

            scheduler.step()

            print(f"Epoch [{epoch + 1:02d}/{EPOCHS}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            final_val_acc = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"⭐ {dataset_name} 微调结束！最终测试准确率: {final_val_acc:.2f}%, 最佳准确率: {best_val_acc:.2f}%")

        # 记录结果，新增了 Initial_Test_Acc 列
        results.append({
            'Dataset': dataset_name,
            'Ratio(%)': 0 if ratio == 'B' else int(ratio),
            'Initial_Test_Acc(%)': initial_val_acc,  # 保存刚才测出的基线精度
            'Final_Test_Acc(%)': final_val_acc,
            'Best_Test_Acc(%)': best_val_acc
        })

    # ================= 6. 保存结果到 CSV =================
    print(f"\n--- 正在将所有微调结果保存至 {csv_save_path} ---")
    fieldnames = ['Dataset', 'Ratio(%)', 'Initial_Test_Acc(%)', 'Final_Test_Acc(%)', 'Best_Test_Acc(%)']
    with open(csv_save_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("✅ 全部实验自动化执行完毕！")


if __name__ == '__main__':
    main()