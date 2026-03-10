import torch
import numpy as np
import os
import csv

# 1. 定义特征文件路径和 CSV 保存路径
base_dir = '/root/autodl-tmp/'
csv_save_path = '/root/autodl-tmp/total_volume_results.csv'

feature_files = {
    'train_B': 'train_B_features.pt',
    'train_B_20': 'train_B_20_features.pt',
    'train_B_40': 'train_B_40_features.pt',
    'train_B_60': 'train_B_60_features.pt',
    'train_B_80': 'train_B_80_features.pt',
    'train_B_100': 'train_B_100_features.pt'
}


def calculate_volume(features):
    """
    计算输入特征矩阵的感知流形体积 Vol(Z)
    现在的 features 形状预期为 (25000, 512)
    """
    m, p = features.shape
    if m == 0:
        return 0.0

    # 1. 计算所有 25000 个样本的全局均值
    Z_mean = np.mean(features, axis=0)

    # 2. 全局特征中心化
    Z_hat = features - Z_mean

    # 3. 构建全局正则化协方差矩阵 (512x512)
    I = np.eye(p)
    Sigma_adjusted = I + (1.0 / m) * (Z_hat.T @ Z_hat)

    # 4. 计算体积
    sign, log_det_ln = np.linalg.slogdet(Sigma_adjusted)
    log_det_log2 = log_det_ln / np.log(2.0)

    if sign <= 0:
        print("警告: 协方差矩阵行列式计算异常！")
        return 0.0

    vol_Z = 0.5 * log_det_log2
    return vol_Z


def main():
    print("--- 开始计算各数据集的总体流形体积 ---")

    csv_data = []

    for name, filename in feature_files.items():
        file_path = os.path.join(base_dir, filename)

        # 提取比例
        if name == 'train_B':
            ratio = 0
        else:
            ratio = int(name.split('_')[-1])

        if not os.path.exists(file_path):
            print(f"⚠️ 找不到文件: {file_path}，已跳过。")
            continue

        # 加载数据
        data = torch.load(file_path, weights_only=False)
        all_features = data['features']  # 直接获取全部 25000 个特征

        # 【核心修改点】不再按类别切分，直接将全体特征传入计算
        total_volume = calculate_volume(all_features)

        csv_data.append({
            'Dataset': name,
            'Ratio(%)': ratio,
            'Total_Volume': total_volume
        })

        print(f"[{name.ljust(12)}] 比例: {ratio:>3}% | 总体积 Vol(Z) = {total_volume:.4f}")

    # --- 将结果写入 CSV 文件 ---
    print(f"\n--- 正在将结果保存至 {csv_save_path} ---")

    # 表头去掉了 Std_Volume，因为现在每个数据集只算出一个总值
    fieldnames = ['Dataset', 'Ratio(%)', 'Total_Volume']

    with open(csv_save_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print("✅ CSV 文件保存成功！")


if __name__ == "__main__":
    main()