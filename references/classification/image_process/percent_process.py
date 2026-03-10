import os
import random
import shutil
from pathlib import Path


def generate_replaced_datasets():
    # 原始 train_B 数据集路径
    src_dir = Path("/root/autodl-tmp/cifar-100-images/train_B")

    # 需要替换的比例
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    # 检查原路径是否存在
    if not src_dir.exists():
        print(f"错误：找不到路径 {src_dir}，请检查当前工作目录。")
        return

    for ratio in ratios:
        ratio_int = int(ratio * 100)
        # 为当前比例创建新的数据集文件夹，例如 train_B_20
        target_dir = Path(f"/root/autodl-tmp/cifar-100-images/train_B_{ratio_int}")

        print(f"\n=====================================")
        print(f"开始生成 {ratio_int}% 替换数据集: {target_dir}")

        # 1. 完整复制一份 train_B 到目标文件夹作为基础
        # 如果已经存在旧的生成文件夹，先删除以防干扰
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(src_dir, target_dir)

        # 2. 遍历目标文件夹中的 100 个类别子文件夹
        # os.listdir 配合 is_dir 确保只处理目录
        class_folders = [d for d in target_dir.iterdir() if d.is_dir()]

        replaced_total_count = 0

        for class_path in class_folders:
            # 获取该类别下的所有图片文件（假设图片为 .png 或 .jpg）
            images = list(class_path.glob("*.*"))
            num_total = len(images)

            if num_total == 0:
                continue

            # 计算该类需要替换的图片数量
            num_replace = int(num_total * ratio)
            if num_replace == 0:
                continue

            # 3. 【修改点】先从该类中随机挑选出唯一的一张"源图片"
            source_img_path = random.choice(images)

            # 将源图片从列表中剔除，剩下的作为"待覆盖候选池"
            # 这样就能 100% 杜绝自己覆盖自己的报错
            candidates = [img for img in images if img != source_img_path]

            # 确定实际需要替换的数量 (防止在 100% 替换时，抽样数量大于候选池数量导致报错)
            actual_replace_count = min(num_replace, len(candidates))

            # 从候选池中抽出最终要被覆盖的"目标图片"
            images_to_replace = random.sample(candidates, actual_replace_count)

            # 4. 执行同类替换
            for target_img_path in images_to_replace:
                # 物理覆盖：将唯一的 source 的图片内容 拷贝给 target
                shutil.copyfile(source_img_path, target_img_path)

            replaced_total_count += actual_replace_count

        print(f"完成！总共在该数据集中替换了 {replaced_total_count} 张图片。")


if __name__ == "__main__":
    generate_replaced_datasets()