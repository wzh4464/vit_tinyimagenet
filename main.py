###
# File: ./main.py
# Created Date: Monday, December 2nd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 2nd December 2024 3:54:11 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import requests
import zipfile
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# 确定存储路径
tiny_imagenet_dir = "/backup/TinyImageNet"
os.makedirs(tiny_imagenet_dir, exist_ok=True)

# 数据集 URL
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


# 下载函数
def download_and_extract(url, dest_dir):
    zip_path = os.path.join(dest_dir, "tiny-imagenet-200.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Tiny ImageNet from {url}...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    else:
        print("Tiny ImageNet archive already exists. Skipping download.")

    # 解压数据集
    extracted_dir = os.path.join(dest_dir, "tiny-imagenet-200")
    if not os.path.exists(extracted_dir):
        print("Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
    else:
        print("Tiny ImageNet is already extracted.")
    return extracted_dir


# 数据预处理和加载器
def prepare_tiny_imagenet_data(data_dir, batch_size=64):
    # 数据集路径
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # 验证集的标签映射处理
    val_image_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")
    if os.path.exists(val_annotations_file):
        with open(val_annotations_file, "r") as f:
            annotations = f.readlines()
        annotations = [line.strip().split("\t")[:2] for line in annotations]
        class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate({ann[1] for ann in annotations})
        }
        val_labels = {
            os.path.join(val_image_dir, ann[0]): class_to_idx[ann[1]]
            for ann in annotations
        }

    # 定义数据变换
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),  # Tiny ImageNet 的原始大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 加载数据
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # 为验证集应用手动标签
    if val_labels:
        val_dataset.samples = [
            (img_path, val_labels[img_path])
            for img_path, _ in val_dataset.samples
            if img_path in val_labels
        ]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    return train_loader, val_loader


# 主流程
def main():
    print("Downloading and preparing Tiny ImageNet...")
    extracted_dir = download_and_extract(TINY_IMAGENET_URL, tiny_imagenet_dir)

    print("Preparing data loaders...")
    train_loader, val_loader = prepare_tiny_imagenet_data(extracted_dir)

    print("Data loaders are ready!")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = main()
