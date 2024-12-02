###
# File: ./main.py
# Created Date: Monday, December 2nd 2024
# Author: Zihan
# -----
# Last Modified: Monday, 2nd December 2024 4:51:18 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import requests
import zipfile
import torch
import torchvision
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
def prepare_tiny_imagenet_data(data_dir):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "train"), transform=train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, "val"), transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4
    )

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
