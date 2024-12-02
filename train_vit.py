import torch
import torch.nn as nn
import torch.optim as optim
from main import (
    download_and_extract,
    prepare_tiny_imagenet_data,
    tiny_imagenet_dir,
    TINY_IMAGENET_URL,
)
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models.vision_transformer import Encoder
import torch.nn.functional as F


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        total_epochs,
        total_steps_per_epoch,
        warmup_start_lr=1e-6,
        base_lr=1e-3,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.total_steps_per_epoch = total_steps_per_epoch
        self.warmup_steps = warmup_epochs * total_steps_per_epoch
        self.total_steps = total_epochs * total_steps_per_epoch
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # 线性 warm up
            lr = self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * (
                self.current_step / self.warmup_steps
            )
        else:
            # cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


class TinyViT(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=8,
        num_classes=200,
        depth=6,
        num_heads=6,
        hidden_dim=384,
    ):
        super(TinyViT, self).__init__()

        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.embed_dim = hidden_dim

        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, self.embed_dim) * 0.02
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification Head
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = (
            self.patch_embed(x).flatten(2).transpose(1, 2)
        )  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, embed_dim]
        x = x + self.pos_embed

        x = self.encoder(x)[:, 0]  # [B, embed_dim] (CLS token output)
        return self.head(x)


# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义训练过程
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 创建带 warm-up 的调度器
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,  # 预热5个epoch
        total_epochs=num_epochs,
        total_steps_per_epoch=len(train_loader),
        warmup_start_lr=1e-6,
        base_lr=learning_rate,
    )

    total_steps = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 调整学习率
            current_lr = scheduler.step()

            running_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], "
                    f"Loss: {running_loss / (step + 1):.4f}, LR: {current_lr:.6f}"
                )

        # 每个 epoch 打印平均损失
        avg_loss = running_loss / total_steps
        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed with Average Loss: {avg_loss:.4f}"
        )

        # 验证
        validate_model(model, val_loader)


# 验证函数
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算验证损失
            val_loss += criterion(outputs, labels).item()

        avg_val_loss = val_loss / len(val_loader)
        print(
            f"Validation Accuracy: {100 * correct / total:.2f}%, "
            f"Validation Loss: {avg_val_loss:.4f}"
        )


# 主函数
def main():
    print("Downloading and preparing Tiny ImageNet...")
    extracted_dir = download_and_extract(TINY_IMAGENET_URL, tiny_imagenet_dir)

    print("Preparing data loaders...")
    train_loader, val_loader = prepare_tiny_imagenet_data(extracted_dir)

    print("Creating TinyViT model...")
    model = TinyViT(num_classes=200).to(device)

    print("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.0001)


if __name__ == "__main__":
    main()
