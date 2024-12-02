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
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
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
            scheduler.step()

            running_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], "
                    f"Loss: {running_loss / (step + 1):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
                )

        # 每个 epoch 打印一次平均损失
        avg_loss = running_loss / total_steps
        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed with Average Loss: {avg_loss:.4f}"
        )

        # 在每个 epoch 结束时验证模型
        validate_model(model, val_loader)

    # 保存模型
    torch.save(model.state_dict(), "./tiny_vit_tiny_imagenet.pth")
    print("Model saved successfully!")


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
