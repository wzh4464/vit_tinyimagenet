import torch
import torch.nn as nn
import torch.optim as optim
from main import (
    download_and_extract,
    prepare_tiny_imagenet_data,
    tiny_imagenet_dir,
    TINY_IMAGENET_URL,
)
import timm
import torch.nn.functional as F

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 插值函数调整位置编码
def interpolate_pos_encoding(model, img_size):
    # Extract the patch size correctly
    patch_size = model.patch_embed.patch_size[0]

    old_num_patches = model.patch_embed.num_patches
    model.patch_embed.num_patches = (img_size // patch_size) ** 2

    # Interpolate the positional embeddings
    new_pos_embed = nn.Parameter(
        F.interpolate(
            model.pos_embed[:, 1:]
            .view(1, int(old_num_patches**0.5), int(old_num_patches**0.5), -1)
            .permute(0, 3, 1, 2),
            size=(
                img_size // patch_size,
                img_size // patch_size,
            ),
            mode="bicubic",
        )
        .permute(0, 2, 3, 1)
        .view(1, -1, model.embed_dim)
    )
    model.pos_embed = nn.Parameter(
        torch.cat([model.pos_embed[:, :1], new_pos_embed], dim=1)
    )
    return model


# 定义训练过程
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_steps = len(train_loader)

        print(f"Starting epoch {epoch + 1}/{num_epochs}...")

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{total_steps}], "
                    f"Loss: {running_loss / (step + 1):.4f}"
                )

        # 每个 epoch 打印平均损失
        avg_loss = running_loss / total_steps
        print(
            f"Epoch [{epoch+1}/{num_epochs}] completed with Average Loss: {avg_loss:.4f}"
        )

        # 验证
        validate_model(model, val_loader)

    # 保存模型
    torch.save(model.state_dict(), "./deit_tiny_imagenet.pth")
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

    print("Loading DeiT model...")
    model = timm.create_model(
        "deit_small_patch16_224", pretrained=True, num_classes=200
    )

    # Adjust input size
    model.default_cfg["input_size"] = (3, 64, 64)

    # Update the PatchEmbed layer's img_size
    model.patch_embed.img_size = (64, 64)

    # Interpolate positional encoding
    model = interpolate_pos_encoding(model, img_size=64).to(device)

    print("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.0001)


if __name__ == "__main__":
    main()
