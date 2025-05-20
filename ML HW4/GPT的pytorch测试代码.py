# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 12:39:27 2025

@author: Qingyu Zhang
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# ✅ 使用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 当前 GPU:", torch.cuda.get_device_name(0))

# ✅ 数据增强 + 归一化（更贴近真实训练）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),  # ResNet50 输入要求
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

# ✅ 模型选择：ResNet50
model = models.resnet50(weights=None, num_classes=10).to(device)

# ✅ 损失函数、优化器、学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# ✅ 训练函数（多轮）
def train_model(model, epochs=10):
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = running_loss / len(trainloader)
        acc = evaluate(model)
        train_losses.append(avg_loss)
        test_accuracies.append(acc)

        print(f"✅ Epoch {epoch+1}: Loss={avg_loss:.4f}, Test Accuracy={acc:.2f}%")

    return train_losses, test_accuracies

# ✅ 测试函数
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

# ✅ 正式开练！
start = time.time()
losses, accs = train_model(model, epochs=10)
end = time.time()
print(f"\n🎯 总训练时间：{end - start:.2f} 秒")

# ✅ 可视化训练曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses, label='Train Loss')
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accs, label='Test Accuracy', color='green')
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.show()
