---
course: "planning-optimizing-ai-infrastructure"
chapter: "strategies-for-performance-optimization"
lesson: "practical-applying-mixed-precision-training"
sourceId: 7034
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-5-strategies-for-performance-optimization/practical-applying-mixed-precision-training"
title: "实操练习：应用混合精度训练"
description: "一个动手实验，修改标准的PyTorch训练脚本以使用自动混合精度（AMP），并衡量性能提升。"
order: 7
plots: ["plots/7034-0.json"]
sourceHash: "77784169364661b0deafde04ff9007bc4e2ccc6201e0799fa0515020d13ce7a8"
sourceCorrections: []
---

这个实操练习将指导你修改一个标准的PyTorch训练脚本，使其使用自动混合精度（AMP）来优化性能。我们的目标是衡量由此带来的训练速度和内存消耗方面的改进，亲身展示少量代码修改如何带来大幅性能提升。

### 实验先决条件

为了完成本练习，你将需要一个具备以下条件的环境：

- 一个支持Tensor Core的NVIDIA GPU（伏特、图灵、安培架构或更新版本）。尽管AMP可以在旧款GPU上运行，但$FP16$计算带来的加速在具有专用Tensor Core的硬件上最为明显。
- 已安装PyTorch（`pip install torch torchvision`）。
- 你可以在第3章中配置的云实例上运行此练习，或在符合硬件要求的本地机器上运行。

### 基准：一个标准的FP32训练脚本

首先，我们来建立一个基准。以下脚本设置了一个简单的卷积神经网络 (neural network)（CNN），并使用标准的32位浮点精度（$FP32$）在CIFAR-10数据集上进行训练。这将作为我们的比较点。

将以下代码保存为`fp32_baseline.py`。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 1. 数据加载与准备
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4)

# 2. 一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 3. 标准FP32训练循环
print("开始FP32基准训练...")
start_time = time.time()

for epoch in range(5):  # 循环数据集5次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向 + 反向 + 优化
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:    # 每100个小批量打印一次
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

end_time = time.time()
total_time = end_time - start_time
print(f'FP32训练完成，耗时 {total_time:.2f} 秒')
```

从你的终端运行此脚本：

```bash
python fp32_baseline.py
```

记录总训练时间。你也可以在另一个终端窗口中使用`nvidia-smi`命令监控GPU内存使用情况。这将作为我们的比较基准。

### 应用自动混合精度

为了启用混合精度训练，我们将使用PyTorch内置的`torch.cuda.amp`模块。这需要对我们的训练循环进行两项主要修改：

1. **`torch.cuda.amp.autocast`**: 这是一个上下文 (context)管理器，你将其包裹在模型的前向传播周围。它指示PyTorch为每个操作自动选择最优数据类型（$FP16$或$FP32$）。那些受益于$FP16$且数值稳定的操作，例如Tensor Core上的卷积和全连接层，将以半精度运行。其他需要更高精度的操作，例如规约操作，将保持$FP32$精度。
2. **`torch.cuda.amp.GradScaler`**: 使用$FP16$训练可能导致一个称为梯度下溢的问题。因为$FP16$数据类型范围有限，非常小的梯度值可能变为零，停止学习过程。`GradScaler`通过在反向传播 (backpropagation)前向上缩放损失值来解决此问题。这使得所有产生的梯度都变大，防止它们变为零。在优化器更新权重 (weight)之前，缩放器会将梯度缩回其正确值。

### 适用于AMP的修改脚本

现在，我们来修改脚本以集成AMP。这些修改出乎意料地少，这也证明了PyTorch API设计的出色。

将此新版本保存为`amp_optimized.py`。更改已在注释中突出显示。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 1. 数据加载与准备（此处无改动）
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=4)

# 2. 模型定义（此处无改动）
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# AMP修改1：初始化一个GradScaler
scaler = torch.cuda.amp.GradScaler()

# 3. 修改后的AMP训练循环
print("开始AMP优化训练...")
start_time = time.time()

for epoch in range(5):  # 循环数据集5次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # 梯度清零
        optimizer.zero_grad()

        # AMP修改2：使用autocast包裹前向传播
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # AMP修改3：使用scaler进行反向传播和优化器步进
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % 100 == 99:    # 每100个小批量打印一次
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

end_time = time.time()
total_time = end_time - start_time
print(f'AMP训练完成，耗时 {total_time:.2f} 秒')
```

运行优化后的脚本：

```bash
python amp_optimized.py
```

### 分析结果

脚本完成后，将总训练时间与`fp32_baseline.py`基准运行进行比较。你应该会观察到训练时间有明显减少。如果你使用`nvidia-smi`监控了GPU内存使用情况，你也将看到明显下降。

你的结果会根据具体的GPU有所不同，但它们可能看起来像这样：

| 指标 | FP32 (基准) | AMP (FP16/FP32) | 改进 |
| --- | --- | --- | --- |
| 训练时间 (5个epoch) | 约75秒 | 约48秒 | 快约36% |
| 峰值GPU内存 | 约1.8 GB | 约1.1 GB | 少约39% |

性能提升主要来自两个方面。首先，$FP16$操作在支持Tensor Core的GPU上快得多。其次，使用半精度数据减少了模型、激活和梯度的内存占用，这减少了内存传输所花费的时间。



![训练时间比较：FP32 对比 AMP](plots/7034-0.json)



> 性能比较显示，当使用自动混合精度时，总训练时间有明显减少。

这个实验表明AMP是一种强大且易于实现的技巧。仅需三行代码，你通常可以获得大幅加速和内存节省，使其成为训练过程遇到瓶颈时首先考虑的优化之一。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR) 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  这篇基础论文介绍了混合精度训练的概念和技术，展示了如何结合FP16和FP32以提高训练速度和内存效率。
- [Automatic Mixed Precision Package - torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) — PyTorch Developers (2025)
  PyTorch官方文档，详细说明了如何使用`torch.cuda.amp.autocast`和`torch.cuda.amp.GradScaler`进行混合精度训练。
- [Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html) — NVIDIA Developer Documentation (2023)
  Publisher: NVIDIA Corporation
  提供了在NVIDIA GPU上优化深度学习性能的全面指导，包括混合精度训练和Tensor Core使用的解释。
