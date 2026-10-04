---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-training-loops-for-keras-devs"
lesson: "practical-pytorch-training-evaluation"
sourceId: 5765
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/practical-pytorch-training-evaluation"
title: "动手实践：实现一个完整的训练和评估循环"
description: "从头开始编写一个完整的 PyTorch 神经网络训练和评估脚本。"
order: 8
plots: ["plots/5765-0.json"]
sourceHash: "65978f613d6a38e00300e5d100c0be3f12f8cdd0499a0780c214b22d28451a73"
sourceCorrections: []
---

PyTorch 训练流程包含多个组成部分，例如损失函数 (loss function)、优化器和梯度计算机制。这些组件将组合起来，从头构建一个完整的训练和评估循环。

本练习旨在加深你对 PyTorch 如何处理模型训练的理解，并与你熟悉的 TensorFlow Keras 中更抽象的 `model.fit()` 和 `model.evaluate()` 方法形成清晰的对比。

我们将进行一项常见任务：图像分类。为了将重心放在循环机制上，我们将使用 FashionMNIST 数据集（一个常用于基准测试的 MNIST 替代品）以及一个相对简单的卷积神经网络 (neural network) (CNN)。

### 实践先决条件

在开始编写代码之前，请确保你已安装 PyTorch 和 TorchVision。如果你在一个新环境中工作，通常可以通过以下方式安装它们：

```bash
pip install torch torchvision
```

我们还需要 `matplotlib` 用于末尾的一个简单可视化，所以如果你没有安装，请 `pip install matplotlib`。

### 1. 准备工作：导入模块和设备配置

首先，让我们导入必要的 PyTorch 模块并配置我们的设备（如果可用则使用 GPU，否则使用 CPU）。这是大多数 PyTorch 脚本的标准起始点。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

使用 `torch.device` 可使你的代码具有可移植性，并在不同硬件配置上高效运行。

### 2. 数据准备

我们将使用 `torchvision` 加载和预处理 FashionMNIST 数据集。`torchvision.transforms` 提供了方便的数据增强和归一化 (normalization)工具。

```python
# 转换
transform = transforms.Compose([
    transforms.ToTensor(), # 将 PIL 图像或 NumPy ndarray 转换为张量，并缩放到 [0,1] 范围
    transforms.Normalize((0.5,), (0.5,)) # 使用均值 0.5 和标准差 0.5 进行归一化
])

# 加载 FashionMNIST 数据集
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建 DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 供参考，类别名称
classes = ('T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
           '凉鞋', '衬衫', '运动鞋', '包', '踝靴')
```

这里，`transforms.ToTensor()` 将图像转换为 PyTorch 张量，而 `transforms.Normalize()` 将像素值调整到特定范围，这通常有助于训练稳定性。`DataLoader` 处理批处理、混洗和并行数据加载。

### 3. 定义模型

让我们定义一个简单的 CNN。如果你已经学习了第2章，这个结构看起来会很熟悉。我们继承 `nn.Module`，并在 `__init__` 中定义层，在 `forward` 方法中定义前向传播。

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1), # 28x28x1 -> 28x28x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28x16 -> 14x14x16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1), # 14x14x16 -> 14x14x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14x32 -> 7x7x32
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), # 展平 7*7*32 特征图
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1) # 为全连接层展平输出
        x = self.fc_layers(x)
        return x

# 实例化模型并将其移动到设备
model = SimpleCNN(num_classes=len(classes)).to(device)
print(model)
```

该模型包含两个卷积层，接着是最大池化，然后是两个全连接层。这是一种标准但有效的图像分类模型结构，适用于 FashionMNIST 等任务。

### 4. 损失函数 (loss function)和优化器

接下来，我们选择一个损失函数和一个优化器。对于多类别分类，`CrossEntropyLoss` 是一个常见选择。对于优化器，`Adam` 是一种流行且有效的算法。

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

注意我们如何将 `model.parameters()` 传递给优化器。这会告诉优化器在训练期间需要更新哪些张量（我们模型的权重 (weight)和偏置 (bias)）。学习率 `lr` 是一个你可以调整的超参数 (parameter) (hyperparameter)。

### 5. 训练循环

这是 PyTorch 显式操作的优势所在。我们手动遍历 epoch 和批次。

```python
def train_one_epoch(epoch_index, tb_writer=None):
    running_loss = 0.
    last_loss = 0.
    correct_predictions = 0
    total_samples = 0

    # 将模型设置为训练模式
    model.train(True) # or model.train()

    for i, data in enumerate(train_loader):
        # 每个数据实例都是一个输入 + 标签对
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # 将数据移动到配置的设备

        # 为每个批次清除梯度！
        optimizer.zero_grad()

        # 为当前批次进行预测
        outputs = model(inputs)

        # 计算损失及其梯度
        loss = criterion(outputs, labels)
        loss.backward() # 为每个 requires_grad=True 的参数 x 计算 dloss/dx

        # 调整学习权重
        optimizer.step() # 使用梯度 x.grad 更新 x 的值

        # 收集数据并报告
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        if i % 100 == 99: # 每 100 个批次记录一次
            last_loss = running_loss / 100 # 每个批次的损失
            current_accuracy = 100 * correct_predictions / (batch_size * (i + 1)) # 近似准确率
            print(f'  Epoch {epoch_index + 1}, Batch {i + 1:5d} loss: {last_loss:.3f}, Accuracy: {current_accuracy:.2f}%')
            if tb_writer:
                tb_writer.add_scalar('Training Loss', last_loss, epoch_index * len(train_loader) + i)
                tb_writer.add_scalar('Training Accuracy', current_accuracy, epoch_index * len(train_loader) + i)
            running_loss = 0.

    epoch_accuracy = 100 * correct_predictions / total_samples
    return last_loss, epoch_accuracy # 返回最后一个批次的损失和当前 epoch 的准确率
```

让我们分解批次循环中的关键步骤：

1. **`inputs, labels = inputs.to(device), labels.to(device)`**: 当前批次的数据被移动到 GPU（如果可用）。
2. **`optimizer.zero_grad()`**: 这非常重要。PyTorch 默认会累积梯度。如果你不在每个批次开始时清除它们，你将累积来自前一个批次的梯度，导致更新不正确。
3. **`outputs = model(inputs)`**: 前向传播，输入数据流经网络以产生预测。
4. **`loss = criterion(outputs, labels)`**: 损失函数 (loss function)将模型的预测 (`outputs`) 与真实标签 (`labels`) 进行比较。
5. **`loss.backward()`**: 这是 `autograd` 自动微分发生作用的地方。它计算损失对所有 `requires_grad=True` 的模型参数 (parameter)的梯度。
6. **`optimizer.step()`**: 优化器使用 `backward()` 调用中计算的梯度更新模型的参数。

以下图表说明了典型 PyTorch 训练循环中一个批次的操作流程：

> 训练循环中处理一个批次的操作流程。

### 6. 评估循环

在每个 epoch 之后，或在训练结束时，你都会想在单独的数据集（例如，验证集或测试集）上评估你的模型，以了解其泛化能力。评估循环与训练循环相似，但有主要区别：

```python
def evaluate_model(loader):
    # 将模型设置为评估模式
    model.eval() # or model.train(False)

    running_vloss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 禁用梯度计算以进行评估
    with torch.no_grad(): # 重要！
        for i, vdata in enumerate(loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)

            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()

            _, predicted = torch.max(voutputs.data, 1)
            total_samples += vlabels.size(0)
            correct_predictions += (predicted == vlabels).sum().item()

    avg_vloss = running_vloss / (i + 1) # 所有验证批次的平均损失
    accuracy = 100 * correct_predictions / total_samples
    print(f'Validation Loss: {avg_vloss:.3f}, Validation Accuracy: {accuracy:.2f}%')
    return avg_vloss, accuracy
```

评估循环的主要区别：

1. **`model.eval()`**: 这会将模型设置为评估模式。这很重要，因为某些层（如 Dropout 和 BatchNorm）在训练和评估期间表现不同。`model.eval()` 确保它们使用评估时的行为。
2. **`with torch.no_grad():`**: 这个上下文 (context)管理器会禁用梯度计算。在评估期间，我们不需要计算梯度，这可以节省内存和计算。它确保评估过程的任何部分都不会意外修改模型的梯度。

### 7. 组合所有部分：主训练脚本

现在，让我们将这些函数组合成一个主脚本，用于训练模型几个 epoch 并进行评估。

```python
num_epochs = 5 # 仅作演示；通常需要更多 epoch
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

print("Starting Training...")
for epoch in range(num_epochs):
    print(f'EPOCH {epoch + 1}:')

    # 训练
    model.train(True)
    avg_loss, train_acc = train_one_epoch(epoch) # 为简单起见，此处不使用 TensorBoard 写入器
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)

    # 验证
    model.eval()
    avg_vloss, val_acc = evaluate_model(test_loader)
    val_losses.append(avg_vloss)
    val_accuracies.append(val_acc)

    print(f'EPOCH {epoch + 1} Summary: Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_vloss:.3f}, Val Acc: {val_acc:.2f}%')
    print("-" * 30)

print('Finished Training')

# 可选：保存模型
# torch.save(model.state_dict(), 'fashion_mnist_cnn.pth')
# print('模型已保存到 fashion_mnist_cnn.pth')
```

该脚本迭代 `num_epochs` 次。在每个 epoch 中，它会调用 `train_one_epoch` 在训练数据上训练模型，然后调用 `evaluate_model` 评估其在测试数据上的性能。我们还会收集损失和准确率值，这对于后续绘图很有用。

### 8. 训练进度可视化（可选）

一种常见做法是绘制训练和验证损失/准确率随 epoch 变化的图表，以监测过拟合 (overfitting)情况并了解训练动态。

```python
import matplotlib.pyplot as plt

epochs_range = range(1, num_epochs + 1)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss', color='#1c7ed6', marker='o')
plt.plot(epochs_range, val_losses, label='Validation Loss', color='#fd7e14', marker='x')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy', color='#1c7ed6', marker='o')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', color='#fd7e14', marker='x')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
```

如果你运行完整脚本，应该会看到类似的图表（实际值会因运行而异）：



![训练和验证损失](plots/5765-0.json)



> 显示训练和验证损失随 epoch 减少的典型趋势的示例图。

### 给 TensorFlow 开发者的建议

对于 TensorFlow Keras 的开发者而言，`model.compile()` 和 `model.fit()` 抽象了许多细节，PyTorch 的做法可能最初看起来更冗长。然而，这种显式操作是 PyTorch 的优势之一。

- **完全控制**: 你可以直接控制训练过程的每一步。这使得实现自定义训练逻辑、非标准反向传播 (backpropagation)方案或复杂指标计算变得更容易。
- **透明性**: 你可以清楚地看到发生了什么：梯度何时清零，损失如何计算，权重 (weight)何时更新。这对于调试和理解模型行为非常有帮助。
- **灵活性**: 需要在优化器步骤之前修改梯度吗？想对模型的不同部分应用不同的优化器吗？PyTorch 的手动循环使这些任务变得简单。

虽然 Keras 为标准工作流程提供了便利，但掌握 PyTorch 训练循环有助于你处理更广泛的研究问题并实现更复杂的训练程序。这个动手练习已经打下了基础。随着你的进步，你会发现自己会为这个基本循环添加更多功能，例如学习率调度器、早期停止和自定义日志记录，所有这些都自然地融入到这种显式结构中。

## 参考资料

- [Learning PyTorch with Examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html) — Justin Johnson (2024)
  Publisher: PyTorch Foundation
  PyTorch官方教程，演示了PyTorch核心概念，包括张量、自动求导、`nn.Module`、优化器和基本训练循环，为自定义训练过程提供了基础。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，提供了深度学习的理论基础，涵盖了与训练循环相关的卷积神经网络、优化算法和损失函数等主题。
- [Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747) — Han Xiao, Kashif Rasul, Roland Vollgraf (2017)
  Journal: arXiv preprint arXiv:1708.07747; DOI: [10.48550/arXiv.1708.07747](https://doi.org/10.48550/arXiv.1708.07747)
  介绍Fashion-MNIST数据集的原始论文，该数据集在图像分类的实践练习中被使用。
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  详细说明PyTorch自动微分引擎（Autograd）的官方文档，这对于理解训练循环中的梯度计算和管理至关重要。
