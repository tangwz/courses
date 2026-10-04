---
course: "introduction-to-deep-learning"
chapter: "building-training-dnns"
lesson: "practice-training-mnist-classifier"
sourceId: 5061
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/practice-training-mnist-classifier"
title: "动手实践：在MNIST上训练分类器"
description: "构建、训练并评估一个简单的前馈神经网络，用于分类MNIST手写数字。"
order: 10
plots: ["plots/5061-0.json"]
sourceHash: "f76d1435435f462a64d66da4f54c844ff0e44972f959b05fba4ab66c9b0baa9b"
sourceCorrections: []
---

我们将构建、训练并评估一个简单的前馈神经网络 (neural network)，用于对手写数字进行分类，这些数字来自著名的MNIST数据集。本次实践将涵盖数据准备、使用框架定义模型、选择损失函数 (loss function)和优化器、训练循环以及模型性能评估等方面。

我们将使用PyTorch（一个常用的深度学习 (deep learning)库）来完成此任务。

### 环境配置

首先，请确保已安装PyTorch和Torchvision。如果尚未安装，通常可以使用pip或conda进行安装。然后，导入所需的模块：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 检查GPU是否可用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用设备: {device}")
```

### 加载和准备MNIST数据集

MNIST数据集包含70,000张手写数字（0-9）的灰度图像，每张大小为28x28像素。其中60,000张用于训练，10,000张用于测试。`torchvision`提供了方便的数据集访问方式。我们将应用转换，将图像转换为PyTorch张量并对其像素值进行归一化 (normalization)。归一化有助于训练稳定；我们使用MNIST的标准均值(0.1307)和标准差(0.3081)。

```python
# 应用于数据的转换
transform = transforms.Compose([
    transforms.ToTensor(), # 将图像转换为PyTorch张量
    transforms.Normalize((0.1307,), (0.3081,)) # 归一化像素值
])

# 下载并加载训练数据
trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

# 下载并加载测试数据
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
testloader = DataLoader(testset, batch_size=1000,
                                         shuffle=False, num_workers=2)
```

`DataLoader`封装了数据集，提供了一个迭代器，以便于批处理、数据打乱和并行数据加载。

### 定义神经网络 (neural network)模型

我们将定义一个简单的前馈神经网络（多层感知机），它带有两个隐藏层并使用ReLU激活函数 (activation function)。输入层接收扁平化的28x28图像（784个特征），输出层有10个神经元（每个数字类别对应一个），此处未应用激活函数，因为我们将使用`CrossEntropyLoss`，它会在内部应用LogSoftmax。

```python
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128) # 输入层 -> 隐藏层1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)   # 隐藏层1 -> 隐藏层2
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)    # 隐藏层2 -> 输出层

    def forward(self, x):
        x = self.flatten(x)    # 扁平化图像
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)        # 原始分数（logits）
        return x

# 实例化模型并将其移至相应设备（GPU或CPU）
model = SimpleMLP().to(device)
print(model)
```

### 编译模型：损失函数 (loss function)和优化器

对于多类别分类，`CrossEntropyLoss`是一个标准选择。它在一个类中结合了`LogSoftmax`和`NLLLoss`（负对数似然损失）。我们将使用Adam优化器，这是许多问题的一种常用且有效的选择。

```python
# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001) # 学习率 = 0.001
```

### 训练模型

训练过程包括对数据集进行多次迭代（训练轮次）。在每个训练轮次中，我们以批次形式遍历数据。对于每个批次：

1. 将数据和标签移至指定设备（GPU/CPU）。
2. 清零上一批次累积的梯度（`optimizer.zero_grad()`）。
3. 执行前向传播：将输入批次送入模型以获得预测结果（`outputs = model(inputs)`）。
4. 计算预测结果与真实标签之间的损失（`loss = criterion(outputs, labels)`）。
5. 执行反向传播 (backpropagation)：计算损失相对于模型参数 (parameter)的梯度（`loss.backward()`）。
6. 使用计算出的梯度更新模型参数（`optimizer.step()`）。

我们将训练几个训练轮次，并定期打印损失值。

```python
num_epochs = 5 # 对训练数据集迭代的次数
training_losses = [] # 用于存储损失值以便绘图

print("开始训练...")
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 获取输入；数据是 [inputs, labels] 的列表
        inputs, labels = data[0].to(device), data[1].to(device)

        # 清零参数梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播并优化
        loss.backward()
        optimizer.step()

        # 打印统计信息
        running_loss += loss.item()
        if (i + 1) % 200 == 0: # 每200个小批次打印一次
            avg_loss = running_loss / 200
            print(f'训练轮次 [{epoch + 1}/{num_epochs}], 批次 [{i + 1}/{len(trainloader)}], 损失: {avg_loss:.4f}')
            training_losses.append({"epoch": epoch + (i+1)/len(trainloader), "loss": avg_loss})
            running_loss = 0.0

print('训练完成')
```

### 监控训练进度

在训练过程中可视化损失值有助于了解模型是否正在有效学习。损失值下降通常表明模型正在学习。



![每200个批次的训练损失](plots/5061-0.json)



> 示例训练损失曲线，显示随训练轮次下降。实际值取决于具体的运行情况。

### 评估模型性能

训练完成后，我们将在未见过的测试数据集上评估模型的性能。我们遍历测试数据，进行预测，并将其与真实标签进行比较以计算准确率。在评估期间禁用梯度计算（`torch.no_grad()`）有必要，以节省内存和计算资源，因为我们不更新权重 (weight)。

```python
correct = 0
total = 0
# 由于我们不进行训练，因此无需计算梯度
with torch.no_grad():
    model.eval() # 将模型设置为评估模式
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # 通过网络运行图像来计算输出
        outputs = model(images)
        # 能量最高的类别被我们选为预测结果
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    model.train() # 将模型设置回训练模式（如果使用dropout/batchnorm则很重要）

accuracy = 100 * correct / total
print(f'网络在10000张测试图像上的准确率: {accuracy:.2f} %')
```

经过5个训练轮次后，这个简单网络在MNIST上的典型准确率可能在96-97%左右。这表明我们的网络已经能够相当好地识别手写数字。

这个动手实践示例讲解了使用PyTorch构建和训练一个基本神经网络 (neural network)的核心步骤。你加载了数据，定义了模型架构，选择了损失和优化策略，执行了训练循环，并评估了结果。这为处理更复杂的深度学习 (deep learning)问题提供了起点。在后续章节中，我们将学习提升性能和处理更复杂数据类型的方法。

## 参考资料

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Foundation (2024)
  Publisher: PyTorch Foundation
  提供 PyTorch 的全面指南和 API 参考，是神经网络实际实现所需的基础资源。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涉及深度学习的理论和实践方面，包括前馈网络、优化和训练。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference for Learning Representations; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了 Adam 优化算法，该算法广泛用于训练深度神经网络，并且是本实践中的核心组成部分。
- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  这篇开创性论文介绍了 MNIST 数据集，并开创了用于手写数字识别的卷积神经网络。
