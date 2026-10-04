---
course: "deep-learning-regularization-optimization"
chapter: "combining-techniques-practical"
lesson: "practice-building-tuned-model"
sourceId: 5017
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-8-combining-techniques-practical/practice-building-tuned-model"
title: "动手实践：构建与调整正则化/优化模型"
description: "全面实践环节，使用组合方法构建模型并对其进行调整。"
order: 9
plots: ["plots/5017-0.json", "plots/5017-1.json"]
sourceHash: "509bbfdce21f609e99b6a264e7c268d109b47c1728133dbb9b370462b97b7fae"
sourceCorrections: []
---

将各种正则化 (regularization)技术（L1/L2、Dropout、提前停止、数据增强）和优化方法（SGD变体、Adam、RMSprop、学习率调度）整合到典型的深度学习 (deep learning)流程中是主要内容。这种方法旨在构建和调整模型，观察这些方法如何共同作用以改善泛化能力。

我们的目标是在Fashion-MNIST数据集上训练一个用于图像分类的卷积神经网络 (neural network)（CNN）。我们将从一个基础模型开始，逐步添加组件，同时监控其对训练过程和验证表现的影响。

### 环境与数据集设置

首先，确保您已安装PyTorch和`torchvision`。我们将使用Fashion-MNIST，这是一个包含28x28灰度服装图像的数据集，分为10个类别。它是比MNIST数字稍复杂一些的标准基准数据集。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 设备配置
device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')

# 超参数（初始）
num_epochs = 15
batch_size = 128
learning_rate = 0.001

# 数据加载和转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # 对灰度图像进行归一化
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                  download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                 download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

### 基线模型

让我们定义一个没有显式正则化 (regularization)的简单CNN架构，并使用标准优化方法。

```python
# 简单的CNN架构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 将输出展平以用于全连接层
        # 输入图像 28x28 -> pool1 -> 14x14 -> pool2 -> 7x7
        # 输出特征: 32 通道 * 7 * 7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 个类别

    def forward(self, x):
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1) # 展平
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        return out

# 实例化基线模型、损失函数和优化器
model_base = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_base = optim.Adam(model_base.parameters(), lr=learning_rate)

# --- 基线模型训练循环占位符 ---
# 通常您会在此处训练此模型，记录每个 epoch 的训练/验证损失和准确率。
# 为简洁起见，我们将模拟结果。
print("基线模型已定义。（训练模拟如下）")
```

训练基线模型后，我们可能会观察到如下所示的模拟学习曲线。通常，训练损失稳定下降，而验证损失在几个周期后开始增加，这表明模型过拟合 (overfitting)。



![基线模型表现（模拟）](plots/5017-0.json)



> 基线模型的模拟学习曲线。注意验证损失增加和验证准确率停滞不前，而训练损失/准确率持续改善，这是过拟合的典型表现。

### 整合正则化 (regularization)与优化技术

现在，让我们通过添加批量归一化 (normalization)、Dropout和L2正则化（权重 (weight)衰减）来改进我们的模型。我们仍将继续使用Adam优化器。

#### 修改架构

我们需要在卷积层之后（通常在激活函数 (activation function)之前）添加`nn.BatchNorm2d`，并在全连接层中通常在激活函数之后添加`nn.Dropout`。

```python
# 改进型CNN架构
class EnhancedCNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # 添加了BN
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # 添加了BN
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # 添加了Dropout
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 卷积块 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        # 卷积块 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # 展平与全连接层
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.dropout(out) # 在最后一层之前应用dropout
        out = self.fc2(out)
        return out

# 实例化改进模型和损失函数
model_enhanced = EnhancedCNN(dropout_rate=0.5).to(device)
criterion = nn.CrossEntropyLoss() # 相同的损失函数

# --- 优化器设置说明 ---
# L2正则化（权重衰减）直接在优化器中添加
l2_lambda = 0.0001 # L2强度示例
optimizer_enhanced = optim.Adam(model_enhanced.parameters(),
                                lr=learning_rate,
                                weight_decay=l2_lambda)

# --- 改进模型训练循环占位符 ---
# 与之前类似的训练循环，但使用model_enhanced和optimizer_enhanced。
# 请记住，由于BN和Dropout的存在，需要适当设置model.train()和model.eval()。
print(f"改进模型已定义，包含Dropout、BatchNorm和L2权重衰减 (lambda={l2_lambda})。")
```

#### 变化：

1. **批量归一化 (`nn.BatchNorm2d`)**：在每个卷积层之后、ReLU激活之前添加。这有助于训练稳定，可能允许使用更高的学习率，并提供轻微的正则化效果。
2. **Dropout (`nn.Dropout`)**：在第一个全连接层激活之后添加。这会在训练期间随机将一部分输入设置为零，防止模型过度依赖特定神经元，并促进特征冗余。
3. **L2正则化（权重衰减）**：通过`weight_decay`参数 (parameter)直接集成到`optim.Adam`优化器中。这会对大权重进行惩罚，促使模型更简单。
4. **优化器**：我们继续使用Adam，它通常开箱即用表现良好，特别是与批量归一化结合时。

#### 训练考量

在训练带有Dropout和批量归一化的模型时，管理模型的状态很重要：

- 在每个周期的训练循环之前使用`model.train()`。这会启用Dropout并确保BN使用批次统计数据。
- 在验证/测试循环之前使用`model.eval()`。这会禁用Dropout并确保BN使用训练期间累积的均值和方差的运行估计值。

### 比较表现

训练改进模型后，我们将其学习曲线与基线模型进行比较。



![基线模型与改进模型表现对比（模拟）](plots/5017-1.json)



> 学习曲线的模拟对比。改进模型显示出较慢的初始训练收敛速度（由于正则化 (regularization)），但取得了更低的验证损失和更高的验证准确率，训练与验证指标之间的差距也更小，这表明泛化能力更好。

观察：

- **正则化效果**：改进模型的训练损失可能比基线模型下降得慢，其最终训练损失/准确率可能略差。这是预料之中的；正则化限制了模型完美拟合训练数据的能力。
- **泛化能力提升**：与基线模型相比，改进模型的验证损失应更低，验证准确率更高。训练和验证曲线之间的差距也应更小，这表明过拟合 (overfitting)现象减少。
- **稳定性**：批量归一化 (normalization)通常能使训练曲线更平滑，对初始化也不那么敏感。

### 进一步调整与实验

本次实践环节展示了常用技术的整合。然而，找到最佳组合通常需要反复实验：

- **超参数 (parameter) (hyperparameter)调整**：调整`dropout_rate`、`weight_decay`（L2 lambda）和`learning_rate`。使用随机搜索或更高级的贝叶斯优化等方法。
- **学习率调度**：实现学习率调度（例如，`torch.optim.lr_scheduler.StepLR`或`CosineAnnealingLR`），以可能进一步改善收敛。
- **提前停止**：监控验证损失，当其在一定数量的周期（耐心期）内不再改善时停止训练，以防止过拟合 (overfitting)并节省计算资源。
- **数据增强**：为训练集的`transforms.Compose`管道添加数据增强（例如，随机水平翻转、小幅度旋转）。这作为另一种强大的正则化 (regularization)形式。

结合学习率调度器并提供提前停止逻辑的示例：

```python
# ... (改进模型和数据集设置同上) ...

optimizer_enhanced = optim.Adam(model_enhanced.parameters(), lr=learning_rate, weight_decay=l2_lambda)

# 添加学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer_enhanced, step_size=5, gamma=0.1) # 每5个周期降低学习率

# --- 包含调度器和提前停止逻辑的训练循环占位符 ---
# 在您的周期循环中：
#   model_enhanced.train()
#   ... (前向传播、损失计算、反向传播) ...
#   optimizer_enhanced.step()
#   scheduler.step() # 更新学习率
#
#   model_enhanced.eval()
#   ... (验证循环) ...
#   检查验证损失以判断提前停止条件
# ---

print("训练设置包括Adam、L2、Dropout、BN、学习率调度器。")
```

### 结论

这次动手练习展示了如何将Dropout、批量归一化 (normalization)和权重 (weight)衰减等正则化 (regularization)技术与Adam等适当的优化策略结合起来，从而获得比简单基线模型泛化能力更好的模型。通过系统地添加这些组件并使用验证指标和学习曲线监控其效果，您可以有效对抗过拟合 (overfitting)，并构建更可靠的深度学习 (deep learning)系统。请记住，这些技术的特定组合和调整在很大程度上取决于数据集、模型架构和具体的任务。实验是该过程的常规部分。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教材，涵盖深度学习基础知识，包括各种正则化技术（L1/L2、dropout、提前停止）、优化算法（SGD变体、Adam）和卷积神经网络。
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Sergey Ioffe and Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Volume: 37; Pages: 448-456; DOI: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167)
  介绍批标准化的原始论文，该技术用于提高深度神经网络训练的稳定性和性能。
- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  介绍了Dropout，一种广泛用于深度学习模型的正则化技术，解释了其减少过拟合的机制。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma and Jimmy Ba (2015)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  描述Adam优化器的开创性论文，这是一种在深度学习中广泛采用的自适应学习率优化算法。
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch深度学习框架的官方文档，提供了模块（例如nn.Module、nn.Conv2d、nn.BatchNorm2d、nn.Dropout）、优化器（torch.optim）和数据工具（torchvision.datasets、DataLoader）的全面API参考。
