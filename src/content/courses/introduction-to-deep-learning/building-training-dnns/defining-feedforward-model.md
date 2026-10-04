---
course: "introduction-to-deep-learning"
chapter: "building-training-dnns"
lesson: "defining-feedforward-model"
sourceId: 5055
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/defining-feedforward-model"
title: "定义一个前馈网络模型"
description: "展示如何使用所选框架定义顺序模型的架构。"
order: 4
plots: []
sourceHash: "a5c6bfbb0c6efcc3ae9c53720f822bb98b100fdfdc08e0c1a42d5a2f39823888"
sourceCorrections: []
---

你了解神经网络 (neural network)的组成部分：由神经元构成的层，通过权重 (weight)连接，并使用激活函数 (activation function)引入非线性。那么，如何将这种认识转化为可以训练的实际代码呢？深度学习 (deep learning)框架在此发挥作用。

PyTorch 和 TensorFlow（通常通过其更高级别的 API Keras 使用）等框架提供了高效构建、训练和评估神经网络所需的模块和自动化功能。它们在后台处理复杂的运算，如梯度计算（自动微分），并利用硬件加速（如 GPU），让你能够专注于网络的设计和训练流程。尽管这些思想是可通用的，但在此我们将使用 PyTorch 进行示例。

### 使用 `torch.nn.Sequential` 定义模型

对于许多常见的前馈网络，其架构是简单的线性层堆叠：一个层的输出直接作为下一层的输入。PyTorch 正好为此提供了一个便捷的容器，名为 `torch.nn.Sequential`。你可以通过定义一个有序的层模块序列并将其传递给其构造函数来定义模型。

让我们看看你将要使用的基本构建块：

1. **线性层 (`torch.nn.Linear`)**：这些是密集连接层，在其他场合也称为全连接层。它们对输入数据应用线性变换：$y = xW^T + b$。定义 `Linear` 层时，你必须指定输入特征的数量 (`in_features`) 和输出特征的数量 (`out_features`)。

   ```python
   import torch
   import torch.nn as nn

   # 示例：一个将 784 个输入特征映射到 128 个输出特征的线性层
   layer1 = nn.Linear(in_features=784, out_features=128)
   ```
2. **激活函数 (activation function) (`torch.nn.ReLU`、`torch.nn.Sigmoid` 等)**：这些模块将非线性激活函数逐元素应用于前一层的输出。它们通常不改变数据的形状，因此在将它们定义为单独的层时，无需指定输入/输出大小。

   ```python
   # 示例：应用 ReLU 激活函数
   activation1 = nn.ReLU()
   ```

现在，让我们使用 `nn.Sequential` 将这些组合成一个简单的两层网络。假设我们有 784 个特征的输入数据（例如来自 MNIST 数据集的展平 28x28 图像），并且我们希望将其分类为 10 个类别。我们的网络可以有一个包含 128 个神经元的隐藏层和 ReLU 激活，然后是一个包含 10 个神经元（每个类别一个）的输出层。

```python
import torch
import torch.nn as nn

# 定义模型架构为层序列
model = nn.Sequential(
    nn.Linear(in_features=784, out_features=128), # 输入层 (784) 到隐藏层 (128)
    nn.ReLU(),                                    # 隐藏层激活函数
    nn.Linear(in_features=128, out_features=10)   # 隐藏层 (128) 到输出层 (10)
    # 注意：输出层的激活函数（例如 Softmax）通常是单独应用的，
    # 或者为了数值稳定性集成到损失函数中，
    # 特别是对于分类任务。
)

# 打印模型结构以验证
print(model)
```

执行此代码将打印模型的摘要：

```
Sequential(
  (0): Linear(in_features=784, out_features=128, bias=True)
  (1): ReLU()
  (2): Linear(in_features=128, out_features=10, bias=True)
)
```

注意一个重要细节：一个 `Linear` 层的 `out_features` 必须与下一个 `Linear` 层的 `in_features` 匹配（在考虑任何不改变特征维度的中间层，如激活函数之后）。当你这样定义它们时，PyTorch 会自动为每个 `Linear` 层创建并初始化权重 (weight)矩阵 ($W$) 和偏置 (bias)向量 (vector) ($b$)。

以下是此简单网络架构的可视化表示：

> 一个简单的单隐藏层前馈网络，使用顺序构建块定义。数据从输入特征流过线性变换、ReLU 激活、另一次线性变换，到达最终的输出得分。

### 通过继承 `torch.nn.Module` 定义模型

`nn.Sequential` 方法对于线性堆叠非常适合，但深度学习 (deep learning)架构可能更加复杂。你可能需要跳跃连接（其中较早层的输出被添加到较后层的输出，这在 ResNets 中很常见）、多个输入或输出，或者在网络不同部分共享权重 (weight)的层。对于这些情况，PyTorch 提供了一种更灵活、更强大的方法：通过继承 `torch.nn.Module` 基类来定义你的模型。

这主要包含两个步骤：

1. **在 `__init__` 中定义层**：在你自定义类的构造函数 (`__init__`) 中，你将网络将使用的所有层定义为类属性。你在这里实例化 `nn.Linear`、`nn.ReLU`、`nn.Conv2d`（用于 CNN）等模块。这里是创建层（及其参数 (parameter)）的地方。在你的构造函数中，首先调用 `super().__init__()` 很重要，以确保基类得到正确初始化。
2. **在 `forward` 中定义前向传播**：你实现 `forward` 方法，该方法接收输入张量作为参数，并明确定义数据如何流经你在 `__init__` 中定义的层。你按照期望的序列在输入数据上调用层，这允许复杂的数据流路由、分支或合并。

让我们使用这种方法重新定义我们之前的简单网络进行比较：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F # 通常用于无状态函数，如激活函数

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 调用父类 (nn.Module) 构造函数
        super(SimpleNet, self).__init__()
        # 将层定义为该类的属性
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        # 如果愿意，我们也可以在这里定义激活层：
        # self.relu = nn.ReLU()

    def forward(self, x):
        # 定义数据流经网络的方式
        # 将输入通过第一个线性层
        x = self.layer1(x)
        # 应用 ReLU 激活。这里我们使用函数版本。
        x = F.relu(x)
        # 将结果通过第二个线性层
        x = self.layer2(x)
        # 输出激活（例如 softmax）可能会在这里或稍后在损失函数中应用
        return x

# 使用特定大小实例化模型
input_features = 784
hidden_neurons = 128
output_classes = 10
model_custom = SimpleNet(input_features, hidden_neurons, output_classes)

# 打印模型结构。它显示了在 __init__ 中定义的层
print(model_custom)
```

运行此代码会产生一个类似的摘要，列出定义的层：

```
SimpleNet(
  (layer1): Linear(in_features=784, out_features=128, bias=True)
  (layer2): Linear(in_features=128, out_features=10, bias=True)
)
```

在这个 `forward` 方法中，我们使用了 `torch.nn.functional` 模块中的 `F.relu`。该模块提供了许多常见操作的函数版本，包括激活函数 (activation function)。这些函数是“无状态”的，意味着它们没有关联的参数（权重或偏置 (bias)），无需在训练期间进行追踪或更新。使用 `F.relu` 通常比在 `__init__` 中定义 `self.relu = nn.ReLU()` 然后在 `forward` 中调用 `x = self.relu(x)` 稍微更简洁。然而，这两种方法都是有效的。

尽管对于简单的顺序模型来说稍微更冗长，但继承 `nn.Module` 赋予你对前向传播逻辑的完全控制。这使得它对于实现你之后会遇到的更高级和自定义的网络架构不可或缺，例如具有特定层排列的卷积神经网络 (neural network) (CNN) 和涉及循环结构的循环神经网络 (RNN)。

在 `nn.Sequential` 和继承 `nn.Module` 之间做出选择取决于你所需架构的复杂程度。对于直接的前馈设计，从 `nn.Sequential` 开始，因为它通常更快、更容易阅读。当你需要更大的灵活性来定义非线性数据流、自定义操作或网络中的共享组件时，转向继承 `nn.Module`。无论采用哪种方法，准确定义层并确保输入/输出维度正确对齐 (alignment)是构建功能性神经网络模型的根本。

## 参考资料

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Foundation (2024)
  介绍PyTorch中所有神经网络模块的基类，阐述模型定义方法和结构。
- [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) — PyTorch Foundation (2024)
  说明PyTorch中用于序列模块的容器，展示其在线性堆叠网络层中的应用。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供前馈神经网络、激活函数和通用深度学习架构的理论基础。
- [Neural Networks with PyTorch](https://pytorch.org/tutorials/beginner/buildmodel_tutorial.html) — PyTorch Foundation (2021)
  Publisher: PyTorch Foundation
  提供在PyTorch中构建神经网络的实践指南，展示`nn.Sequential`和`nn.Module`两种方法。
