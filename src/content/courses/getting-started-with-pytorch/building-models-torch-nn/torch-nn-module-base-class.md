---
course: "getting-started-with-pytorch"
chapter: "building-models-torch-nn"
lesson: "torch-nn-module-base-class"
sourceId: 2149
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/torch-nn-module-base-class"
title: "`torch.nn.Module` 基类"
description: "理解PyTorch中所有神经网络模型的基本构建块。"
order: 1
plots: []
sourceHash: "b539ef99df314289400259e9da42ff0caa6ad40fe263edd055714b1b41ba180d"
sourceCorrections: []
---

构建神经网络 (neural network)在PyTorch中围绕着一个主要的理念：`torch.nn.Module`。可以将`nn.Module`视为一个基础蓝图或基类，所有的神经网络模型、层，甚至复杂的复合结构都是以此为基础构建的。它提供了一种标准化的方式来封装模型参数 (parameter)、管理这些参数的辅助函数（例如在CPU和GPU之间移动它们），以及定义输入数据在网络中流动的逻辑。

无论何时，当你在PyTorch中定义自定义神经网络时，通常会通过创建一个继承自`nn.Module`的Python类来完成。这种继承为你的自定义类提供了大量内置功能，这些功能对于深度学习 (deep learning)工作流程来说非常必要。

### `nn.Module`的核心结构

本质上，使用`nn.Module`需要在你的自定义类中实现两个主要方法：

1. **`__init__(self)`:** 构造函数。你在这里定义和初始化网络的组件，例如层（卷积层、线性层等）、激活函数 (activation function)，甚至其他`nn.Module`实例（子模块）。这些组件通常被作为类实例（`self`）的属性进行赋值。
2. **`forward(self, input_data)`:** 此方法定义了网络的*前向传播*。它规定了输入数据（`input_data`）如何流经在`__init__`中定义的层和组件。`forward`方法接收一个或多个输入张量，并返回一个或多个输出张量。PyTorch的Autograd系统会根据此`forward`方法中执行的操作自动构建计算图，从而实现自动微分。

以下是一个自定义模块的骨架：

```python
import torch
import torch.nn as nn

class MySimpleNetwork(nn.Module):
    def __init__(self):
        super(MySimpleNetwork, self).__init__()
        # 在此处定义层或组件
        # 示例：一个线性层
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        # 示例：一个激活函数实例
        self.activation = nn.ReLU()

    def forward(self, x):
        # 定义数据流经组件的方式
        x = self.layer1(x)
        x = self.activation(x)
        return x

# 实例化网络
model = MySimpleNetwork()
print(model)
```

执行此代码将打印出网络结构的表示，展示`nn.Module`如何帮助组织你的组件。

### 参数 (parameter)和子模块

`nn.Module`的一个重要特点是其自动注册和管理可学习参数的能力。当你在`__init__`方法中将一个PyTorch层（如`nn.Linear`、`nn.Conv2d`等）的实例作为属性赋值时，`nn.Module`会识别该层的内部参数（权重 (weight)和偏置 (bias)）。

这些参数是`torch.nn.Parameter`类的实例，它是`torch.Tensor`的一个特殊子类。主要区别在于`Parameter`对象默认自动设置`requires_grad=True`，并且它们会注册到父`nn.Module`中。这种注册使得PyTorch可以轻松收集模型的所有可学习参数，这对于在训练期间将它们传递给优化器来说非常重要。

你也可以直接使用`nn.Parameter`定义你自己的自定义可学习参数：

```python
class CustomModuleWithParameter(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个可学习的参数张量
        self.my_weight = nn.Parameter(torch.randn(5, 2))
        # 一个普通的张量属性（不会自动跟踪用于优化）
        self.my_info = torch.tensor([1.0, 2.0])

    def forward(self, x):
        # 示例用法
        return torch.matmul(x, self.my_weight)

module = CustomModuleWithParameter()

# 访问模块跟踪的参数
for name, param in module.named_parameters():
    print(f"Parameter name: {name}, Shape: {param.shape}, Requires grad: {param.requires_grad}")
```

注意`my_weight`被列为参数，而`my_info`则没有。这种自动跟踪简化了管理深度网络中可能数千或数百万参数的过程。

### `nn.Module`的主要功能

除了定义结构和管理参数 (parameter)之外，`nn.Module`还提供了一些有用的方法，可供你的自定义类继承：

- `parameters()`: 返回模块内（包括子模块中的）所有`nn.Parameter`对象的迭代器。这通常用于将模型的参数提供给优化器。
- `named_parameters()`: 类似于`parameters()`，但会生成（参数名，参数对象）的元组。这有助于检查或有选择地修改特定参数。
- `children()`: 返回直接子模块（定义为属性的子模块）的迭代器。
- `modules()`: 返回网络中所有模块的迭代器，从模块自身开始，然后递归遍历所有子模块。
- `state_dict()`: 返回一个Python字典，该字典包含模块的完整状态，主要将每个参数和缓冲区名称映射到其对应的张量。这对于保存模型权重 (weight)非常重要。
- `load_state_dict()`: 将状态（通常来自保存的文件）加载回模块，恢复参数和缓冲区。
- `to(device)`: 将模块的所有参数和缓冲区移动到指定设备（例如，GPU的`'cuda'`或CPU的`'cpu'`）。这对于硬件加速非常重要。
- `train()`: 将模块及其子模块设置为训练模式。这会影响像Dropout和BatchNorm这样的层，它们在训练和评估期间表现不同。
- `eval()`: 将模块及其子模块设置为评估模式。

理解`nn.Module`非常重要，因为它建立了在PyTorch中定义任何神经网络 (neural network)架构的标准模式。在接下来的章节中，我们将使用这个基类来构建包含各种层、激活函数 (activation function)和损失函数 (loss function)的网络。

## 参考资料

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Developers (2024)
  PyTorch `nn.Module`基础类的官方文档，详细介绍了其API和功能。
- [Defining a Neural Network](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  一个入门教程，演示如何在PyTorch中使用`nn.Module`构建和组织神经网络。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，提供了神经网络、其组成部分和学习原理的理论背景，`nn.Module`有助于实现这些。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本全面的实践指南，涵盖了使用PyTorch构建和训练深度学习模型，包含大量`nn.Module`的使用示例。
