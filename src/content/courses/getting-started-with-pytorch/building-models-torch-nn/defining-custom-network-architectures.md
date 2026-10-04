---
course: "getting-started-with-pytorch"
chapter: "building-models-torch-nn"
lesson: "defining-custom-network-architectures"
sourceId: 2152
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/defining-custom-network-architectures"
title: "定义自定义网络架构"
description: "如何通过继承 `nn.Module` 来创建自己的神经网络结构。"
order: 2
plots: []
sourceHash: "63811b9d3e7b973e5cebddaf3ce9a84b79bf584980d377afcd7c6d1bd063e472"
sourceCorrections: []
---

许多网络架构需要比简单的线性层堆叠更复杂的设计。尽管 `torch.nn.Sequential` 对于线性模型很方便，但更复杂的设计常常是必需的。你可能需要跳跃连接（如在 ResNet 中）、多输入/输出路径，或以非顺序方式使用的层。在这种情况下，通过继承 `torch.nn.Module` 来定义你自己的自定义网络架构就变得必不可少。这种方法在指定数据如何通过模型流动方面提供了最大的灵活性。

这个基本过程包含两个主要步骤：

1. **在构造函数 (`__init__`) 中定义层**：创建一个继承自 `torch.nn.Module` 的 Python 类。在其 `__init__` 方法中，你必须首先调用父类的构造函数 (`super().__init__()`)。然后，实例化网络所需的所有层（例如 `nn.Linear`、`nn.Conv2d`、`nn.ReLU` 等），并将它们作为类实例的属性（使用 `self`）进行分配。这些层就成为你的自定义模块的子模块。
2. **在 `forward` 方法中定义数据流**：为你的类实现 `forward` 方法。此方法将输入张量作为参数 (parameter)，并定义输入数据如何通过你在 `__init__` 中定义的层进行传播。此方法的输出是你的网络对于给定输入的最终输出。PyTorch 的 Autograd 系统会根据此 `forward` 方法中执行的操作自动构建计算图。

我们从一个基本例子开始：一个实现为自定义模块的简单线性回归模型。

```python
import torch
import torch.nn as nn

class SimpleLinearModel(nn.Module):
    def __init__(self, input_features, output_features):
        # 调用父类构造函数
        super().__init__()
        # 定义单个线性层
        self.linear_layer = nn.Linear(input_features, output_features)
        print(f"已初始化 SimpleLinearModel，输入特征数={input_features}，输出特征数={output_features}")
        print(f"已定义层: {self.linear_layer}")

    def forward(self, x):
        # 定义前向传播：将输入通过线性层
        print(f"前向传播输入形状: {x.shape}")
        output = self.linear_layer(x)
        print(f"前向传播输出形状: {output.shape}")
        return output

# --- 使用示例 ---
# 定义输入和输出维度
in_dim = 10
out_dim = 1

# 实例化自定义模型
model = SimpleLinearModel(input_features=in_dim, output_features=out_dim)

# 创建一些模拟输入数据（batch_size=5，特征数=10）
dummy_input = torch.randn(5, in_dim)
print(f"\n模拟输入张量形状: {dummy_input.shape}")

# 将数据通过模型
output = model(dummy_input)
print(f"模型输出张量形状: {output.shape}")

# 检查参数（自动注册）
print("\n模型参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  名称: {name}, 形状: {param.shape}")
```

在此示例中：

- `SimpleLinearModel` 继承自 `nn.Module`。
- `__init__` 调用 `super().__init__()` 并定义 `self.linear_layer = nn.Linear(...)`。此层现在是一个已注册的子模块。
- `forward(self, x)` 接收输入 `x` 并将其通过 `self.linear_layer`，然后返回结果。

PyTorch 会自动追踪 `nn.Linear` 层的参数（权重 (weight)和偏置 (bias)），因为它们被作为属性分配在 `nn.Module` 子类中。我们可以通过查看 `model.parameters()` 或 `model.named_parameters()` 来验证这一点。

### 构建多层感知机 (MLP)

现在，我们来构建一个稍微复杂一点的模型，一个两层 MLP，在层之间带有一个 ReLU 激活函数 (activation function)。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F # 通常用于函数式 API，如激活函数

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 定义层
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU() # 将激活函数定义为层
        self.layer2 = nn.Linear(hidden_size, output_size)
        print(f"已初始化 SimpleMLP: 输入={input_size}, 隐藏层={hidden_size}, 输出={output_size}")
        print(f"层 1: {self.layer1}")
        print(f"激活函数: {self.activation}")
        print(f"层 2: {self.layer2}")

    def forward(self, x):
        # 定义前向传播序列
        print(f"前向传播输入形状: {x.shape}")
        x = self.layer1(x)
        print(f"经过层 1 后的形状: {x.shape}")
        x = self.activation(x) # 应用 ReLU 激活
        # 替代方案：x = F.relu(x) # 使用函数式 API
        print(f"经过激活函数后的形状: {x.shape}")
        x = self.layer2(x)
        print(f"经过层 2（输出）后的形状: {x.shape}")
        return x

# --- 使用示例 ---
# 定义维度
in_size = 784 # 示例：展平的 28x28 图像
hidden_units = 128
out_size = 10   # 示例：用于分类的 10 个类别

# 实例化 MLP
mlp_model = SimpleMLP(input_size=in_size, hidden_size=hidden_units, output_size=out_size)

# 创建模拟输入（批大小=32）
dummy_mlp_input = torch.randn(32, in_size)
print(f"\n模拟 MLP 输入形状: {dummy_mlp_input.shape}")

# 前向传播
mlp_output = mlp_model(dummy_mlp_input)
print(f"MLP 输出形状: {mlp_output.shape}")

# 检查参数
print("\nMLP 模型参数:")
for name, param in mlp_model.named_parameters():
     if param.requires_grad:
        print(f"  名称: {name}, 形状: {param.shape}")
```

这里，`forward` 方法明确规定了序列：输入 -> `layer1` -> `activation` -> `layer2` -> 输出。请注意，`nn.ReLU` 等激活函数通常也在 `__init__` 中定义为层，并在 `forward` 中调用。另外，你也可以直接在 `forward` 方法中使用其函数式等效项（例如，导入 `torch.nn.functional as F` 后使用 `F.relu(x)`），特别是对于没有可学习参数 (parameter)的激活函数。

### 可视化 MLP 结构

我们可以可视化 `SimpleMLP` 的 `forward` 方法中定义的数据流。

> 数据流经 `SimpleMLP` 模型，如其 `forward` 方法中所定义。

### 继承 `nn.Module` 的优点

- **灵活性**：这是主要优势。你可以实现任何架构，包括具有多个输入/输出、残差连接（其中输入被加回到后续层的输出）、共享层或 `forward` 传递中自定义操作的架构。`nn.Sequential` 仅限于严格的线性层序列。
- **可读性和组织性**：复杂的架构通常在类结构中组织时更容易理解，其中层在 `__init__` 中定义，它们的交互方式在 `forward` 中定义。
- **参数 (parameter)管理**：PyTorch 会自动发现并注册在 `__init__` 方法中作为属性分配的任何 `nn.Module`（例如 `self.layer1 = nn.Linear(...)`）。这意味着 `model.parameters()` 将正确地给出所有子模块的所有可学习参数（权重 (weight)、偏置 (bias)），使其可以轻松传递给优化器。
- **嵌套**：自定义模块可以包含其他模块（包括 `nn.Sequential` 或其他自定义模块），从而允许你构建分层和可重用的组件。

通过继承 `nn.Module`，你可以完全控制网络的结构，从而实现针对特定任务的复杂深度学习 (deep learning)模型。这种方法是构建更复杂的前馈网络的标准做法。

## 参考资料

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Authors (2024)
  Publisher: PyTorch Foundation
  描述了神经网络的基类，解释了如何在__init__中定义层以及如何在forward中定义数据流。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  为深度神经网络提供了全面的理论基础，包括多层感知机和通用架构原则。
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.48550/arXiv.1512.03385](https://doi.org/10.48550/arXiv.1512.03385)
  介绍了残差连接的概念，这是一种关键的高级架构，需要自定义nn.Module的灵活性。
- [Learn the Basics: Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) — PyTorch Authors (2021)
  Publisher: PyTorch
  一个实用的教程，演示了如何使用torch.nn.Module和torch.nn.Sequential构建各种神经网络架构。
