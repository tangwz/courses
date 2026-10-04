---
course: "getting-started-with-pytorch"
chapter: "building-models-torch-nn"
lesson: "sequential-containers"
sourceId: 2161
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/sequential-containers"
title: "简单模型的顺序容器"
description: "使用`nn.Sequential`快速定义线性堆叠层的模型。"
order: 5
plots: []
sourceHash: "330c3fee71d03da918bc8148325a72eb00aebe7b1add1266550c7ebd131002cd"
sourceCorrections: []
---

虽然通过子类化`torch.nn.Module`定义自定义网络架构提供了最大程度的灵活性，但许多常见模型都涉及层的一个直观序列，其中一层的输出直接作为下一层的输入。对于这些线性堆叠，PyTorch提供了一个方便的容器：`torch.nn.Sequential`。

`nn.Sequential`作为一个包装器，接收一个有序的模块序列（如层和激活函数 (activation function)），并在输入数据通过时，以该特定顺序执行它们。可以将其视为为您的数据转换构建一个流程。当您不需要复杂的数据流逻辑、跳跃连接或多输入/多输出路径时，这种方法简化了模型定义。

### 使用`nn.Sequential`定义模型

您可以通过将您想要包含的模块作为参数 (parameter)传递给其构造函数来创建`Sequential`模型。顺序很重要，因为它决定了数据流。

让我们构建一个简单的两层前馈网络，它接收一个784维的输入（例如扁平化的MNIST图像），通过一个包含128个单元和ReLU激活的隐藏层，最后生成一个10维输出（用于10个类别）。

```python
import torch
import torch.nn as nn
from collections import OrderedDict

# 定义输入、隐藏层和输出维度
input_size = 784
hidden_size = 128
output_size = 10

# 方法1：直接将模块作为参数传递
model_v1 = nn.Sequential(
    nn.Linear(input_size, hidden_size), # 第1层：线性变换
    nn.ReLU(),                         # 激活函数1：非线性
    nn.Linear(hidden_size, output_size) # 第2层：线性变换
)

# 打印模型结构
print("Model V1 (Unnamed Layers):")
print(model_v1)

# 示例用法：创建一个虚拟输入张量
# 假设批量大小为64
dummy_input = torch.randn(64, input_size)
output = model_v1(dummy_input)
print("\nOutput shape:", output.shape) # 预期：torch.Size([64, 10])
```

这创建了一个模型，其中输入数据首先经过`nn.Linear(784, 128)`，然后应用`nn.ReLU()`激活，最后结果通过`nn.Linear(128, 10)`。请注意，定义是多么紧凑。`Sequential`容器自动处理将一个模块的输出作为下一个模块的输入传递。

### 在`nn.Sequential`中命名层

尽管前一种方法可行，但层只被分配了默认的数字索引（`0`、`1`、`2`等）。这可能会使得后续调试或访问特定层变得更难。为了清晰度和可访问性，更好的做法是使用Python `collections`模块中的`OrderedDict`来为您的层提供名称。

```python
# 方法2：使用OrderedDict进行命名层
model_v2 = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_size)), # 全连接层1
    ('relu1', nn.ReLU()),                      # ReLU激活
    ('fc2', nn.Linear(hidden_size, output_size)) # 全连接层2
]))

# 打印模型结构
print("\nModel V2 (Named Layers):")
print(model_v2)

# 现在可以通过名称访问特定层
print("\nAccessing fc1 weights shape:", model_v2.fc1.weight.shape)
# 如果需要，也可以使用整数索引访问
print("Accessing layer at index 0:", model_v2[0])
# 或者如果使用OrderedDict，直接通过字符串名称访问
print("Accessing layer by name 'relu1':", model_v2.relu1)
```

使用`OrderedDict`保留了插入顺序（这对`nn.Sequential`非常重要），同时允许您引用诸如`model_v2.fc1`或`model_v2.relu1`之类的层。这显著提高了代码的可读性和可维护性，特别是对于稍长的序列，使得检查模型的特定部分变得更容易。

> 数据流经使用命名层通过`nn.Sequential`定义的`model_v2`。输入按线性顺序通过`fc1`、`relu1`和`fc2`。

### 何时使用`nn.Sequential`

`nn.Sequential`特别适合于：

1. **简单的全连接网络：** 模型中的层线性堆叠，没有分支或跳过，例如基本的多层感知机（MLP）或某些CNN的初始特征提取阶段。
2. **定义可复用模块：** 创建自包含的层模块（例如，一个包含`Conv2d`、`BatchNorm2d`和`ReLU`的卷积模块），然后可以将其作为单个模块整合到更大的自定义`nn.Module`结构中。
3. **快速原型开发：** 快速组装标准架构以测试想法或建立基线。

### 局限性

`nn.Sequential`的主要局限在于其严格的线性性质。它假定一个单一输入和一个单一输出，数据顺序流经所有包含的模块。您不能直接使用它来定义具有更复杂拓扑的模型，例如：

- **跳跃连接：** 像ResNet这样的架构，其中较早层的输出被添加到较晚层的输出中，这需要在自定义的`forward`方法中显式实现。
- **多输入或多输出：** 处理几个不同输入流或生成多个输出张量的模型不能仅凭`nn.Sequential`来表示。
- **共享层：** 架构中完全相同的层实例在网络拓扑的不同点被应用。
- **条件逻辑：** 任何数据流依赖于运行时条件或需要对从一层的输出到下一层输入的数据进行操作的场景。

对于任何表现出这些特点的架构，您必须通过子类化`torch.nn.Module`并自行实现`forward`方法来定义一个自定义模型，这将给予您对数据流的完全控制，如前所述在“定义自定义网络架构”一节中讨论的。

总之，`nn.Sequential`提供了一种清晰高效的方法来定义线性堆叠神经网络 (neural network)层的常见模式。它是一个有价值且方便的工具，适用于更简单的架构和组件模块，补充了自定义`nn.Module`类这种更灵活的方法。现在您可以使用`nn.Module`或`nn.Sequential`定义模型结构了，下一步是定义模型将优化的目标函数，这将引出损失函数 (loss function)。

## 参考资料

- [torch.nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) — PyTorch Authors (2024)
  Publisher: PyTorch Foundation
  `torch.nn.Sequential` 容器的官方文档，详细介绍了其功能和构建线性神经网络的方法。
- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Authors (2024)
  Publisher: PyTorch Foundation
  `torch.nn.Module` 基类的官方文档，解释了如何在 PyTorch 中构建自定义神经网络层和模型，是 `nn.Sequential` 的基础。
- [Learn the Basics: Build the Neural Network](https://pytorch.org/tutorials/beginner/basics/build_model_tutorial.html) — PyTorch Authors (2021)
  Publisher: PyTorch
  PyTorch 官方入门教程，演示了如何使用 `nn.Module` 和 `nn.Sequential` 构建神经网络，展示了它们的实际应用。
