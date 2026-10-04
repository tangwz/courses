---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-nn-module-for-keras-users"
lesson: "keras-layers-vs-torch-nn-module"
sourceId: 5744
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/keras-layers-vs-torch-nn-module"
title: "定义网络组件：Keras 层与 torch.nn.Module"
description: "比较 TensorFlow Keras 和 PyTorch 的 `torch.nn.Module` 中层的定义及使用方式。"
order: 1
plots: []
sourceHash: "09aa374271b17cad4e72d3e8eb637e9b4694af54d25efb08cbbc23538aa54bf3"
sourceCorrections: []
---

从 TensorFlow Keras 转向 PyTorch 时，了解网络组件的定义方式是首要的一步。`tf.keras.layers.Layer` 是 Keras 中所有层的基类，这些层是模型的基本构成单元。PyTorch 通过其 `torch.nn.Module` 类提供了一个相似但有所不同的基准。每个神经网络 (neural network)模块，从单个层到整个复杂模型，都是 `nn.Module` 的子类。该类提供了构建神经网络所需的核心功能：它可以注册和管理参数 (parameter)（如权重 (weight)和偏置 (bias)），容纳子模块（其他 `nn.Module` 实例），并定义前向计算。

### `torch.nn.Module` 类：你的网络蓝图

将 `torch.nn.Module` 视为神经网络 (neural network)中任何具有可学习参数 (parameter)或执行独立计算步骤的部分的蓝图。其主要职责包括：

1. **参数管理**：`nn.Module` 可以包含 `torch.nn.Parameter` 实例。这些是特殊的张量，它们会被自动注册为模型参数。当你要求模型提供其参数时（例如，传递给优化器），`nn.Module` 会确保所有这些已注册的参数，包括嵌套子模块中的参数，都是可访问的。
2. **子模块组织**：一个 `nn.Module` 可以将其他 `nn.Module` 实例作为属性包含。这使你能够通过分层方式组合更简单的模块来构建复杂的架构。
3. **定义计算**：每个 `nn.Module` 子类通常会实现一个 `forward()` 方法。此方法接收输入张量并执行已定义的操作，返回输出张量。这与 Keras `Layer` 中的 `call()` 方法直接对应。

尽管 Keras 层有一个 `build()` 方法，常用于延迟权重 (weight)创建（即在首次得知输入形状时创建权重），但 PyTorch 模块通常在其 `__init__()` 构造函数中直接定义其层以及参数的形状。像 `nn.Linear` 这样的预构建层的权重在实例化时创建，这要求你届时指定输入特征维度。

### 在 `__init__` 中定义层和参数 (parameter)

在 PyTorch 中，标准做法是在模块的 `__init__` 构造函数中定义并初始化所有组成层和参数。当你将另一个 `nn.Module` 的实例（如 `nn.Linear` 或 `nn.Conv2d`）作为自定义模块的属性时，PyTorch 会自动将其识别为子模块。

让我们看一个简单例子。如果你想创建一个包含线性层和 ReLU 激活函数 (activation function)的模块：

```python
import torch
import torch.nn as nn

class SimplePyTorchModule(nn.Module):
    def __init__(self, input_features, output_features):
        super(SimplePyTorchModule, self).__init__()
        # 在此处定义子模块（层）
        self.linear_layer = nn.Linear(input_features, output_features)
        self.activation = nn.ReLU()

        # 自定义可学习参数示例
        # self.my_custom_bias = nn.Parameter(torch.zeros(output_features))

    def forward(self, x):
        # 使用子模块定义计算流程
        x = self.linear_layer(x)
        x = self.activation(x)
        # 如果 self.my_custom_bias 已定义：
        # x = x + self.my_custom_bias 
        return x

# 实例化模块
module = SimplePyTorchModule(input_features=10, output_features=5)
print(module)
```

输出:

```
SimplePyTorchModule(
  (linear_layer): Linear(in_features=10, out_features=5, bias=True)
  (activation): ReLU()
)
```

在这个 `__init__` 方法中：

- `super(SimplePyTorchModule, self).__init__()` 对于调用基类 `nn.Module` 的构造函数非常重要。
- `self.linear_layer = nn.Linear(input_features, output_features)` 创建了 PyTorch 预构建线性层的一个实例，并将其作为属性分配。它的参数（权重 (weight)和偏置 (bias)）会自动注册到 `SimplePyTorchModule`。
- `self.activation = nn.ReLU()` 创建了 ReLU 激活函数的一个实例。ReLU 本身没有可学习参数，但它仍然是一个 `nn.Module`。

这类似于你在自定义 `tf.keras.Layer` 子类的 `__init__` 或 `build` 方法中定义层的方式。然而，组合成更大的模型通常有所不同。在 Keras 中，你可能会将这些添加到 `tf.keras.Sequential` 模型中，或使用函数式 API 连接它们。在 PyTorch 中，你通常通过嵌套直接构建这些 `nn.Module` 结构。

### 实现 `forward()` 方法

`forward()` 方法是你定义模块实际计算的地方。它接受一个或多个输入张量，并返回一个或多个输出张量。你使用在 `__init__` 中定义的层和参数 (parameter)来转换输入。

继续 `SimplePyTorchModule`：

```python
# （在 SimplePyTorchModule 类内部）
# def forward(self, x):
#     x = self.linear_layer(x)
#     x = self.activation(x)
#     return x
```

这里，`self.linear_layer(x)` 调用了 `nn.Linear` 实例的 `forward` 方法。`nn.Module` 实例是可调用的；当你调用 `module_instance(input)` 时，它会在内部调用 `module_instance.forward(input)`。

PyTorch 的一个重要特点是其动态计算图（即“按运行定义”）。`forward()` 方法只是普通的 Python 代码。这意味着你可以使用标准的 Python 控制流（循环、条件语句）来定义复杂、自适应的计算。运算图是在 `forward()` 方法执行时即时构建的。这与 TensorFlow 的传统图模式（即“按定义运行”）形成对比，在传统模式下，图通常是先静态定义的（尽管 TensorFlow 2.x 默认的 Eager Execution 模式行为更像 PyTorch 的“按运行定义”）。

### 可学习参数 (parameter)：`torch.nn.Parameter`

尽管像 `nn.Linear` 这样的预构建层管理着自己的参数，但有时你需要直接在模块中定义自己的自定义可学习参数。为此，你需要用 `torch.nn.Parameter` 封装一个 `torch.Tensor`。这会告诉 PyTorch 该张量应被视为模型参数，意味着它的 `requires_grad` 属性默认将被设置为 `True`（因此在反向传播 (backpropagation)期间会为其计算梯度），并且它将被包含在 `model.parameters()` 返回的列表中。

```python
import torch
import torch.nn as nn

class ModuleWithCustomParameter(nn.Module):
    def __init__(self, num_features):
        super(ModuleWithCustomParameter, self).__init__()
        # 每个特征的可学习缩放因子
        self.scale = nn.Parameter(torch.ones(num_features)) 
        # 每个特征的可学习偏置
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # x 的预期形状为 (batch_size, num_features)
        return x * self.scale + self.bias

custom_param_module = ModuleWithCustomParameter(5)
# 你可以查看其参数：
for name, param in custom_param_module.named_parameters():
    print(f"{name}: {param.data}") 
```

这与在 Keras `Layer` 的 `build` 方法中使用 `self.add_weight()` 来创建和注册可训练权重 (weight)相似。

### 嵌套模块以构建分层架构

`nn.Module` 的强大能力之一是它能够包含其他 `nn.Module` 实例。这使你能够通过分层方式组合更简单、可重用的块来构建复杂模型。父模块会自动发现并管理其嵌套子模块中的所有参数 (parameter)。

考虑构建一个稍微复杂一些的网络，它使用我们之前定义的 `SimplePyTorchModule` 作为构建块：

```python
class AdvancedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(AdvancedNetwork, self).__init__()
        # 将 SimplePyTorchModule 嵌套为一个“块”
        self.block1 = SimplePyTorchModule(input_dim, hidden_dim1)
        self.intermediate_linear = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = self.block1(x) # 使用 SimplePyTorchModule 的前向传播
        x = self.intermediate_linear(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# 实例化示例
adv_net = AdvancedNetwork(input_dim=20, hidden_dim1=15, hidden_dim2=10, output_dim=2)
print(adv_net)
```

输出:

```
AdvancedNetwork(
  (block1): SimplePyTorchModule(
    (linear_layer): Linear(in_features=20, out_features=15, bias=True)
    (activation): ReLU()
  )
  (intermediate_linear): Linear(in_features=15, out_features=10, bias=True)
  (relu): ReLU()
  (output_layer): Linear(in_features=10, out_features=2, bias=True)
)
```

如你所见，`AdvancedNetwork` 包含了 `block1`，它是 `SimplePyTorchModule` 的一个实例。来自 `block1.linear_layer`、`intermediate_linear` 和 `output_layer` 的所有参数都将成为 `adv_net.parameters()` 的一部分。这种可组合性是有效组织 PyTorch 代码的核心所在。

下图说明了一个自定义 PyTorch `nn.Module` 的结构，该模块包含其他模块（包括预构建的和其他可能的自定义模块）和参数。

> 一个 PyTorch `nn.Module` (`MyNetwork`) 在 `__init__` 中定义了其组件（子模块如 `CustomBlockA`、`nn.ReLU`、`nn.Linear`，以及直接的 `nn.Parameter` 如 `global_bias`），并在 `forward` 中定义了它们的计算流程。`CustomBlockA` 本身是另一个 `nn.Module`，展示了嵌套的用法。

### `torch.nn` 中的预构建层

PyTorch 在 `torch.nn` 包中提供了一系列丰富的预构建层，例如：

- `nn.Linear`：全连接层。
- `nn.Conv1d`、`nn.Conv2d`、`nn.Conv3d`：用于不同维度的卷积层。
- `nn.RNN`、`nn.LSTM`、`nn.GRU`：循环层。
- `nn.BatchNorm1d`、`nn.BatchNorm2d`：批归一化 (normalization)层。
- `nn.Dropout`：Dropout 层。
- 激活函数 (activation function)，如 `nn.ReLU`、`nn.Sigmoid`、`nn.Tanh`、`nn.Softmax`（尽管许多激活函数在 `torch.nn.functional` 中也可用，并且可以直接在 `forward` 方法中应用）。

这些预构建层本身都是 `nn.Module` 的子类。你通过在模块的 `__init__` 方法中实例化它们，然后在 `forward` 方法中使用适当的输入来调用它们。这与你在 TensorFlow 中使用 `tf.keras.layers` 中的层的方式相似。后续章节将更详细地介绍这些常见层类型，比较它们的 PyTorch 实现与 Keras 对应部分。

从 Keras 的 `Layer` 转向 PyTorch 的 `nn.Module` 需要接受一种结构，在这种结构中，你在 `__init__` 中明确定义网络模块的组成部分，并在 `forward` 中定义它们的计算流程。这种明确的定义提供了细粒度控制，并与 Python 的动态能力自然结合，使得构建和调试即使非常复杂的模型架构也变得简单。

## 参考资料

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Team (2024)
  PyTorch中所有神经网络模块基类的官方文档，涵盖其功能和设计模式。
- [tf.keras.layers.Layer class](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer) — TensorFlow Team (2024)
  Keras `Layer` 类的官方文档，为从TensorFlow过渡到PyTorch的开发者提供了直接的比较点。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本内容详实的书籍，为使用PyTorch构建神经网络和深度学习模型提供了实用指导。
- [Defining a Neural Network](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tut.html) — PyTorch Team (2024)
  Publisher: PyTorch
  一个官方PyTorch教程，展示了如何使用`torch.nn.Module`类及其`forward`方法来创建和构建神经网络。
