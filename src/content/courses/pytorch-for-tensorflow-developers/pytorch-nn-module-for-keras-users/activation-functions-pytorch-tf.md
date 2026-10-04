---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-nn-module-for-keras-users"
lesson: "activation-functions-pytorch-tf"
sourceId: 5747
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/activation-functions-pytorch-tf"
title: "激活函数：比较与分析"
description: "识别并使用 PyTorch 中的常用激活函数，了解它们在 TensorFlow Keras 中的对应方式。"
order: 4
plots: ["plots/5747-0.json"]
sourceHash: "1379868c96094fb64a1ebc07ffffbe3d15d359b7ac7d4da2d3c9e3da272478f9"
sourceCorrections: []
---

激活函数 (activation function)是神经网络 (neural network)中的非线性入口，使其能够学习复杂的模式。如果你熟悉 Keras 中激活函数的使用，会发现 PyTorch 提供类似的功能，但有其自身的应用习惯。PyTorch 中激活函数的实现和使用方法，并与 Keras 中的对应部分进行比较。

在 Keras 中，你通常通过以下两种方式之一指定激活函数：

1. 作为层的字符串参数 (parameter)，例如 `Dense(units=64, activation='relu')`。
2. 作为独立的层对象，例如 `tf.keras.layers.ReLU()`。

PyTorch 提供两种主要的激活函数应用方法，这两种方法都常用：

1. **函数式 API (`torch.nn.functional`)**：大多数激活函数都可以在 `torch.nn.functional` 模块中作为简单的函数使用（通常导入为 `F`）。这些函数接收张量作为输入，并返回激活后的张量。这种做法在自定义 `nn.Module` 类的 `forward` 方法中很常见，因为激活函数通常是无状态的操作。
2. **模块化 API (`torch.nn`)**：许多激活函数也有对应的 `nn.Module` 类（例如 `nn.ReLU()`、`nn.Sigmoid()`）。这些类可以被实例化并像其他任何层一样添加到你的网络中，这在构建 `nn.Sequential` 模型或当激活函数可能包含可学习参数（例如 `nn.PReLU`）时特别有用。

让我们来看看一些常用激活函数及其使用方式的比较。

### 常用激活函数 (activation function)

下面是广泛使用的激活函数的一个比较性概述：

#### 1. ReLU（修正线性单元）

ReLU 函数因其简洁和有效性而成为热门选择，有助于解决梯度消失问题。
它的定义如下：


$$
\text{ReLU}(x) = \max(0, x)
$$


- **Keras**：

  - 字符串：层中的 `activation='relu'`。
  - 层：`tf.keras.layers.ReLU()`
- **PyTorch**：

  - 函数式：`torch.nn.functional.relu(input_tensor)`
  - 模块：`nn.ReLU()`

  ```python
  import torch
  import torch.nn as nn
  import torch.nn.functional as F

  # 示例数据
  x = torch.randn(2, 3) # 批次大小为 2，每个样本 3 个特征

  # 函数式 ReLU
  output_functional = F.relu(x)

  # 模块式 ReLU
  relu_module = nn.ReLU()
  output_module = relu_module(x)

  print("输入:\n", x)
  print("函数式 ReLU 输出:\n", output_functional)
  print("模块式 ReLU 输出:\n", output_module)
  ```

#### 2. Sigmoid

Sigmoid 函数将值压缩到 0 到 1 之间。它常用于二分类问题的输出层。
其公式为：


$$
\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$


- **Keras**：
  - 字符串：`activation='sigmoid'`
  - 层：`tf.keras.layers.Activation('sigmoid')` 或 `tf.keras.activations.sigmoid`
- **PyTorch**：
  - 函数式：`torch.sigmoid(input_tensor)` 或 `F.sigmoid(input_tensor)`
  - 模块：`nn.Sigmoid()`

#### 3. Tanh（双曲正切）

Tanh 将值压缩到 -1 到 1 之间。因其输出以零为中心，故在隐藏层中常比 Sigmoid 更受青睐。
公式为：


$$
\text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$


- **Keras**：
  - 字符串：`activation='tanh'`
  - 层：`tf.keras.layers.Activation('tanh')` 或 `tf.keras.activations.tanh`
- **PyTorch**：
  - 函数式：`torch.tanh(input_tensor)` 或 `F.tanh(input_tensor)`
  - 模块：`nn.Tanh()`

#### 4. Softmax

Softmax 函数通常用于多分类问题的输出层。它将原始分数向量 (vector)（logits）转换为概率分布。
对于向量 $x = (x_1, x_2, \ldots, x_J)$， $x_i$ 的 Softmax 值为：


$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{J} e^{x_j}}
$$


- **Keras**：

  - 字符串：`activation='softmax'`
  - 层：`tf.keras.layers.Softmax()` 或 `tf.keras.activations.softmax`
  - Keras 层通常会正确推断 Softmax 的作用轴。
- **PyTorch**：

  - 函数式：`F.softmax(input_tensor, dim=...)`
  - 模块：`nn.Softmax(dim=...)`
  - **注意**：在 PyTorch 中，你*必须*为 `softmax` 指定 `dim` 参数 (parameter)。对于输入为 `(batch_size, num_classes)` 的典型分类任务，你将使用 `dim=1` 将 Softmax 应用于 `num_classes` 维度。

  ```python
  # 示例：PyTorch 中的 Softmax
  logits = torch.randn(2, 5) # 批次大小为 2，5 个类别

  # 函数式 Softmax
  probs_functional = F.softmax(logits, dim=1)

  # 模块式 Softmax
  softmax_module = nn.Softmax(dim=1)
  probs_module = softmax_module(logits)

  print("Logits:\n", logits)
  print("概率（函数式）:\n", probs_functional)
  print("概率（模块式）:\n", probs_module)
  print("每个样本的概率总和（应为 1）:\n", probs_module.sum(dim=1))
  ```

#### 5. LeakyReLU

LeakyReLU 是 ReLU 的一个变体，在单元不活跃时允许一个小的非零梯度，有助于缓解“ReLU 死亡”问题。
它的定义如下：


$$
\text{LeakyReLU}(x) = \begin{cases} x & \text{若 } x > 0 \\ \text{负斜率} \times x & \text{若 } x \le 0 \end{cases}
$$


`negative_slope` 是一个小的常数，通常默认为 0.01。

- **Keras**：
  - 层：`tf.keras.layers.LeakyReLU(alpha=0.01)`（其中 `alpha` 是负斜率）
- **PyTorch**：
  - 函数式：`F.leaky_relu(input_tensor, negative_slope=0.01)`
  - 模块：`nn.LeakyReLU(negative_slope=0.01)`

下图可视化了这些常用激活函数的一部分：ReLU、Sigmoid、Tanh 和 LeakyReLU（负斜率为 0.1 以便更好地观察）。



![常用激活函数](plots/5747-0.json)



> ReLU、Sigmoid、Tanh 和 LeakyReLU 激活函数的可视化。

### 在模型定义中应用激活函数 (activation function)

让我们看看在构建简单模型时如何集成这些函数，并比较 Keras 和 PyTorch 的方法。

**Keras 序贯模型：**

```python
# TensorFlow/Keras
import tensorflow as tf

keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
keras_model.summary()
```

**PyTorch `nn.Module`：**
在定义自定义 `nn.Module` 时，你通常会在 `forward` 方法中使用函数式激活函数。

```python
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchCustomModel(nn.Module):
    def __init__(self, input_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # 函数式 ReLU
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1) # 函数式 Softmax，指定维度
        return x

# 实例化模型
pytorch_model_custom = PyTorchCustomModel(input_features=784, num_classes=10)
print(pytorch_model_custom)
```

**PyTorch `nn.Sequential`：**
如果你使用 `nn.Sequential`，你将使用模块化版本的激活函数。

```python
# 使用 nn.Sequential 的 PyTorch 模型
pytorch_model_sequential = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),  # 模块式 ReLU
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1) # 模块式 Softmax，指定维度
)
print(pytorch_model_sequential)
```

### 在函数式和模块式激活函数 (activation function)之间选择

激活函数选择 `torch.nn.functional` 或 `nn.Module` 通常取决于编码风格和具体需求：

- **`torch.nn.functional`（例如 `F.relu`）**：

  - **无状态**：大多数激活函数是无状态的（它们没有可学习的权重 (weight)或在训练期间改变的内部状态，例如 `PReLU` 具有可学习参数 (parameter)）。在自定义 `nn.Module` 的 `forward` 方法中使用函数式可以稍微更简洁。
  - **灵活性**：如果需要，允许在 `forward` 传递中实现更动态的控制流。
- **`nn.Module`（例如 `nn.ReLU()`）**：

  - **一致性**：如果你倾向于在 `__init__` 方法中将网络的所有部分（层、激活函数）定义为模块，这提供了一个统一的结构。
  - **`nn.Sequential`**：在使用 `nn.Sequential` 构建模型时需要，因为它需要 `nn.Module` 实例。
  - **有状态激活函数**：对于确实具有可学习参数的激活函数（例如 `nn.PReLU`），你必须使用模块版本。

许多开发者倾向于在 `forward` 方法中为常见的无状态激活函数使用 `F.relu` 及类似的函数调用，因为这样可以避免为每个激活函数定义属性，从而使 `__init__` 更简洁。然而，使用 `nn.ReLU()` 是完全有效的，有时为了清晰起见或在使用 `nn.Sequential` 时更受青睐。

### 原地操作

一些 PyTorch 激活函数 (activation function)模块（例如 `nn.ReLU`）提供 `inplace=True` 选项：

```python
relu_inplace = nn.ReLU(inplace=True)
# 或者，对于函数式（尽管直接函数式使用较少见）
# x = F.relu_(x) # 注意原地函数式版本中的下划线
```

使用 `inplace=True` 会直接修改输入张量，这可以通过避免为输出分配新张量来节省内存。然而，这应谨慎使用：

1. 如果原始张量的值稍后需要，这可能会使调试变得更困难。
2. 如果网络其他部分需要原始张量进行梯度计算，它可能会干扰 PyTorch 的自动求导机制。例如，如果一个值在计算图的两个不同部分中使用，并且其中一个对其进行原地修改，则另一部分可能会得到意想不到的值。

通常，除非内存极其受限并且你理解其影响，否则使用非原地操作（默认 `inplace=False`）会更安全。

从 Keras 转换过来，你会发现 PyTorch 提供所有你习惯的激活函数。主要的适应点在于理解在哪里以及如何应用它们，无论是通过 `torch.nn.functional` 进行直接应用，还是作为 `nn.Module` 实例集成到你的网络架构中，并始终记住为 `softmax` 等函数指定 `dim` 参数 (parameter)。

## 参考资料

- [torch.nn.modules.activation - PyTorch Documentation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-and-normalization-layers) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch非线性激活模块的官方文档，详细介绍了nn.Module类的使用方法，并链接到torch.nn.functional的对应功能。
- [Keras API Reference: Layers | TensorFlow](https://www.tensorflow.org/api_docs/python/tf/keras/layers) — TensorFlow Authors (2023)
  Keras层（Layers）的官方文档，包括各种激活层如ReLU和通用Activation层，以及如何在Dense层中指定激活函数。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础性教材，为神经网络提供了全面的理论背景，包括对各种激活函数及其数学属性的详细解释。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本使用PyTorch实现深度学习模型的实践指南，提供了关于激活函数如何在PyTorch框架中应用的见解。
