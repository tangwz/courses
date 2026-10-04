---
course: "deep-learning-fundamentals-keras"
chapter: "building-networks-keras"
lesson: "dense-layers"
sourceId: 4887
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/dense-layers"
title: "常见层类型：全连接层"
description: "在 Keras 中了解并实现全连接（Dense）层。"
order: 2
plots: []
sourceHash: "846e55fede95e598221a3d71995c5a02dcefd161c0ec5abca5c2e993834fc931"
sourceCorrections: []
---

神经网络 (neural network)中最基本且最常遇到的层类型是 `Dense` 层，通常被称为全连接层。它作为神经网络架构的基本构建块。

`Dense` 层实现以下运算：

$输出 = 激活(点积(输入, 核) + 偏置)$

让我们逐一分析：

1. `输入`：这是该层的张量输入。对于模型中的第一层，你需要指定其形状。对于后续层，Keras 会根据前一层的输出形状自动推断输入形状。
2. `核`：这是该层的权重 (weight)矩阵。它是训练过程中学习到的核心组成部分之一。该层会将输入张量乘以这个核矩阵。
3. `偏置`：这是一个偏置 (bias)向量 (vector)，也是一个可学习的参数 (parameter)。它被加到点积的结果上。添加偏置会增加模型的灵活性，使其能更好地拟合数据。可以把它看作简单线性方程 $y = mx + b$ 中的 y 轴截距。
4. `激活`：这是一个应用于结果的逐元素激活函数 (activation function)。激活函数为网络引入非线性，使其能够学习复杂的模式。我们将在下一节详细介绍激活函数，常见的例子包括 ReLU、Sigmoid 和 Softmax。

`Dense` 层的主要特点是其“全连接”性质。`Dense` 层中的每个神经元（或单元）都接收来自前一层中*每个*神经元的输入。这种全面的连接性使该层能够学习前一层输出所表示的所有特征之间的相互作用。

> 全连接 `Dense` 层的可视化图。前一层的每个神经元（N1、N2、N3）都连接到 `Dense` 层中的每个单元（U1、U2）。

### 在 Keras 中实现全连接层

在 Keras 中，你可以使用 `keras.layers.Dense` 轻松添加 `Dense` 层。你需要指定的最重要参数 (parameter)是 `units`。

- `units`：这个正整数定义了输出空间的维度，这相当于该层中神经元的数量。例如，`units=64` 表示该层将输出一个形状为 `(batch_size, 64)` 的张量。`units` 的选择影响了层的表示能力。更多的单元可能使层学习更复杂的模式，但也会增加参数数量和过拟合 (overfitting)的风险。
- `activation`：此参数指定要使用的激活函数 (activation function)。你可以提供内置激活函数（如 `'relu'` 或 `'softmax'`）的名称，或传递一个可调用激活函数对象。如果你不指定激活函数，则不应用任何激活（它使用线性激活，$a(x)=x$）。
- `input_shape`：对于 `Sequential` 模型中的*第一*层，Keras 需要知道输入数据的形状。你可以使用 `input_shape` 参数指定这一点，它应该是一个元组（例如，对于展平的 28x28 图像，`input_shape=(784,)`）。你无需包含批次维度。对于后续层，Keras 会自动推断输入形状。

以下是如何在简单的 `Sequential` 模型中使用 `Dense` 层：

```python
import keras
from keras import layers

# 定义输入形状（例如，用于展平的 MNIST 图像）
input_shape = (784,)

model = keras.Sequential(
    [
        # 第一个全连接层：需要输入形状
        layers.Dense(units=128, activation='relu', input_shape=input_shape, name='hidden_layer_1'),

        # 第二个全连接层：输入形状从前一层的输出（128 个单元）推断
        layers.Dense(units=64, activation='relu', name='hidden_layer_2'),

        # 输出层：10 个单元（例如，用于 10 个数字类别），softmax 激活用于分类
        layers.Dense(units=10, activation='softmax', name='output_layer')
    ],
    name="simple_mlp"
)

# 显示模型的架构和参数
model.summary()
```

执行 `model.summary()` 将输出详细说明层、它们的输出形状以及可训练参数的数量。请注意参数数量如何与输入维度、单元数量以及每层的偏置 (bias)相关。例如，`hidden_layer_1` 具有 $784 \times 128$ 个权重 (weight)加上 $128$ 个偏置。

### 作用和用途

全连接层用途多样：

- **隐藏层：** 它们构成了标准前馈网络（多层感知器或 MLP）的骨干。堆叠多个带有非线性激活的 `Dense` 层，使网络能够学习输入数据的分层表示。
- **输出层：** 它们常被用作分类和回归模型中的最终层。
  - 对于二元分类，一个具有 1 个单元和 `sigmoid` 激活的 `Dense` 层是常见的。
  - 对于多类别分类，使用一个具有 `N` 个单元（N 是类别数量）和 `softmax` 激活的 `Dense` 层。
  - 对于回归任务，使用一个具有 1 个单元（或更多，用于多输出回归）且通常没有激活（线性激活）的 `Dense` 层。
- **特征提取后：** 在更复杂的架构中，例如卷积神经网络 (neural network) (CNN) 或循环神经网络 (RNN)，`Dense` 层通常被放置在主要的特征提取块（卷积层或循环层）*之后*，以便根据提取的特征执行最终的分类或回归。

`Dense` 层是你在 Keras 中构建各种神经网络架构时将广泛使用的基本组成部分。了解其操作方式以及如何配置它是掌握实际深度学习 (deep learning)开发的重要一步。

## 参考资料

- [keras.layers.Dense](https://keras.io/api/layers/core_layers/dense/) — Keras Team (2024)
  Keras Dense层的官方文档，提供详细的API用法和参数。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了神经网络的综合学术基础，包括全连接层的数学原理。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=3VwOEAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了使用Keras实现Dense层和神经网络的实用示例和解释。（第4版）
