---
course: "getting-started-with-tensorflow"
chapter: "building-models-keras"
lesson: "activation-functions"
sourceId: 1715
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/activation-functions"
title: "激活函数"
description: "了解并应用 ReLU、Sigmoid、Softmax 等常见激活函数。"
order: 4
plots: ["plots/1715-0.json"]
sourceHash: "ac18b8d2d8f42ab0aa7b5f41c9fffe8cd5339a3c8ed45b2b4a5df937a9bc08fd"
sourceCorrections: []
---

构建 Keras 神经网络 (neural network)时，需要使用其 Sequential 和 Functional API 堆叠层。然而，仅仅堆叠像 `Dense` 这样的层通常是不够的。如果只使用执行线性运算的层（如矩阵乘法后加上偏置 (bias)），那么堆叠起来的整体结果仍然是一个线性函数。线性函数的组合依然是线性函数。为了模拟复杂的模式，神经网络需要引入非线性。这就是激活函数 (activation function)发挥作用的地方。

激活函数被逐元素地应用于层（通常被称为预激活值或 logits）的输出，在传递给下一层之前对其进行转换。这种非线性转换使得网络能够学习输入和输出之间更为复杂的映射。

### 非线性为何重要

考虑一个只包含线性层的简单网络。每个层计算 $output = W \cdot input + b$，其中 $W$ 是权重 (weight)矩阵，$b$ 是偏置 (bias)向量 (vector)。如果你堆叠两个这样的层，输出会变成：

$output_2 = W_2 \cdot (W_1 \cdot input + b_1) + b_2 = (W_2 W_1) \cdot input + (W_2 b_1 + b_2)$

这仍然是 $W' \cdot input + b'$ 的形式，它是一个线性转换。无论你堆叠多少个线性层，网络都只能表示线性关系。激活函数 (activation function)打破了这种线性，使网络能够近似任意复杂的函数。

### 常见激活函数 (activation function)

Keras 提供了多种内置激活函数。下面介绍一些最常用的函数。

#### 整流线性单元 (ReLU)

整流线性单元，或称 ReLU，是深度学习 (deep learning)中最受欢迎的激活函数之一，尤其适用于隐藏层。它计算高效且通常表现良好。

它的定义简单：如果输入为正，则直接返回输入值；否则返回零。


$$
f(x) = \max(0, x)
$$


- **优点：**
  - 计算效率高 (仅是一个阈值操作)。
  - 对于正输入避免了梯度消失问题，很多情况下训练速度比 Sigmoid 或 Tanh 更快。
- **缺点：**
  - **ReLU 死亡问题：** 如果神经元的输出持续为零（例如，由于大的负偏差或大的梯度更新），它们可能变得不活跃。一旦神经元的输出变为零，流经它的梯度也将为零，从而阻止其权重 (weight)更新。
  - 输出不是零中心化的。

ReLU 常是前馈和卷积神经网络 (neural network) (CNN)中隐藏层的默认选择。

#### Sigmoid

Sigmoid 函数将其输入压缩到 (0, 1) 范围内。


$$
f(x) = \sigma(x) = \frac{1}{1 + e^{-x}}
$$


- **优点：**
  - 输出 0 到 1 之间的值，这对于二分类问题中表示概率很有用。
- **缺点：**
  - **梯度消失：** 对于非常大或非常小的输入，函数会饱和（输出接近 1 或 0），梯度变得非常小。这会显著减缓或阻碍深度网络的学习过程。
  - 输出不是零中心化的。这有时会减慢收敛速度。

Sigmoid 主要用于二分类模型的输出层，其中输出需要被解释为概率。由于 ReLU 及其变体的普及，它现在在隐藏层中较不常见。

#### Softmax

Softmax 函数是 Sigmoid 函数在多分类问题中的推广。它接收一个任意实值分数（logits）向量 (vector)作为输入，并将其转换为一个 0 到 1 之间的值向量，这些值的总和为 1。这些输出可以被解释为每个类别的概率。

对于输入向量 $x = [x_1, x_2, ..., x_N]$，第 $i$ 个元素的 softmax 输出是：


$$
f(x)_i = \frac{e^{x_i}}{\sum_{j=1}^{N} e^{x_j}}
$$


- **优点：**
  - 输出多个类别的概率分布。
- **缺点：**
  - 通常只适用于多分类任务的输出层。

Softmax 是多分类网络中最后一层的标准激活函数。

#### 双曲正切 (Tanh)

双曲正切函数（或称 Tanh）将其输入压缩到 (-1, 1) 范围内。


$$
f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$


- **优点：**
  - 输出是零中心化的，这有时可以帮助收敛，相比 Sigmoid。
- **缺点：**
  - 仍然存在梯度消失问题，与 Sigmoid 相似，尽管其梯度通常更大。

Tanh 曾是隐藏层中流行的选择，但现在已大体被 ReLU 及其变体替代。它有时仍在使用，特别是在循环神经网络（RNN）中，尽管现代 RNN 架构常使用其他门控机制。



![常见激活函数](plots/1715-0.json)



> ReLU、Sigmoid 和 Tanh 激活函数的图示。注意它们不同的输出范围和形状，尤其是在 $x=0$ 附近。

### 激活函数 (activation function)的选择

激活函数的选择取决于层的位置和具体的任务：

- **隐藏层：** ReLU 通常是推荐的起点，因为它效率高且表现好。Leaky ReLU（在输入为负时允许小梯度）或 ELU（指数线性单元）等变体有时可以帮助缓解“ReLU 死亡”问题。Tanh 偶尔使用，但现在较不常见。Sigmoid 很少用于隐藏层。
- **输出层：**
  - **二分类：** Sigmoid 通常用于输出 0 到 1 之间的概率。
  - **多分类：** Softmax 用于输出跨多个类别的概率分布。
  - **回归：** 通常不应用激活函数（或线性激活， $f(x)=x$），因为输出需要能够取任何实数值。

### 在 Keras 中实现激活函数 (activation function)

你可以使用 `activation` 参数 (parameter)为大多数 Keras 层指定激活函数：

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 在Dense层内使用激活参数
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)), # 隐藏层使用ReLU
    layers.Dense(10, activation='softmax') # 输出层使用Softmax（多分类）
])

# 使用Activation层显式表示的等效方法
model_explicit = keras.Sequential([
    layers.Dense(64, input_shape=(784,)),
    layers.Activation('relu'), # 单独应用ReLU
    layers.Dense(10),
    layers.Activation('softmax') # 单独应用Softmax
])

# 你也可以直接传递函数对象
model_object = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=(784,)),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()
```

Keras 通过字符串名称（例如，`'relu'`、`'sigmoid'`、`'softmax'`、`'tanh'`、`'linear'`）识别激活函数。使用 `activation` 参数是最常见和简洁的方法。独立的 `layers.Activation` 层提供了灵活性，尤其是在使用函数式 API 或自定义架构时，你可能希望独立应用激活函数。

理解激活函数对于构建高效的神经网络 (neural network)十分重要。它们是引入必要非线性的核心组成部分，使模型能够学习超越简单线性关系的复杂模式。当你使用函数式 API 或自定义层构建更复杂的模型时，你会看到如何策略性地放置这些非线性转换来支持强大的计算。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本全面的书籍详述了神经网络理论，涵盖了非线性的必要性以及ReLU、sigmoid和tanh等各种激活函数及其特性。
- [Deep Sparse Rectifier Networks](https://doi.org/10.55982/aistats.2011.27) — Xavier Glorot, Antoine Bordes, and Yoshua Bengio (2011)
  Journal: Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: Microtome Publishing; Volume: 15; Pages: 315-323; DOI: [10.55982/aistats.2011.27](https://doi.org/10.55982/aistats.2011.27)
  这篇论文介绍了并评估了整流线性单元（ReLU）激活函数，展示了其在训练深度神经网络方面的优势。
- [Activations](https://keras.io/api/layers/activations/) — Keras team (2024)
  Keras官方文档，提供了Keras API中各种激活函数的详细信息和使用示例。
