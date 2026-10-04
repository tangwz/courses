---
course: "deep-learning-fundamentals-keras"
chapter: "building-networks-keras"
lesson: "activation-functions"
sourceId: 4890
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/activation-functions"
title: "激活函数"
description: "说明激活函数（ReLU、Sigmoid、Tanh、Softmax）的作用及在 Keras 中的使用方法。"
order: 3
plots: ["plots/4890-0.json"]
sourceHash: "4e65f04eb78f7ab60b78be1f2964a8a75cc0d9f6530c1666815b1e43375b0d33"
sourceCorrections: []
---

在构建使用 `Dense` 层等网络结构时，信息如何在这些层*之间*流动和转换是一个重要的考虑因素。如果只是简单地堆叠 `Dense` 层而不对其输出进行任何修改，那么整个网络，无论有多少层，都将表现为一个单一的、大型线性变换。这是因为线性函数的组合仍然是线性函数。为了对大多数数据（如图像、文本或结构化数据集）中存在的复杂非线性关系进行建模，在网络中引入非线性是必要的。这正是激活函数 (activation function)的作用所在。

激活函数接收前一层（通常是输入的加权和加上一个偏置 (bias)）的输出信号，并对其应用一个固定的数学运算，对于大多数常见的激活函数来说，这种运算是逐元素进行的。这个转换后的输出随后成为下一层的输入。

我们来看一些最常用的激活函数，以及如何在 Keras 中实现它们。

### 为什么要非线性？

设想一下，如果要分离两类非线性可分的数据点（例如，形成同心圆的点）。一个简单的线性模型（$y = wx + b$）只能画出直线（或在更高维度上的超平面）。无论您堆叠多少个线性层，仍然只能生成一个线性的决策边界。激活函数 (activation function)打破了这种线性，使得网络能够学习复杂的曲线和形状，通过在每个层扭曲数据表示空间，有效地使模式更易于区分。

### 常用激活函数 (activation function)

#### Sigmoid

Sigmoid 函数将其输入压缩到 0 到 1 之间。其数学形式为：


$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$


它曾一度广受欢迎，其输出范围（0 到 1）使其适合用作二分类模型的输出层，输出可以被解释为概率。然而，由于以下几个缺点，它在隐藏层中的使用已不再普遍：

1. **梯度消失：** 对于非常大或非常小的输入，Sigmoid 函数的梯度会变得极其接近零。在反向传播 (backpropagation)过程中，这些小梯度在层层相乘后可能会变得非常小，从而使得早期层的权重 (weight)无法得到有效更新。
2. **非零均值：** 它的输出总是正的。这有时会减缓梯度下降 (gradient descent)的收敛速度。

在 Keras 中，您可以直接在层内使用它：

```python
import keras
from keras import layers

# 在 Dense 层中的使用示例
output_layer = layers.Dense(1, activation='sigmoid')
```

#### Tanh (双曲正切)

双曲正切函数，即 tanh，与 Sigmoid 密切相关，但将其输入压缩到 -1 到 1 之间。


$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2 \sigma(2x) - 1
$$


由于其零均值特性（输出范围从 -1 到 1），与 Sigmoid 相比，它通常有助于模型收敛，因此过去在隐藏层中是优于 Sigmoid 的选择。然而，对于大正或大负输入，它仍然存在梯度消失问题。

在 Keras 中的使用：

```python
# 在 Dense 层中的使用示例
hidden_layer = layers.Dense(64, activation='tanh')
```

#### ReLU (修正线性单元)

修正线性单元，即 ReLU，已成为大多数深度学习 (deep learning)应用中隐藏层的默认激活函数。它简单却非常有效。


$$
ReLU(x) = \max(0, x)
$$


如果输入为正，它直接输出输入值；否则输出零。

优点：

1. **计算效率高：** 计算成本非常低（只需一个比较操作）。
2. **减轻梯度消失：** 对于正输入，梯度是常数 (1)，这使得在反向传播过程中梯度能更好地流动，相比 Sigmoid 或 Tanh。

缺点：

1. **ReLU 死亡问题：** 如果神经元的输入持续低于零，它在训练期间有时会“停滞”。在这种状态下，梯度为零，神经元的权重将不会更新。Leaky ReLU 或 ELU (指数线性单元) 等变体试图通过允许负输入有一个小的非零梯度来解决此问题。

在 Keras 中的使用：

```python
# 在 Dense 层中的使用示例（隐藏层最常用）
hidden_layer = layers.Dense(128, activation='relu')
```

#### Softmax

Softmax 函数通常用于多类别分类网络的*输出层*。与前面那些独立逐元素操作的函数不同，Softmax 对最后一层输出的整个向量 (vector)进行操作。它将原始分数（logits）向量转换为概率分布，每个元素都在 0 到 1 之间，并且所有元素的和为 1。

对于输入向量 $x$，第 $i$ 个元素的 Softmax 输出是：


$$
Softmax(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$


分母中的和是输入向量 $x$ 中所有元素的总和。这确保了输出代表每个类别的概率。

在 Keras 中的使用（通常用于多类别分类的最终层）：

```python
# 10 类别分类问题的示例
output_layer = layers.Dense(10, activation='softmax')
```

### 激活函数 (activation function)的可视化

以下图表描绘了 Sigmoid、Tanh 和 ReLU 函数的形状：



![常用激活函数](plots/4890-0.json)



> Sigmoid（蓝色）、Tanh（绿色）和 ReLU（红色）激活函数在不同输入值范围内的比较。请注意不同的输出范围和形状。Softmax 没有显示，因为它操作的是向量 (vector)而非单个标量输入。

### 激活函数 (activation function)的选择

这里有一些一般性的指导原则：

- **隐藏层：** 从 ReLU 开始。它计算效率高且通常效果良好。如果遇到神经元死亡问题，可以考虑 Leaky ReLU 或其他变体。Tanh 有时也会使用，尤其是在循环网络中，但 ReLU 在前馈网络和 CNN 中更常见。
- **输出层（二分类）：** 使用 Sigmoid（带 1 个输出单元）。
- **输出层（多类别分类）：** 使用 Softmax（带 N 个输出单元，N 是类别数量）。
- **输出层（回归）：** 不使用激活函数（这相当于线性激活：$f(x)=x$）。输出可以取任意实数值。

### 在 Keras 中实现激活函数 (activation function)

如示例所示，添加激活函数最常见的方式是通过层定义中的 `activation` 参数 (parameter)：

```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)), # 带 ReLU 的输入层
    layers.Dense(64, activation='relu'),                      # 带 ReLU 的隐藏层
    layers.Dense(10, activation='softmax')                     # 10 个类别的输出层
])
```

或者，尤其在使用函数式 API 或如果要在没有 `activation` 参数的层（例如有时批归一化 (normalization)层）之后应用激活函数时，可以使用 `Activation` 层：

```python
from keras.layers import Activation

# 函数式 API 示例代码片段
inputs = keras.Input(shape=(784,))
x = layers.Dense(128)(inputs)
x = Activation('relu')(x) # 使用 Activation 层应用 ReLU
x = layers.Dense(64)(x)
x = Activation('relu')(x)
outputs = layers.Dense(10, activation='softmax')(x) # 在这里使用参数

model = keras.Model(inputs=inputs, outputs=outputs)
```

了解激活函数对于构建有效神经网络 (neural network)来说是必要的。它们是非线性的来源，使网络能够学习从输入到输出的复杂映射。选择正确的激活函数，特别是对于输出层，很大程度上取决于您处理的具体任务（分类、回归）。在接下来的部分中，我们将继续组合这些构建模块，以构建和查看完整的 Keras 模型。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了神经网络的理论背景，涵盖激活函数、其数学性质和应用。
- [Rectified Linear Units Improve Restricted Boltzmann Machines](http://www.icml2010.org/papers/432.pdf) — Vinod Nair, Geoffrey E. Hinton (2010)
  Journal: Proceedings of the 27th International Conference on Machine Learning (ICML-10); Publisher: International Conference on Machine Learning (ICML); Pages: 807-814; DOI: [10.5555/3104322.3104425](https://doi.org/10.5555/3104322.3104425)
  介绍了并评估了整流线性单元（ReLU）作为激活函数，展示了其有效性。
- [Activations](https://keras.io/api/layers/activations/) — Keras team (2023)
  Keras激活层的官方文档，详细说明了其使用方法和可用函数。
