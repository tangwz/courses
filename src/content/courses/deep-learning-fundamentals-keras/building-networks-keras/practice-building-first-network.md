---
course: "deep-learning-fundamentals-keras"
chapter: "building-networks-keras"
lesson: "practice-building-first-network"
sourceId: 4903
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/practice-building-first-network"
title: "实践：使用 Keras 构建你的第一个网络"
description: "动手实践，使用 Sequential 和 Functional 两种 API 构建一个简单的神经网络。"
order: 7
plots: []
sourceHash: "9201c63e6a5b79c5b32d558173c16102cd7cac2a77aac776b7e82329816434ca"
sourceCorrections: []
---

使用 Keras 构建你的第一个神经网络 (neural network)。此处详细说明了如何使用 Sequential 和 Functional API 建立适合基本分类任务的简单模型。这些模型使用了层和激活函数 (activation function)等基本构成部分。

我们将从创建一些合成数据开始。这样我们可以纯粹关注网络构建，暂时无需考虑复杂的数据加载流程。我们将使用 Scikit-learn 的 `make_moons` 函数生成一个二维数据集，其中两个类别以非线性方式交织在一起，这为神经网络提供了一个不错的简单测试。

```python
import numpy as np
from sklearn.datasets import make_moons
import keras
from keras import layers
from keras import models
from keras import utils

# 生成合成数据
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

# 打印形状以验证
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# 可选：可视化数据（需要 matplotlib）
# import matplotlib.pyplot as plt
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
# plt.title("合成月亮数据集")
# plt.xlabel("特征 1")
# plt.ylabel("特征 2")
# plt.show()
```

这段代码生成 100 个数据点，每个点有两个特征（`X.shape` 将是 `(100, 2)`），以及对应的二元标签（`y.shape` 将是 `(100,)`）。

### 使用 Sequential API 构建网络

Sequential API 非常适合层线性堆叠的模型，即层逐个排列。这是开始使用的最简便方式。让我们构建一个含有一个隐藏层的小网络：

1. 一个输入层，通过第一个层中的 `input_shape` 隐式定义（或显式使用 `keras.Input`）。我们的数据有 2 个特征。
2. 一个 `Dense` 隐藏层，包含 8 个神经元和 ReLU 激活函数 (activation function)。
3. 一个输出 `Dense` 层，包含 1 个神经元和 `sigmoid` 激活函数，适合二元分类。

```python
# 使用 Sequential API 定义模型
model_sequential = models.Sequential(
    [
        keras.Input(shape=(2,), name="input_layer"), # 定义输入形状
        layers.Dense(8, activation='relu', name="hidden_layer"),
        layers.Dense(1, activation='sigmoid', name="output_layer")
    ],
    name="sequential_model"
)

# 显示模型的结构
model_sequential.summary()
```

运行 `model_sequential.summary()` 会提供层的简洁概览，包括它们的输出形状和可训练参数 (parameter)的数量：

```text
Model: "sequential_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ hidden_layer (Dense)            │ (None, 8)              │            24 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_layer (Dense)            │ (None, 1)              │             9 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 总参数: 33 (132.00 B)
 可训练参数: 33 (132.00 B)
 不可训练参数: 0 (0.00 B)
```

输出形状中的 `None` 表示批次大小，它可以是变化的。请留意参数是如何计算的：

- 隐藏层：(2 个输入特征 \* 8 个神经元) + 8 个偏置 (bias) = 16 + 8 = 24 个参数。
- 输出层：(隐藏层的 8 个输入 \* 1 个神经元) + 1 个偏置 = 8 + 1 = 9 个参数。

### 使用 Functional API 构建相同的网络

Functional API 为构建具有多个输入/输出或共享层的复杂模型提供了更大的灵活性。尽管我们当前的网络很简单，但让我们使用此 API 重新构建它，以便理解其工作流程。

核心思路是将层定义为在张量上操作的函数。

1. 使用 `keras.Input` 定义输入张量，指定输入数据的形状。
2. 在张量上调用层，按顺序连接它们。
3. 通过指定输入和输出张量来创建 `Model` 实例。

```python
# 定义输入张量
inputs = keras.Input(shape=(2,), name="input_tensor")

# 定义隐藏层，连接到输入
hidden = layers.Dense(8, activation='relu', name="hidden_layer")(inputs)

# 定义输出层，连接到隐藏层
outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(hidden)

# 创建模型
model_functional = models.Model(inputs=inputs, outputs=outputs, name="functional_model")

# 显示模型的结构
model_functional.summary()
```

摘要输出将与 Sequential 模型的结果相同，确认我们已构建了相同的结构：

```text
Model: "functional_model"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_tensor (InputLayer)       │ [(None, 2)]            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ hidden_layer (Dense)            │ (None, 8)              │            24 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_layer (Dense)            │ (None, 1)              │             9 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 总参数: 33 (132.00 B)
 可训练参数: 33 (132.00 B)
 不可训练参数: 0 (0.00 B)
```

Functional API 的一个显著优点是能够轻松可视化模型图。你可以使用 Keras 的 `plot_model` 工具（这需要安装 `pydot` 和 `graphviz` 库）。

```python
# 生成函数式模型的视觉表示
# 注意：需要安装 pydot 和 graphviz
# utils.plot_model(model_functional, show_shapes=True, show_layer_activations=True, to_file="functional_model.png")
```

这会生成一个如下图所示的图表，清晰地显示数据流经各层的情况：

> 一张表示函数式模型结构的图，显示输入层连接到隐藏的全连接层（ReLU 激活），该层又连接到输出的全连接层（Sigmoid 激活）。每层输出的形状都有标注。

### API 选择

- **Sequential API**：适用于简单、线性堆叠的层。对于直观的结构，它简洁易读。
- **Functional API**：更适合具有非线性拓扑的模型，例如多输入或多输出、共享层或残差连接的模型。它提供更大的灵活性。

你现在已经成功使用 Keras 构建了你的第一个神经网络 (neural network)结构！你定义了层、指定了激活函数 (activation function)，并使用 Sequential 和 Functional 两种 API 连接了它们。下一步，将在第 3 章中讲到，是通过定义损失函数 (loss function)和优化器来编译这些模型，然后使用 `fit` 方法在数据上训练它们。

## 参考资料

- [Guide to the Keras Sequential model](https://keras.io/guides/sequential_model/) — Keras team (2024)
  Publisher: Keras Documentation (part of TensorFlow/Google)
  Keras Sequential API 的官方指南，展示其用于线性层堆叠的模型构建。
- [Guide to the Keras functional API](https://keras.io/guides/functional_api_guide/) — fchollet (2023)
  Keras Functional API 的官方指南，涵盖包括多输入/输出在内的灵活网络架构构建。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  Keras 创建者撰写的书籍，提供使用 Keras 构建神经网络的实用指导，并解释层和激活函数等核心概念。
- [sklearn.datasets.make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) — scikit-learn developers (2024)
  用于生成示例中使用的合成二维分类数据集的官方文档。
