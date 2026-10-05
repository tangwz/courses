# Sequential 模型 API

来源：[原文](https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/sequential-model-api)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Keras 提供了几种方式来定义模型架构。最简单的方式是 `Sequential` API，它专门为线性堆叠的层构建的模型设计。您可以将其想象成使用乐高®积木进行搭建，其中每个层都直接连接到前一个层，并且一个层的输出按顺序成为下一个层的输入。这种方式非常直观，足以应对许多常见的网络类型，特别是标准多层感知器（MLP）或许多卷积神经网络 (neural network)（CNN）等前馈网络。

如果您的模型遵循从输入到输出的直接路径，没有分支、合并或多个输入/输出，那么 `Sequential` 模型通常是定义它的最简洁方式。

### 创建 Sequential 模型

您可以通过将层实例列表传递给其构造函数来创建 `Sequential` 模型。TensorFlow 需要知道模型应预期输入数据的形状。您只需为模型的*第一个*层指定此项；后续层可以根据前一个层的输出形状自动推断其输入形状。

让我们构建一个简单例子：一个包含两个全连接（`Dense`）层的模型。假设我们正在构建一个分类器，用于处理包含784个特征向量 (vector)的输入数据（可能是展平的28x28像素图像）。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 定义模型
model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="input_layer"), # 明确定义输入形状
        layers.Dense(64, activation="relu", name="dense_layer_1"),
        layers.Dense(10, activation="softmax", name="output_layer") # 10个类别的输出层
    ],
    name="my_simple_mlp"
)

# 显示模型的架构
model.summary()
```

在此例子中：

1. 我们导入所需的模块。
2. 我们创建了一个 `keras.Sequential` 实例。
3. 我们传递了一个层列表：
   - `keras.Input(shape=(784,))`: 这在技术上不是一个层，但它定义了预期的输入形状。提供 `Input` 对象是为 Sequential 模型指定输入形状的推荐方式。它使模型的结构从一开始就明确。形状 `(784,)` 表示每个输入样本是一个包含784个元素的扁平向量。
   - `layers.Dense(64, activation="relu")`: 第一个隐藏层是一个具有64个单元的全连接层，并使用修正线性单元（ReLU）激活函数 (activation function)。Keras 从前一个 `Input` 对象推断出其输入形状（784个单元）。
   - `layers.Dense(10, activation="softmax")`: 输出层有10个单元（分类问题中每个潜在类别一个），并使用 `softmax` 激活函数为每个类别生成类似概率的输出。Keras 从前一个 `Dense` 层推断出其输入形状（64个单元）。
4. `model.summary()` 打印出模型的有用概况，显示了层、它们的输出形状以及可训练参数 (parameter)的数量。

你会注意到 `model.summary()` 的输出详细列出了每个层，包括其类型、输出形状和参数数量。这是验证架构的绝佳工具。

```text
模型: "my_simple_mlp"
_________________________________________________________________
 层 (类型)                输出形状              参数数量
=================================================================
 dense_layer_1 (Dense)       (None, 64)                50240

 output_layer (Dense)        (None, 10)                650

=================================================================
总参数量: 50,890
可训练参数量: 50,890
不可训练参数量: 0
_________________________________________________________________
```

`None` 在输出形状中代表批次大小，它通常是灵活的，在定义模型架构时不确定。

### 使用 `add()` 逐步构建

此外，您可以初始化一个空的 `Sequential` 模型，并使用 `.add()` 方法逐步添加层。如果您有逻辑来决定添加哪些层，这种方式有时会更易读。

这里是使用 `.add()` 构建的相同模型：

```python
# 另一种方式：初始化并添加层
model_added = keras.Sequential(name="my_simple_mlp_added")
model_added.add(keras.Input(shape=(784,), name="input_layer")) # 定义输入形状
model_added.add(layers.Dense(64, activation="relu", name="dense_layer_1"))
model_added.add(layers.Dense(10, activation="softmax", name="output_layer"))

# 显示模型的架构
model_added.summary()
```

这会产生与前一种方法相同的模型结构。

### 何时不使用 Sequential

`Sequential` API 的优点是其简单性，但这也伴随着局限性。它不适用于以下模型：

- 层具有多个输入或产生多个输出。
- 需要层共享（在模型中多次使用同一层实例）。
- 需要非线性连接图（例如，残差连接）。

对于这些更复杂的架构，TensorFlow 提供了函数式 API，我们将在后面章节中介绍。然而，对于许多标准任务，`Sequential` 模型提供了一种清晰高效的方式来定义您的网络。

在接下来的章节中，我们将更详细地了解可以在这类模型中堆叠的常见层类型和激活函数 (activation function)。

## 参考资料

- [A guide to the Keras Sequential model](https://www.tensorflow.org/guide/keras/sequential_model) — François Chollet (2024)
  提供使用 Keras Sequential API 定义模型的详细说明和示例，涵盖其创建、使用和局限性。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，涵盖深度学习架构的理论基础，包括前馈网络、多层感知机和卷积神经网络。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，提供使用 TensorFlow 和 Keras 构建与训练机器学习模型的实现示例，包含 Sequential API 的具体示例。

---

[上一节](01-Keras%EF%BC%9ATensorFlow%20%E7%9A%84%E9%AB%98%E7%BA%A7%20API.md) · [下一节](03-Keras%20%E5%B8%B8%E7%94%A8%E5%B1%82.md)
