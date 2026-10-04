---
course: "deep-learning-fundamentals-keras"
chapter: "building-networks-keras"
lesson: "specifying-input-shape"
sourceId: 4893
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-2-building-networks-keras/specifying-input-shape"
title: "指定输入形状"
description: "学习如何正确定义 Keras 模型的输入形状。"
order: 4
plots: []
sourceHash: "24958c66c3c667a9ccfb2d6da45074adda92af0d09fca7824edf23a0f7009c3f"
sourceCorrections: []
---

构建神经网络 (neural network)时，您遇到的第一个实际问题之一就是告知模型它将接收何种数据。Keras 需要在创建层（特别是接收原始输入的第一个层）的所需参数 (parameter)（权重 (weight)和偏差）*之前*，知晓输入数据的维度。后续层通常可以根据前一层输出的大小自行确定其输入大小，但起始点需要明确定义。

思考一下我们之前介绍过的 `Dense` 层。在内部，它对输入特征及其权重矩阵进行矩阵乘法。如果一个输入样本有 $N$ 个特征，且 `Dense` 层有 $M$ 个单元，则权重矩阵的形状需要是 $(N, M)$。除非 Keras 知道 $N$ 的值（即每个输入样本中的特征数量），否则它无法正确创建此权重矩阵。

这就是指定输入形状的作用所在。您实质上定义了将输入到模型的数据约定。

Keras 中定义输入形状有两种主要方法，对应于我们讨论过的两种 API：

### 在 Sequential API 中使用 `input_shape`

使用 `Sequential` 模型时，您使用 `input_shape` 参数 (parameter)为堆栈中的*第一层*提供输入数据的形状。此参数接受一个元组，表示单个输入样本的尺寸，**不包括批次维度**。

例如，如果您的输入数据由 784 个特征的扁平向量 (vector)组成（如扁平化的 MNIST 图像），您将这样定义第一个 `Dense` 层：

```python
import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Input(shape=(784,)), # 使用 Input 层是推荐方式
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax") # 用于 10 个类别的输出层
    ],
    name="my_simple_sequential_model"
)

# 或者，直接在第一个 Dense 层上使用 input_shape：
# model = keras.Sequential(
#     [
#         layers.Dense(64, activation="relu", input_shape=(784,)), # 注意 1D 形状的逗号
#         layers.Dense(10, activation="softmax")
#     ],
#     name="my_alternative_sequential_model"
# )

model.summary()
```

注意 `(784,)` 中的逗号。这表示一个大小为 784 的一维元组。如果您省略逗号，`(784)` 将被解释为整数 784，这是不正确的。虽然直接在第一层上使用 `input_shape` 可以，但将 `keras.Input(shape=...)` 明确定义为 `Sequential` 列表中的第一个元素，通常会更清晰，并与 Functional API 方法更一致。

### 在 Functional API 中使用 `keras.Input`

Functional API 提供了一种更明确的方式，使用 `keras.Input` 对象定义模型的入口点。此对象表示将保存输入数据的符号张量。您使用 `shape` 参数 (parameter)指定形状，同样不包括批次维度。

```python
import keras
from keras import layers

# 定义输入张量
inputs = keras.Input(shape=(784,), name="input_features")

# 使用函数式风格连接层
x = layers.Dense(64, activation="relu", name="hidden_layer_1")(inputs)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

# 创建模型
model = keras.Model(inputs=inputs, outputs=outputs, name="my_functional_model")

model.summary()
```

这里，`keras.Input(shape=(784,))` 创建了我们层图的起点。所有后续层都会使用前一层的输出作为输入进行调用。这种方法使得数据流变得明确。

### 理解形状元组

`shape` 元组描述了您数据中*一个样本*的尺寸。批次大小（一次处理多少个样本）通常被省略，因为模型应该能够处理任意大小的批次。

以下是一些常见示例：

- **向量 (vector)数据：** 如果每个数据点都是特征的扁平列表（例如，来自 CSV 文件），则形状为 `(特征数量,)`。示例： `(150,)` 表示 150 个特征。
- **图像数据：** 对于标准图像格式（常用于卷积神经网络 (neural network) (CNN)，后续会介绍），形状通常包括高、宽和颜色通道。使用常见的“通道在后”格式，这将是 `(高, 宽, 通道)`。示例： `(28, 28, 1)` 用于 28x28 灰度图像，或 `(64, 64, 3)` 用于 64x64 彩色图像 (RGB)。
- **序列数据：** 对于时间序列或文本等数据（常用于循环神经网络 (RNN)，后续会介绍），形状通常是 `(时间步长, 每个时间步长的特征)`。示例： `(100, 50)` 用于 100 个时间步长的序列，其中每个时间步长有 50 个特征。

下图说明了输入形状如何在 Functional API 模型中定义数据流入第一层（本例中为 Dense 层）的入口点。

> 一个流程，显示输入规范如何引导数据进入网络的第一个处理层。输入中定义的 `shape` 决定了进入全连接层的每个数据样本的预期维度。

一旦为第一层定义了输入形状（通过 `input_shape` 或 `keras.Input`），Keras 会通过跟踪数据流经网络时的输出形状，自动推断所有后续层的输入形状。您通常只需关注在模型的入口处明确指定形状。正确设置这个初始形状是构建一个正确配置的 Keras 模型的重要一步。

## 参考资料

- [Guide to the Keras functional API](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG1-wCzMtCwY9K7N67vLAnWZaw-Sg0HpGBB9Ia5KLUhnRIAu-KApqbywr7fOOJ34E_aYV9_PglozHVEnrL0LPbYvt3m9IKBke-9g7N0mjccPWxknk0Wdv8fbVSDnnTOJZYCz9FXfkeAveGuDszLVpNWVHRg4rwfGEv1RsbcqE8dQ0F6HOSTeOuIEP7mmCwlDF9gnq-KljjyFr0h6x6EFm8CSBEbXtE=) — Keras Team (2024)
  Publisher: Keras (official documentation)
  详细解释了如何使用 `keras.Input` 和函数式 API 构建模型。
- [Guide to the Keras Sequential model](https://keras.io/guides/sequential_model/) — fchollet (2020)
  Publisher: Keras
  解释了如何使用 Sequential 模型，包括为第一层指定 `input_shape` 参数。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  由 Keras 的创建者撰写，此书涵盖了 Keras 模型构建的基础，包括输入定义。(第2版)
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的指南，通过示例讲解 Keras 模型构建和输入形状指定。(第3版)
