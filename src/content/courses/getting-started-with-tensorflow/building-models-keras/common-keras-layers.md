---
course: "getting-started-with-tensorflow"
chapter: "building-models-keras"
lesson: "common-keras-layers"
sourceId: 1712
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-3-building-models-keras/common-keras-layers"
title: "Keras 常用层"
description: "介绍 Keras 基本层：Dense、Conv2D、MaxPooling2D、Flatten、Dropout、LSTM。"
order: 3
plots: []
sourceHash: "4cf52d534a26fce30d94bc4af6628d8d8673fb79bf981d062fb4c258992a39c4"
sourceCorrections: []
---

层是 Keras 模型的基本构成要素，就像墙壁中的砖块一样。每一层都会对通过它的数据执行特定的转换。通过以有意义的方式堆叠这些层，您可以构建您的神经网络 (neural network)架构。Keras 提供了一套完整的预置层，简化了复杂模型的实现。

### Dense 层

`Dense` 层，也称为全连接层，是最基本和常见的层类型之一。`Dense` 层中的每个神经元都接收来自前一层*所有*神经元的输入（因此称为“全连接”）。它计算其输入的加权和，加上偏置 (bias)，然后通常应用一个激活函数 (activation function)。

从数学上讲，`Dense` 层执行的操作可以表示为：


$$
output = activation(dot(input, kernel) + bias)
$$


其中：

- `input` 代表输入张量。
- `kernel` 是由该层创建和管理的权重 (weight)矩阵。
- `bias` 是由该层创建和管理的偏置向量 (vector)（除非 `use_bias` 设置为 `False`）。
- `activation` 是应用的逐元素激活函数。

`Dense` 层的常见用途包括：

- 分类模型中的最后一层，通常使用 `softmax` 激活来输出每个类别的概率。
- 回归模型中的最后一层，通常不使用激活函数（或使用线性激活）。
- 多层感知机 (MLP) 的构成部分。

您主要通过指定输出神经元数量（单元数）和激活函数来配置 `Dense` 层。

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 示例：一个包含 64 个输出单元和 ReLU 激活函数的 Dense 层
dense_layer = layers.Dense(units=64, activation='relu')

# 示例：用于 10 类分类问题的最终 Dense 层
output_layer = layers.Dense(units=10, activation='softmax')
```

### Conv2D 层

卷积层，特别是 `Conv2D`，是现代计算机视觉模型的根基。它们专门用于处理网格状数据，例如图像（像素的 2D 网格）。

`Conv2D` 层不是全连接的，而是使用可学习的滤波器（或核）在输入空间维度（高和宽）上滑动，应用卷积操作。每个滤波器在其感受野中检测特定的局部模式（例如边缘、角点、纹理）。该层输出特征图，其中每个图对应于一个滤波器在整个输入上的响应。

`Conv2D` 的重要参数 (parameter)：

- `filters`：该层应学习的输出滤波器（特征图）的数量。
- `kernel_size`：一个整数或元组，指定卷积窗口（滤波器）的高度和宽度。常见的大小是 (3, 3) 或 (5, 5)。
- `strides`：一个整数或元组，指定核在输入上滑动时的步长。默认值为 (1, 1)。
- `padding`：要么是 `'valid'`（不填充），要么是 `'same'`（输出具有与输入相同的空间维度，如果需要则应用填充）。
- `activation`：卷积后要使用的激活函数 (activation function)。

卷积层在捕获空间层次结构方面表现出色；靠近输入的层学习简单模式，而更深层的层将这些模式组合起来学习更复杂的结构。

```python
# 示例：一个 2D 卷积层
conv_layer = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')

# 输入形状示例 (batch_size, height, width, channels)
# 假设 MNIST 图像的输入形状为 (None, 28, 28, 1)
input_shape = (28, 28, 1)
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
    # ... 其他层
])
# print(model.summary()) # 将显示输出形状
```

### MaxPooling2D 层

池化层通常与卷积层结合使用，用于对特征图进行下采样。`MaxPooling2D` 是一个常见的选择。它减小输入的空间维度（高和宽），同时保留每个池化窗口中最重要的信息（最大值）。

最大池化的好处：

- 减少后续层的计算负担。
- 提供一定程度的平移不变性（输入中的微小位移不会显著改变池化输出）。
- 通过提供抽象表示来帮助防止过拟合 (overfitting)。

`MaxPooling2D` 的重要参数 (parameter)：

- `pool_size`：一个整数或元组，指定池化窗口的大小。常见的是 (2, 2)。
- `strides`：一个整数或元组，指定池化窗口的步长。如果为 `None`，则默认为 `pool_size`。
- `padding`：`'valid'` 或 `'same'`。

```python
# 示例：一个最大池化层
max_pool_layer = layers.MaxPooling2D(pool_size=(2, 2))

# 通常在 Conv2D 层之后使用
model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)), # 输入示例：128x128 RGB 图像
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2, 2)) # 输出特征图大小变为 64x64
    # ... 其他层
])
# print(model.summary()) # 将显示减小的维度
```

### Flatten 层

`Flatten` 层有一个简单但重要的作用：它将多维张量转换为一维张量（即向量 (vector)）。这通常在从处理空间数据的卷积/池化层过渡到处理向量的全连接层时需要。

它接受输入张量，并通过展平除批维度（通常是第一个维度）之外的所有维度来重塑它。例如，形状为 `(batch_size, height, width, channels)` 的输入将被展平为 `(batch_size, height * width * channels)`。

```python
# 示例：在全连接层之前展平卷积/池化层的输出
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(), # 将 3D 特征图展平为 1D
    layers.Dense(10, activation='softmax') # 连接到全连接层
])
# print(model.summary()) # 显示展平后的形状变化
```

### Dropout 层

`Dropout` 是一种正则化 (regularization)技术，用于防止神经网络 (neural network)中的过拟合 (overfitting)。在训练期间，`Dropout` 层会在每个更新步骤中随机将一部分输入单元设为 0。这迫使网络学习更鲁棒的特征，这些特征不过度依赖任何一个神经元。

参数 (parameter)是 `rate`，它指定要丢弃的输入单元的比例（例如，`rate=0.25` 表示丢弃 25% 的输入）。

重要的是，`Dropout` *只*在训练期间活跃。在评估或推断期间（`model.evaluate()` 或 `model.predict()`），该层简单地传递所有输入，但将它们按 `rate` 等比例缩小，以平衡比训练期间更多单元活跃的情况。

```python
# 示例：在全连接层之后应用 Dropout
dropout_layer = layers.Dropout(rate=0.5) # 训练期间丢弃 50% 的输入

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3), # 应用 30% 的 Dropout
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3), # 应用 30% 的 Dropout
    layers.Dense(10, activation='softmax')
])
```

### LSTM 层

`LSTM`（长短期记忆）层是一种循环神经网络 (neural network) (RNN) 层，专门用于处理序列数据，例如时间序列或自然语言。标准 RNN 由于梯度消失问题，难以捕获序列中的长程依赖关系。

LSTM 通过内部单元状态和门控机制（输入门、遗忘门、输出门）解决此问题。这些门控制信息的流动，允许 LSTM 在长时间内选择性地记住相关信息并遗忘不相关的细节。

尽管内部工作原理复杂，但在 Keras 中使用 `LSTM` 层是直接的。您主要指定 `units` 的数量（输出空间的维度，也是内部隐藏状态的维度）。

```python
# 示例：用于序列处理的 LSTM 层
lstm_layer = layers.LSTM(units=64)

# 常用于文本分类或时间序列预测等任务
# 输入形状可能为 (batch_size, timesteps, features)
model = keras.Sequential([
    layers.Input(shape=(None, 10)), # 可变序列长度，每个时间步 10 个特征
    layers.LSTM(32), # 32 个 LSTM 单元
    layers.Dense(1, activation='sigmoid') # 示例：二元分类输出
])
# print(model.summary())
```

这些只是 `tf.keras.layers` 中可用的一些主力层。Keras 还提供许多其他层，包括不同类型的循环层 (GRU, SimpleRNN)、卷积层 (Conv1D, Conv3D)、归一化 (normalization)层 (BatchNormalization) 和注意力层。了解这些常用层为使用 Keras 构建各种神经网络模型提供了坚实的基础。随着您能力的提升，您将学习如何创造性地组合它们以处理不同的机器学习 (machine learning)问题。

## 参考资料

- [Keras layers API](https://www.tensorflow.org/api_docs/python/tf/keras/layers) — TensorFlow Developers (2024)
  官方文档，提供所有 Keras 层的详细信息和示例。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教科书，涵盖了各种神经网络层（包括全连接层、Conv2D、池化层、Dropout 和 LSTM）的理论基础和数学原理。
- [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍用于图像识别的卷积神经网络（LeNet-5）的奠基性论文，展示了 Conv2D 和池化层的原始应用。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  提出长短期记忆（LSTM）网络的原始论文，解决了循环神经网络在序列处理中的梯度消失问题。
- [Dropout: A Simple Way to Prevent Overfitting Neural Networks](http://jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research (JMLR); Publisher: JMLR; Volume: 15; Pages: 1929−1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  引入 Dropout 作为一种正则化技术，通过在训练期间随机丢弃单元来减少神经网络中的过拟合。
