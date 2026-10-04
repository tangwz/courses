---
course: "autoencoders-representation-learning"
chapter: "classic-autoencoder-architecture"
lesson: "simple-autoencoder-hands-on"
sourceId: 2821
sourceUrl: "https://apxml.com/zh/courses/autoencoders-representation-learning/chapter-2-classic-autoencoder-architecture/simple-autoencoder-hands-on"
title: "搭建简单自编码器：实践操作"
description: "分步指南：如何在示例数据上实现和训练基本自编码器。"
order: 7
plots: ["plots/2821-0.json"]
sourceHash: "077767d8379490937e2843ea2ebee9d79324da1411834aa44721e0593a28cfb7"
sourceCorrections: []
---

搭建一个简单的全连接自编码器，并使用 TensorFlow/Keras 在 MNIST 手写数字数据集上训练它。这个经典自编码器架构的实践实现展示了编码器、瓶颈层和解码器如何协同工作，以习得数据的压缩表示。

我们的目标是训练一个网络，该网络接收一个784维向量 (vector)（即展平的28x28 MNIST图像）作为输入 $x$，将其编码为一个低得多的维度的潜在表示 $z$，然后将 $z$ 解码回一个784维向量 $\hat{x}$，使其与原始输入 $x$ 尽可能接近。

### 设置和数据准备

首先，我们需要导入所需的库并加载 MNIST 数据集。我们会将像素值归一化 (normalization)到 [0, 1] 范围，这是图像数据处理的常规做法，有助于模型训练的稳定性。我们还会将 28x28 的图像展平为784大小的向量 (vector)。

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 加载 MNIST 数据集
(x_train, _), (x_test, _) = mnist.load_data() # 我们只需要图像，不需要标签

# 将像素值归一化到 [0, 1] 并展平图像
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(f"训练数据形状: {x_train.shape}")
print(f"测试数据形状: {x_test.shape}")
```

```text
训练数据形状: (60000, 784)
测试数据形状: (10000, 784)
```

### 定义自编码器模型

我们将使用 Keras 的函数式 API 或顺序式 API 来构建自编码器。对于这个简单例子，顺序式 API 足以应对。

1. **编码器**：一系列全连接层，逐步降低维度。我们将使用 ReLU 激活函数 (activation function)。
2. **瓶颈层**：编码器的最后一层，代表我们压缩后的潜在空间 $z$。它的大小决定了压缩程度。我们选择潜在维度为32。
3. **解码器**：一系列全连接层，与编码器结构相对，逐步将维度恢复到原始的784。最后一层使用 sigmoid 激活函数，因为我们的输入像素已归一化 (normalization)到 0 到 1 之间。Sigmoid 函数的输出值也在此范围，因此适合用于重建输入。

```python
# 定义输入形状和潜在维度
input_dim = 784
latent_dim = 32

# --- 编码器 ---
encoder = keras.Sequential(
    [
        keras.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(latent_dim, activation="relu", name="bottleneck"), # 瓶颈层
    ],
    name="encoder",
)

# --- 解码器 ---
decoder = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(input_dim, activation="sigmoid"), # 输出层
    ],
    name="decoder",
)

# --- 自编码器（编码器 + 解码器） ---
autoencoder = keras.Sequential(
    [
        encoder,
        decoder,
    ],
    name="autoencoder",
)

# 显示模型摘要
encoder.summary()
decoder.summary()
autoencoder.summary()
```

这是我们简单自编码器的结构图：

> 一个简单的全连接自编码器架构。编码器将784维输入映射到32维瓶颈层，解码器则重建784维输出。

### 编译和训练模型

在训练之前，我们需要编译 `autoencoder` 模型。我们需要指定优化器和损失函数 (loss function)。

- **优化器**：Adam 是一种常见且有效的选择。
- **损失函数**：由于输入像素值已归一化 (normalization)到 0 到 1 之间，并且最后一层使用 sigmoid 激活，二元交叉熵 (BCE) 是一个合适的选择。然而，均方误差 (MSE) 也常被使用并表现良好，它衡量的是输入像素 $x_i$ 和重建像素 $\hat{x}_i$ 之间的平均平方差。为了简单起见，我们在这里使用 MSE，正如章节介绍中所讨论的： $L_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \hat{x}_i)^2$

我们训练模型以最小化这个重建损失，将输入数据 `x_train` 同时作为输入和目标输出。

```python
# 编译自编码器
autoencoder.compile(optimizer='adam', loss='mse') # 使用均方误差损失

# 训练自编码器
epochs = 20
batch_size = 256

history = autoencoder.fit(x_train, x_train, # 输入和目标相同
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=True,
                          validation_data=(x_test, x_test)) # 在测试集上评估重建效果
```

在训练过程中，Keras 会为每个迭代周期输出训练集和验证集上的损失。我们会看到损失随着时间推移而降低，这表明模型正在学习更准确地重建输入图像。

我们可以通过绘制损失曲线来可视化训练进程：



![自编码器训练与验证损失 (MSE)](plots/2821-0.json)



> MNIST 数据集上简单自编码器经过20个迭代周期的训练和验证损失（MSE，对数尺度）。两种损失都稳步下降，表明学习成功。

### 评估重建质量

训练结束后，我们可以通过视觉比较原始测试图像与它们的重建结果来评估自编码器的表现。我们使用训练好的 `autoencoder` 模型来预测测试集 `x_test` 的重建图像。

```python
# 预测测试集的重建图像
reconstructed_imgs = autoencoder.predict(x_test)

# --- 可视化 ---
n = 10 # 要显示的数字数量
plt.figure(figsize=(20, 4))
for i in range(n):
    # 显示原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("原始图像", loc='left', fontsize=12, pad=10)

    # 显示重建图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_title("重建图像", loc='left', fontsize=12, pad=10)

plt.suptitle("原始与重建 MNIST 数字对比", fontsize=16)
plt.show()
```

您应该会发现重建的数字虽然可识别，但与原始图像相比略显模糊。这种细节的丢失是预料之中的，因为信息必须通过32维瓶颈层进行压缩。自编码器习得了保留重建所需的最显著特征，同时舍弃了一些更精细的细节或噪声。

### 总结

在本实践部分，我们成功地在 MNIST 数据集上实现了并训练了一个基本的全连接自编码器。我们涉及了：

- 数据准备（归一化 (normalization)、展平）。
- 使用 Keras 定义编码器、瓶颈层和解码器。
- 使用合适的损失函数 (loss function)（MSE）和优化器（Adam）编译模型。
- 训练模型以最小化重建误差。
- 可视化训练进程和重建质量。

这个例子体现了自编码器的核心功能：学习一个压缩表示（编码），并从该表示重建输入（解码）。尽管对于简单数据来说很有效，但这种基本架构存在局限，尤其是在过拟合 (overfitting)和习得的潜在空间结构方面。在下一章中，我们将研究旨在解决这些问题并习得更稳定表示的正则化 (regularization)自编码器。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了自编码器、神经网络架构和训练方法的综合学术论述。
- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  描述了MNIST数据集和用于手写数字识别的早期神经网络应用。
