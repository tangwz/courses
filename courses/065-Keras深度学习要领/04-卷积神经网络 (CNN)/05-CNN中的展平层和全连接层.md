# CNN中的展平层和全连接层

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/flattening-dense-layers-cnns)

[返回章节目录](README.md) · [返回课程目录](../README.md)

卷积层和池化层是有效的特征提取器。这些层处理输入图像（或其他网格状数据）并生成特征图，这些多维张量表示检测到的模式，例如边缘、纹理或更复杂的形状。例如，在经过几个`Conv2D`和`MaxPooling2D`层之后，可能会得到一个形状如 $(height, width, channels)$ 的张量，比如 $(7, 7, 64)$。这个张量保留了空间信息；64个通道中的值对应于降采样后的 $7 \times 7$ 网格中不同位置检测到的特定特征。

然而，对于分类等任务，我们通常需要基于提取的*全部*特征集进行最终预测。标准的（`Dense`）全连接层期望其输入为一个一维向量 (vector)，其中每个元素代表一个单一的特征值，不包含任何二维或三维的空间结构。我们的卷积基的输出是一个多维张量，例如 $(7, 7, 64)$，它与`Dense`层不直接兼容。

### 展平层：连接两端

这就是`Flatten`层的作用。它的任务简单而必要：它接收来自卷积基的多维输出，并将其重塑，或称“展平”，为一个单一的、长的一维向量 (vector)。它通过逐行、逐通道地展开元素来完成此操作。

例如，如果输入张量的形状是 $(height, width, channels)$，`Flatten`层将生成一个长度为 $height \times width \times channels$ 的向量。使用我们之前 $(7, 7, 64)$ 张量的例子，`Flatten`层会将其转换为一个包含 $7 \times 7 \times 64 = 3136$ 个元素的向量。

> 示意图显示了展平层将卷积基的多维输出重塑为一维向量的过程。

`Flatten`层本身不学习任何东西；它不包含可训练的权重 (weight)。它纯粹是一种结构转换，用于连接CNN的特征提取部分（卷积基）到分类或回归部分（头部）。

### 全连接层：从展平特征中学习

一旦特征图被展平为向量 (vector)，我们就可以将这个向量输入到一个或多个标准`Dense`（全连接）层。这些层的工作方式与您在基础神经网络 (neural network)中遇到的层相同。`Dense`层中的每个神经元都接收来自前一层中*所有*神经元（在本例中是展平向量的所有元素）的输入。

这些`Dense`层在CNN架构中的目的是学习卷积基提取的特征组合。卷积层学习*局部*模式（小图像块中的边缘、纹理），而`Dense`层则基于哪些特征在何处被激活，学习整个输入图像的*全局*模式。它们结合这些高层次特征以进行最终预测。

通常，CNN在`Flatten`层之后包含一个或多个`Dense`层：

1. **中间全连接层：** 通常使用ReLU激活（`relu`）来引入非线性，并学习复杂的特征组合。这些层中的单元数量是需要调整的超参数 (parameter) (hyperparameter)（例如，128、256、512）。
2. **输出全连接层：** 最后一层生成网络的预测。
   - 对于二元分类，它有1个单元，使用`sigmoid`激活。
   - 对于多类分类，它有`N`个单元（其中`N`是类别的数量），使用`softmax`激活。
   - 对于回归任务，它有1个或更多单元（取决于目标值的数量），通常使用线性激活（或未指定激活）。

### Keras中的实现

在Keras模型中添加`Flatten`和`Dense`层是简单直接的。您只需在最后一个卷积或池化层之后添加它们。以下是它在Sequential模型中的样子：

```python
import keras
from keras import layers

# 假设'model'是一个已经包含Conv2D/MaxPooling2D层的Sequential模型
# 模型输入形状示例：MNIST为(28, 28, 1)

model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        # --- 卷积基 ---
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # --- 分类器头部 ---
        layers.Flatten(), # 将3D特征图展平为1D
        layers.Dropout(0.5), # 可选：用于正则化的Dropout
        layers.Dense(10, activation="softmax"), # 10个类别的输出层（例如，MNIST数字）
    ]
)

model.summary()
```

在此示例中：

1. 卷积基提取特征，产生多维特征图。
2. `layers.Flatten()`接收最后一个`MaxPooling2D`层的输出，并将其转换为一维向量 (vector)。
3. 可选择地添加`layers.Dropout(0.5)`层用于正则化 (regularization)（我们将在第6章介绍这一点）。
4. 最终的`layers.Dense(10, activation="softmax")`层执行分类，输出10个类别中每个类别的概率。

用于特征提取的卷积基，以及用于分类的`Flatten`加`Dense`层的组合，构成了许多成功CNN的标准架构，这些CNN应用于图像识别和其他方面。

*关于替代方案的说明：* 尽管`Flatten`后接`Dense`很常见，但还存在其他方法，如`GlobalAveragePooling2D`或`GlobalMaxPooling2D`。这些层也能连接卷积图与最终输出，通常减少参数 (parameter)数量并可能提高泛化能力。它们的工作原理是对每个特征图的空间维度（高度、宽度）取平均值或最大值，从而为每个通道生成一个值，直接创建一个一维向量。我们在此主要介绍`Flatten`，因为它是一个基本原理，但请留意这些适用于更复杂架构的替代方案。

## 参考资料

- [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  描述了LeNet-5这一基础CNN架构，阐释了卷积特征提取与全连接层相结合进行分类的模式。这篇论文确立了在全连接层之前使用展平操作的范例。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习的全面理论和实践论述，其中专门章节解释了卷积网络和全连接层在分类头中的作用。
- [Keras API Reference: Flatten layer](https://keras.io/api/layers/reshaping_layers/flatten/) — Keras team (2024)
  Keras `Flatten` 层的官方文档，详细说明了其目的、用法和参数，用于将多维输出转换为一维向量。
- [Keras API Reference: Dense layer](https://keras.io/api/layers/core_layers/dense/) — Keras team (2024)
  Keras `Dense` 层的官方文档，解释了其功能、单元和激活函数等参数，以及它在构建神经网络头部中的作用。

---

[上一节](04-%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84CNN%E6%9E%B6%E6%9E%84.md) · [下一节](06-%E5%9C%A8Keras%E4%B8%AD%E5%A4%84%E7%90%86%E5%9B%BE%E5%83%8F%E6%95%B0%E6%8D%AE.md)
