# 构建一个简单的CNN架构

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/building-simple-cnn-architecture)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`Conv2D`和`MaxPooling2D`层等组成部分可以组装成一个可以工作的、简单的卷积神经网络 (neural network) (CNN)架构。CNN的主要优势在于通过有策略地堆叠这些层，以形成一个特征检测器的层次结构。

对于CNN架构，尤其是在分类任务中，一种常见的模式包含两个主要部分：

1. **卷积层组：** 一系列`Conv2D`和`MaxPooling2D`层，负责从输入图像中提取特征。它处理空间信息并学习不同抽象级别的表示。
2. **分类器头部：** 通常由`Flatten`层和一个或多个`Dense`层组成，这些层位于卷积层组之上。它接收提取的特征并执行最终的分类任务。

下面我们来详细说明如何使用Keras Sequential API构建这种结构。

### 构建卷积层组

卷积层组通常以接收输入图像的`Conv2D`层开始。请记住在模型的第一层中指定`input_shape`参数 (parameter)。这个形状通常包含高度、宽度和颜色通道的数量（例如，对于32x32的RGB图像，形状为`(32, 32, 3)`）。

然后我们交替使用`Conv2D`和`MaxPooling2D`层。一种常见做法是在较靠后的卷积层中增加滤波器数量（`Conv2D`的第一个参数）。这使得网络能够学习更复杂的模式，因为空间维度会因池化操作而减小。

- **`Conv2D`层：** 这些层应用卷积滤波器来检测局部模式。使用ReLU（`activation='relu'`）等激活函数 (activation function)会引入非线性。
- **`MaxPooling2D`层：** 这些层对特征图进行下采样，降低维度，并使学习到的特征对物体位置的变化更稳定。典型的`pool_size=(2, 2)`会将特征图的高度和宽度减半。

下面是如何在Keras中开始构建卷积层组：

```python
import keras
from keras import layers

# 假设输入图像是32x32的RGB图像
input_shape = (32, 32, 3)

# 开始构建卷积层组
model = keras.Sequential(name="simple_cnn_base")
model.add(layers.Input(shape=input_shape)) # 使用Input层明确定义形状

# 第一个卷积块
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# 第二个卷积块
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# （如果需要，可添加更多块）

print("卷积层组后的输出形状:", model.output_shape)
# 卷积层组后的输出形状示例: (None, 5, 5, 64)
# 注意：具体形状取决于input_shape、padding、strides和块的数量。
```

卷积层组的输出是一组特征图（例如，形状为`(None, 5, 5, 64)`），表示从输入中提取的高级特征。`None`维度表示批量大小，它可以是变化的。

> 一个包含两个块的卷积层组的简化流程图。

### 使用Flatten层进行转换

卷积层组的输出是一个3D张量（高度、宽度、通道）。然而，标准的`Dense`层期望1D向量 (vector)输入。这时`Flatten`层就发挥作用了。它简单地将多维特征图重塑为一个单一的长向量，丢弃空间结构但保留了学习到的特征信息。

你将`Flatten`层直接添加到基础层组的最后一个池化层或卷积层之后：

```python
# 继续之前的模型定义...

model.add(layers.Flatten())

print("Flatten层后的输出形状:", model.output_shape)
# Flatten层后的输出形状示例: (None, 1600) (因为 5 * 5 * 64 = 1600)
```

### 添加分类器头部

既然特征已展平为一个1D向量 (vector)，我们就可以添加一个或多个`Dense`层来执行分类任务。

- 一种常见方法是添加至少一个中间`Dense`层，并使用ReLU等非线性激活函数 (activation function)。此层会学习卷积层组提取的特征组合。
- 最后的`Dense`层必须具有等于分类问题中类别数量的单元数。其激活函数取决于分类的性质：
  - `softmax`：用于多类别分类（每个输入仅属于一个类别）。
  - `sigmoid`：用于二分类或多标签分类（每个输入可以属于多个类别）。

现在，让我们为10类别分类问题（如MNIST或CIFAR-10）完成我们的简单CNN架构：

```python
import keras
from keras import layers

# --- 定义完整的模型 ---
num_classes = 10
input_shape = (32, 32, 3) # 类似于CIFAR-10数据的示例

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),

        # 卷积层组
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # 可以在这里添加更多Conv/Pool层

        # 转换为分类器
        layers.Flatten(),

        # 分类器头部
        layers.Dropout(0.5), # 添加Dropout用于正则化（稍后会介绍）
        layers.Dense(128, activation="relu"), # 中间Dense层
        layers.Dense(num_classes, activation="softmax"), # 输出层
    ],
    name="simple_cnn_classifier",
)

# 显示模型架构
model.summary()
```

运行`model.summary()`将产生类似于以下的输出（具体数字取决于所选的层）：

```
Model: "simple_cnn_classifier"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 30, 30, 32)        │        896 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 15, 15, 32)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 13, 13, 64)        │     18,496 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 6, 6, 64)          │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ flatten (Flatten)               │ (None, 2304)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 2304)              │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 128)               │    295,040 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (None, 10)                │      1,290 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 315,722 (1.20 MB)
 Trainable params: 315,722 (1.20 MB)
 Non-trainable params: 0 (0.00 B)
```

这份摘要清楚地显示了层的序列、每个阶段的输出形状以及可训练参数 (parameter)的数量。请注意，空间维度（高度和宽度）在卷积层组中如何减小，而通道（滤波器）的数量通常会增加。展平后，数据会通过标准的全连接层进行分类。

这种结构是许多成功应用于图像识别的CNN的构成要素。虽然简单，但它包含了使用卷积和池化进行分层特征提取，以及随后使用全连接层进行分类的核心思想。在接下来的章节中，我们将了解如何准备图像数据并训练这样一个网络。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了卷积神经网络的基本思想，包括卷积层和池化层，它们构成了本节讨论的架构的基础。
- [Guide to the Sequential model](https://keras.io/guides/sequential_model/) — fchollet (2024)
  Publisher: Keras Documentation
  Keras 顺序式 API 构建神经网络的官方文档，直接适用于模型构建示例。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本提供卷积神经网络理论和实践理解的标准教科书。
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/) — Stanford University, CS231n Teaching Staff (2023)
  Publisher: Stanford University
  课程材料，详细解释了卷积神经网络的架构、层及其在计算机视觉中的应用。

---

[上一节](03-%E6%B1%A0%E5%8C%96%E5%B1%82%20%28MaxPooling2D%29.md) · [下一节](05-CNN%E4%B8%AD%E7%9A%84%E5%B1%95%E5%B9%B3%E5%B1%82%E5%92%8C%E5%85%A8%E8%BF%9E%E6%8E%A5%E5%B1%82.md)
