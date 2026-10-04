---
course: "deep-learning-fundamentals-keras"
chapter: "convolutional-neural-networks-cnns"
lesson: "pooling-layers-maxpooling2d"
sourceId: 4939
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/pooling-layers-maxpooling2d"
title: "池化层 (MaxPooling2D)"
description: "学习池化层的作用及实现，以对特征图进行下采样。"
order: 3
plots: []
sourceHash: "72a39e5683065b5396a9964d16329cd0137a70e9c8bda4bc6fb2734f0f73665f"
sourceCorrections: []
---

应用卷积层后，生成的特征图能够捕获输入中检测到的模式的空间层级结构。然而，这些特征图可能仍然相当大，导致后续层计算开销很高。此外，我们通常希望网络对输入中特征的精确位置具有一定的鲁棒性；无论边缘是向左或向右偏移几个像素被检测到，都不应显著改变整体分类结果。在这种情况下，池化层就发挥作用了。

池化层对特征图的空间维度（宽度和高度）执行下采样操作。它们减少了信息量，仅保留最重要的部分，这有助于降低计算量、抑制过拟拟合，并引入一定程度的平移不变性。与卷积层不同，标准池化层没有可学习的参数 (parameter)；它们应用的是固定操作。

### 最大池化的工作原理

最常见的池化类型是**最大池化**。它的工作方式是定义一个窗口（或池）大小，通常是2x2，以及一个步长，通常与池大小相同。这个窗口在输入特征图上滑动。对于窗口的每个位置，选择该窗口内的最大值，并将其作为输出特征图中的对应元素。

考虑一个4x4的特征图以及一个池大小为2x2、步长为2的最大池化操作。

1. 2x2窗口从左上角开始。它覆盖了(0,0)、(0,1)、(1,0)和(1,1)位置的值。选取这四个值中的最大值。
2. 然后，窗口向右滑动2步（由于步长为2），覆盖(0,2)、(0,3)、(1,2)和(1,3)位置的值。再次选取最大值。
3. 扫描完第一行窗口后，窗口向下移动2步，再次从左侧开始，覆盖(2,0)、(2,1)、(3,0)和(3,1)位置的值。选取最大值。
4. 最后，它再次向右移动，覆盖(2,2)、(2,3)、(3,2)和(3,3)位置的值，并选取最大值。

结果是一个新的2x2特征图，其中每个值都代表了原始4x4图中一个2x2区域的最大激活值。

```text
输入特征图 (4x4)      窗口位置与最大操作      输出特征图 (2x2)
[[ 1  3 | 2  4 ]             max(1,3,5,7)=7  max(2,4,6,8)=8          [[ 7  8 ]
 [ 5  7 | 6  8 ]             --------------------------              [      ]
 [-------+-------]             max(9,1,3,4)=9  max(2,5,6,0)=6          [ 9  6 ]]
 [ 9  1 | 2  5 ]
 [ 3  4 | 6  0 ]]
```

### 最大池化的优点

- **降维：** 正如示例所示，最大池化显著减小了特征图的空间维度（宽度和高度）（例如，从4x4到2x2）。这降低了后续层的参数 (parameter)数量和计算量，使模型更快、内存效率更高。
- **平移不变性（局部）：** 通过选取区域内的最大值，输出对该区域内特征的精确位置的敏感度较低。如果最强的激活略微移动，最大值可能保持不变，这为小范围的平移提供了鲁棒性。
- **特征概括：** 它有助于保留先前的卷积层检测到的最显著特征（最高激活），同时舍弃不太相关的细节。

### Keras中的最大池化：`MaxPooling2D`

Keras提供了`keras.layers.MaxPooling2D`层用于此操作。它通常在卷积层之后添加（常与激活函数 (activation function)配对使用）。

```python
import keras
from keras import layers

# Sequential模型中的使用示例
model = keras.Sequential([
    # 假设输入形状为 (高度, 宽度, 通道数)，例如 (64, 64, 3)
    layers.Input(shape=(64, 64, 3)),

    # 第一个卷积块
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    # 应用最大池化
    layers.MaxPooling2D(pool_size=(2, 2)), # 将空间维度缩小2倍

    # 第二个卷积块
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)), # 再次缩小

    # ... 可能有更多层 ...

    layers.Flatten(),
    layers.Dense(10, activation='softmax') # 示例输出层
])

model.summary()
```

`MaxPooling2D`的主要参数 (parameter)包括：

- `pool_size`: 一个元组，指定池化窗口的高度和宽度（例如，`(2, 2)`）。
- `strides`: 一个元组，指定窗口在高度和宽度方向上的移动步长。如果为`None`（默认值），它将默认为`pool_size`，这对于非重叠池化是常见做法。
- `padding`: 类似于`Conv2D`，通常是`'valid'`（无填充，如果窗口/步长不完全匹配，维度可能会缩小）或`'same'`（用零填充，以便输出维度主要由步长决定，通常是floor(input\_dim / stride)）。`'valid'`是默认值。

让我们看看`MaxPooling2D`在默认步长（`strides=pool_size`）和`padding='valid'`情况下的形状变化：

```python
# 演示形状变化的示例
input_shape = (1, 28, 28, 16) # 批次=1, 高度=28, 宽度=28, 通道数=16
input_tensor = keras.random.uniform(input_shape)

# 应用2x2池化的MaxPooling2D
pooling_layer = layers.MaxPooling2D(pool_size=(2, 2))
output_tensor = pooling_layer(input_tensor)

print(f"输入形状: {input_tensor.shape}")
print(f"MaxPooling2D(2,2)后的输出形状: {output_tensor.shape}")

# 应用3x3池化的MaxPooling2D
pooling_layer_3x3 = layers.MaxPooling2D(pool_size=(3, 3)) # 步长默认为 (3, 3)
output_tensor_3x3 = pooling_layer_3x3(input_tensor)

print(f"MaxPooling2D(3,3)后的输出形状: {output_tensor_3x3.shape}")

# --- 预期输出 ---
# Input shape: (1, 28, 28, 16)
# Output shape after MaxPooling2D(2,2): (1, 14, 14, 16)
# Output shape after MaxPooling2D(3,3): (1, 9, 9, 16) # 28/3 = 9.33 -> 向下取整为 9
```

请注意，空间维度（高度和宽度）根据`pool_size`和步长（此处隐式为`pool_size`）进行缩小，而通道数（特征图）保持不变。

### 其他池化策略

虽然最大池化非常常见，但也存在其他替代方案：

- **平均池化（`AveragePooling2D`）：** 计算池化窗口内的平均值而不是最大值。它提供更平滑的下采样，但可能会稀释非常强烈的局部特征。
- **全局池化（`GlobalMaxPooling2D`，`GlobalAveragePooling2D`）：** 这些层对特征图的*整个*空间维度执行池化，将每个特征图缩减为单个值（最大值或平均值）。它们通常用作最终`Dense`分类层之前`Flatten`的替代方案，能大幅减少参数 (parameter)数量。

在实际应用中，最大池化因其在概括最活跃特征和提供鲁棒性方面的有效性，常被选为CNN图像分类任务的默认池化方式。通常会看到`Conv2D`层后接`MaxPooling2D`层重复多次，以逐步降低空间分辨率，同时增加特征通道的数量。

## 参考资料

- [Gradient-based learning applied to document recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍了LeNet-5架构，其中显著使用了池化层进行下采样并实现局部平移不变性。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 9: Convolutional Networks
  全面阐述了池化层的理论，包括其在卷积神经网络中用于特征图缩减和实现局部不变性的作用。
- [Keras MaxPooling2D layer](https://keras.io/api/layers/pooling_layers/max_pooling2d/) — Keras Team (2024)
  Keras中MaxPooling2D层的官方文档，详细说明了其参数、参数值和用法。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Pooling Layer](http://cs231n.github.io/convolutional-networks/#pool) — Stanford University (2023)
  通过清晰的解释和视觉示例说明了池化操作、其目的以及在卷积神经网络中的常见类型。
