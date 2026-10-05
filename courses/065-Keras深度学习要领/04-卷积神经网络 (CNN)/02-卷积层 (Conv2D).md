# 卷积层 (Conv2D)

来源：[原文](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-4-convolutional-neural-networks-cnns/convolutional-layers-conv2d)

[返回章节目录](README.md) · [返回课程目录](../README.md)

卷积神经网络 (neural network) (CNN)的构成主体是**卷积层**，它承担着在网格状数据中检测局部模式的主要部分。与全连接层不同，全连接层独立处理输入特征并丢失空间信息，而卷积层则在数据的空间背景下进行明确处理。针对二维数据（如图像），`Conv2D` 层是 Keras 中的主要实现方式。

想象一下在图像上滑动一个小型放大镜。这个放大镜不会一次性查看整个图像，而是集中于小区域，寻找边缘、角点或纹理等特定特征。卷积层以类似的方式，使用数字**滤波器**（也称为**卷积核**）进行运作。

### 滤波器（卷积核）：模式检测器

`Conv2D` 层中的滤波器本质上是一个由可学习权重 (weight)构成的小型矩阵。在训练期间，这些滤波器会演变以识别特定模式。例如，一个滤波器可能学会检测垂直边缘，另一个检测水平边缘，还有一个检测特定的颜色或纹理块。

- **尺寸**：滤波器在空间上通常很小（例如，$3 \times 3$ 或 $5 \times 5$ 像素），但会贯穿输入体积的完整深度。如果输入是彩色图像（高度、宽度、3 个通道），一个 $3 \times 3$ 的滤波器实际尺寸将是 $3 \times 3 \times 3$。
- **学习**：这些滤波器中的值（权重）是网络在反向传播 (backpropagation)过程中学习的内容。它们最初是随机的，然后进行调整，以对训练数据中存在的、对任务（例如分类）有益的模式变得敏感。

### 卷积操作：滑动与计算

核心操作是将每个滤波器在输入体积的空间尺寸（高度和宽度）上进行滑动。在每个位置，滤波器会对其当前覆盖的输入补丁执行逐元素乘法，然后将结果求和。通常会将一个偏置 (bias)项加入到这个和中。这个过程本质上是滤波器权重 (weight)与输入补丁之间的点积，再加上一个偏置。


$$
\text{输出值} = \text{激活函数} \left( \sum (\text{输入补丁} \times \text{滤波器}) + \text{偏置} \right)
$$


当滤波器在整个输入上滑动时，这个计算过程会重复进行，产生一个二维数组，称为**特征图**或**激活图**。

- **特征图**：每个特征图都会突出显示输入图像中相应滤波器检测到其特定模式的区域。高激活值表示该特征强烈存在。由于 `Conv2D` 层通常使用多个滤波器，它会产生多个特征图，这些特征图堆叠在一起形成输出体积。此输出体积的深度等于所用滤波器的数量。

考虑一个 $3 \times 3$ 滤波器在 $5 \times 5$ 输入上滑动的简化视图（假设单通道、步长为1、无填充）：

> 滤波器（黄色）覆盖输入的补丁（蓝色）。计算得到输出特征图（绿色）中的一个值。然后滤波器滑动到下一个位置。

### 在 Keras 中配置 `Conv2D` 层

当您向 Keras 模型添加 `Conv2D` 层时，需要指定几个重要参数 (parameter)：

- **`filters`**：这个整数决定了层将学习多少个滤波器。每个滤波器产生一个特征图，因此这定义了输出体积的深度。选择更多的滤波器使层能够学习更多样的模式，但会增加参数数量和计算成本。

  ```python
  # 示例：一个学习 32 种不同滤波器的层
  layers.Conv2D(filters=32, ...)
  ```
- **`kernel_size`**：这指定了滤波器的高度和宽度。它通常以包含两个整数的元组形式给出，例如 `(3, 3)` 或 `(5, 5)`。较小的卷积核捕捉更精细的局部细节，而较大的卷积核捕捉更宽泛的模式。`(3, 3)` 是一个常用起始点。

  ```python
  # 示例：使用 3x3 滤波器
  layers.Conv2D(filters=32, kernel_size=(3, 3), ...)
  ```
- **`strides`**：这个元组 `(sh, sw)` 控制滤波器在每一步水平 (`sw`) 和垂直 (`sh`) 移动多少像素。默认值是 `(1, 1)`，表示滤波器每次移动一个像素。使用大于 1 的步长（例如 `(2, 2)`）会导致滤波器跳过像素，产生更小的输出特征图（下采样）。这可以减少计算量，但可能导致信息丢失。

  ```python
  # 示例：滤波器水平和垂直移动 1 个像素
  layers.Conv2D(..., strides=(1, 1), ...) # 默认

  # 示例：滤波器水平和垂直移动 2 个像素（下采样）
  layers.Conv2D(..., strides=(2, 2), ...)
  ```
- **`padding`**：这决定了如何处理输入的边界。
  - `'valid'`：不应用填充。滤波器只在其能完全重叠输入的地方滑动。这会导致输出特征图的空间尺寸随每层缩小，尤其是在步长为 1 的情况下。
  - `'same'`：填充（通常为零）会自动添加到输入周围，使得输出特征图与输入具有相同的高度和宽度（假设 `strides=(1, 1)`）。这对于构建更深的神经网络 (neural network)，同时避免过快丢失空间分辨率很有帮助。

  ```python
  # 示例：输出尺寸缩小
  layers.Conv2D(..., padding='valid', ...)

  # 示例：输出高度/宽度与输入匹配（步长为 1 时）
  layers.Conv2D(..., padding='same', ...)
  ```
- **`activation`**：指定在卷积和偏置 (bias)添加后，逐元素应用于输出特征图的激活函数 (activation function)。ReLU (`'relu'`) 是卷积层一个非常常见的选择，因为它高效且能够缓解梯度消失问题。

  ```python
  # 示例：使用 ReLU 激活
  layers.Conv2D(..., activation='relu', ...)
  ```
- **`input_shape`**：仅在 `Sequential` 模型中的*第一*层（或在函数式 API 中定义 `Input` 层时）需要。它指定了层预期输入的尺寸，不包括批量大小。对于典型的彩色图像数据集，这会是 `(高度, 宽度, 通道数)`，例如灰度 MNIST 为 `(28, 28, 1)`，彩色 CIFAR-10 为 `(32, 32, 3)`。

  ```python
  # 示例：用于 28x28 灰度图像的模型中的第一层
  model = keras.Sequential([
      layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    input_shape=(28, 28, 1)),
      # ... 其他层
  ])
  ```

### 示例：添加 `Conv2D` 层

以下是如何在 Keras 模型中定义用于处理 $64 \times 64$ RGB 图像的第一层 `Conv2D` 层：

```python
import keras
from keras import layers

# 定义输入形状（高度，宽度，通道数）
input_shape = (64, 64, 3)

# 开始构建一个 Sequential 模型
model = keras.Sequential(name="SimpleCNN")

# 添加第一个 Conv2D 层
model.add(layers.Conv2D(
    filters=32,          # 学习 32 种模式
    kernel_size=(3, 3),  # 使用 3x3 滤波器
    activation='relu',   # 应用 ReLU 激活
    padding='same',      # 保持输出尺寸与输入相同 (64x64)
    input_shape=input_shape # 为第一层指定输入尺寸
))

# 通常在此之后会添加更多层
# model.add(layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
# ...

# 打印模型摘要以检查输出形状
model.summary()
```

这第一层接受一个 $64 \times 64 \times 3$ 的输入，并产生一个 $64 \times 64 \times 32$ 的输出体积（因为使用了 `padding='same'` 且 `strides` 默认为 `(1, 1)`）。输出中 32 个通道中的每一个都对应于由所学习的滤波器之一生成的特征图。

### 为什么卷积对图像处理表现良好

卷积层在处理图像数据方面相比全连接层具有多项优势：

1. **参数 (parameter)共享**：同一个滤波器（权重 (weight)集合）应用于输入中不同的空间位置。与全连接层相比，这大幅减少了参数数量，使得模型更高效且不易过拟合 (overfitting)。在图像某一部分学习到的模式可以在其他地方被识别。
2. **平移不变性（或等变性）**：因为在所有位置都使用相同的滤波器，网络在一定程度上对特征的位置保持不变性（或更准确地说，等变性）。如果一个滤波器学会检测眼睛，它无论出现在图像的左上角还是右下角，都能检测到那只眼睛。
3. **捕获空间层次结构**：通过堆叠卷积层，网络学习分层特征。早期层可能检测简单边缘和纹理，而更靠后的层结合这些来识别更复杂的模式，例如形状、物体或物体的一部分。

理解 `Conv2D` 层及其参数对构建高效的 CNN 极为重要。在接下来的章节中，我们将了解池化层，它们常常伴随卷积层，以及如何将这些组件组合成一个完整的 CNN 架构。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 9
  提供卷积网络的全面理论基础，包括卷积操作、滤波器属性和架构考量。
- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  一篇基础论文，介绍了LeNet-5，展示了卷积神经网络在图像处理任务中的早期成功应用。
- [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python-second-edition) — François Chollet (2021)
  Publisher: Manning Publications
  提供使用Keras实现卷积层的实用指南，解释了如滤波器、核大小、步长和填充等参数（第二版）。
- [Convolutional Neural Networks for Visual Recognition: ConvNet building blocks](http://cs231n.github.io/convolutional-networks/) — Stanford University (2023)
  提供卷积层机制的详细视觉解释和直观理解，包括滤波器、特征图、步长和填充。

---

[上一节](01-%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E6%B1%A0%E5%8C%96%E5%B1%82%20%28MaxPooling2D%29.md)
