# 卷积神经网络（CNN）与 Flux

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/convolutional-neural-networks-flux)

[返回章节目录](README.md) · [返回课程目录](../README.md)

卷积神经网络 (neural network)（CNN）是一种专门用于处理网格状结构数据的神经网络，非常有效。虽然它们可以应用于多种数据类型，例如时间序列（一维网格）或体数据（三维网格），但其最显著的成功是在计算机视觉方面，即分析图像中的二维像素网格。与层中每个输入单元都连接到每个输出单元的全连接网络不同，CNN通过卷积采用权重 (weight)共享架构，使其在处理图像等高维输入时更高效且更具扩展性。

### 卷积操作

卷积神经网络 (neural network) (CNN)的核心是卷积操作。想象一个小的窗口，称为滤波器或卷积核，在输入数据（例如图像）上滑动。这个滤波器通常是一个可学习权重 (weight)的微小矩阵。当它滑动时，它会对当前覆盖的输入部分执行逐元素乘法，然后将结果求和，以在输出特征图中生成一个值。这个过程在整个输入上重复进行。

> 一个 2x2 的滤波器作用于输入的高亮 2x2 区域。高亮输出值 '2' 的计算方式是 $(1 \times 0) + (0 \times 1) + (2 \times 1) + (1 \times 0)$。滤波器随后滑动到其他区域。

卷积层的重要参数 (parameter)包括：

- **滤波器大小（或卷积核大小）：** 定义滑动窗口的尺寸（例如，3x3，5x5）。较小的滤波器捕获局部特征，而较大的滤波器可以相对于其输入捕获更多整体模式。
- **滤波器数量：** 每个滤波器学习检测不同的特征。使用多个滤波器使得该层可以从输入中提取丰富的特征集。
- **步长：** 滤波器每步移动的像素数量。步长为 1 意味着滤波器每次移动一个像素。步长为 2 意味着它跳过一个像素，有效地对输出进行下采样。
- **填充：** 通常，在输入的边界周围添加零。这使得滤波器能够更全面地处理输入的边缘，并能控制输出特征图的空间尺寸。Flux 中的 `SamePad()` 旨在保持输出尺寸与输入相似（给定步长为 1 的情况）。

在 Flux.jl 中，您可以使用 `Conv((filter_height, filter_width), input_channels => output_channels, activation_function; stride=1, pad=0)` 定义一个二维卷积层。例如，一个包含 16 个 3x3 大小滤波器、接收三通道（RGB）图像作为输入并使用 ReLU 激活的层将是：

```julia
using Flux

# 16 个滤波器，每个 3x3，作用于 3 通道输入
# 输出将有 16 个通道
conv_layer = Conv((3, 3), 3 => 16, relu; stride=1, pad=SamePad())
```

`input_channels => output_channels` 对指定了输入的深度和滤波器数量（这决定了输出的深度）。`SamePad()` 是 Flux 中的一个工具，用于计算在步长为 1 时保持空间尺寸不变所需的填充。

### 特征提取与层次结构

卷积层中的每个滤波器都学习检测特定的模式。在处理图像的网络早期层中，滤波器可能会学习检测边缘、角点或颜色斑块等简单特征。当数据通过后续卷积层时，更深层中的滤波器可以结合这些简单特征来检测更复杂的模式，例如纹理、物体局部（如轮子、眼睛）甚至整个物体。这种分层特征学习是卷积神经网络 (neural network) (CNN)的一个强大特性。卷积层在应用其所有滤波器后的输出是一组称为特征图的二维数组，其中每个图对应一个特定的学习特征。

### 使用池化层进行下采样

在卷积和激活之后，通常使用池化层。池化层减少特征图的空间尺寸（宽度和高度），这有多个优点：

- **减少计算负担：** 后续层中参数 (parameter)和计算量减少。
- **控制过拟合 (overfitting)：** 通过在局部范围内概括特征，它提供了一种正则化 (regularization)形式。
- **引入局部平移不变性：** 特征的精确位置变得不那么重要。例如，如果一个微小特征略微移动，最大池化操作可能仍会输出相同的值。

两种最常见的池化类型是：

- **最大池化：** 对于输入特征图中的每个块，它输出最大值。这倾向于保留特征的最强激活。

  ```julia
  # 步长为 2 的 2x2 最大池化层
  max_pool_layer = MaxPool((2, 2); stride=2)
  ```
- **平均池化：** 对于每个块，它输出平均值。这提供了更平滑的下采样。

  ```julia
  # 步长为 2 的 2x2 平均池化层
  avg_pool_layer = MeanPool((2, 2); stride=2)
  ```

通常，池化窗口为 2x2，步长为 2，这会将特征图的高度和宽度减半。

### 使用 Flux.jl 构建 CNN 架构

您通过顺序堆叠这些层（卷积层、激活层、池化层）来构建 CNN。Flux.jl 的 `Chain` 使这变得简单直接。CNN 中一个块的常见模式是 `Conv -> Activation Function -> Pool`。

> 此图表显示了一个常见 CNN 结构。卷积层和池化层提取特征，这些特征随后被展平并传递给全连接层进行分类。

经过多个卷积层和池化层后，生成的高级特征图通常被展平为一维向量 (vector)。这个向量随后作为输入传递给一个或多个标准 `Dense`（全连接）层，这些层执行最终的分类或回归任务。`Flux.flatten` 函数用于此目的。

### Flux.jl 中 CNN 的输入数据形状

Flux.jl 的二维卷积层 (`Conv`) 要求输入数据为特定的四维格式：`(宽度, 高度, 通道数, 批大小)`。

- **宽度 (W)：** 图像的像素宽度。
- **高度 (H)：** 图像的像素高度。
- **通道数 (C)：** 颜色通道的数量。对于灰度图像，此值为 1。对于 RGB 彩色图像，此值为 3。
- **批大小 (N)：** 单批次处理的图像数量。

例如，一个包含 64 张 28x28 像素灰度图像的批次，其形状将是 `(28, 28, 1, 64)`。一个包含 32 张 128x128 像素 RGB 彩色图像的批次，其形状将是 `(128, 128, 3, 32)`。确保您的数据形状正确是在将其输入 CNN 之前的重要步骤。本章前面讨论过的 `MLUtils.jl` 包提供了数据批处理的工具，您经常会重塑数据数组以匹配这种 WHCN 格式。

### Flux.jl 中的一个简单 CNN 示例

让我们在 Flux.jl 中定义一个 CNN 模型，适用于分类 MNIST 数据集中的 28x28 灰度手写数字等任务。

```julia
using Flux

# 定义图像和网络参数
img_width, img_height = 28, 28
input_channels = 1 # 灰度图
num_classes = 10

# 计算卷积/池化层后展平特征的大小
# 第一个 MaxPool((2,2)) 将尺寸减半：28x28 -> 14x14
# 第二个 MaxPool((2,2)) 再次将尺寸减半：14x14 -> 7x7
# 第二个卷积层后的通道数为 16
final_conv_w = img_width ÷ 4
final_conv_h = img_height ÷ 4
channels_after_conv = 16
flattened_size = final_conv_w * final_conv_h * channels_after_conv # 7 * 7 * 16 = 784

# 定义 CNN 模型
model = Chain(
    # 第一个卷积块
    Conv((5, 5), input_channels => 6, relu; pad=SamePad()), # 输入: (28, 28, 1, N) -> 输出: (28, 28, 6, N)
    MaxPool((2, 2)),                                      # 输出: (14, 14, 6, N)

    # 第二个卷积块
    Conv((5, 5), 6 => channels_after_conv, relu; pad=SamePad()), # 输出: (14, 14, 16, N)
    MaxPool((2, 2)),                                           # 输出: (7, 7, 16, N)

    # 展平卷积/池化层的输出
    Flux.flatten,                                              # 输出: (784, N)

    # 全连接层
    Dense(flattened_size, 120, relu),
    Dense(120, 84, relu),
    Dense(84, num_classes)                                     # 用于 10 个类别的输出层
)

# 使用虚拟数据进行测试（例如，一张 28x28 灰度图像）
dummy_image_batch = rand(Float32, img_width, img_height, input_channels, 1)
output = model(dummy_image_batch)
println("Output shape: ", size(output)) # 应为 (10, 1)
```

在这个例子中，`pad=SamePad()` 确保卷积层不改变空间尺寸，使得在执行明确下采样的 `MaxPool` 层之前更容易追踪尺寸。`Flux.flatten` 将池化层输出的四维张量重塑为 `Dense` 层所需的二维矩阵（特征, 批大小）。最后的 `Dense` 层有 `num_classes` 个单元。用于多类别分类的激活函数 (activation function)（如 softmax）通常与损失函数 (loss function)（例如 `Flux.logitcrossentropy`）结合使用，以获得更好的数值稳定性。

这种受 LeNet 启发的结构是一种常见模式。掌握这些组成部分使您能够构建用于图像相关任务的 CNN，包括本章后面的动手练习。CNN 学习分层特征的能力以及它们处理网格状数据的高效性，使它们在深度学习 (deep learning)中不可或缺。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍LeNet-5的原创论文，这是用于数字识别的卷积神经网络的基础架构。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本全面的教材，提供了卷积网络和深度学习概念的详细理论背景。
- [Convolutional Layers (Conv, MaxPool, etc.)](https://fluxml.ai/Flux.jl/stable/model-reference/#Convolution-and-Pooling-Layers) — Flux.jl Developers (2021)
  Publisher: Flux.jl Project
  Flux.jl深度学习框架中卷积层和池化层的官方实现文档。
- [Convolutional Neural Networks for Visual Recognition - Course Notes](http://cs231n.github.io/convolutional-networks/) — Andrej Karpathy, Justin Johnson, Li Fei-Fei (2023)
  广泛使用的课程笔记，提供了CNN架构、操作和设计原理的易懂且详细的解释。

---

[上一节](03-%E6%9E%84%E5%BB%BA%E5%A4%9A%E5%B1%82%E6%84%9F%E7%9F%A5%E5%99%A8%EF%BC%88MLP%EF%BC%89.md) · [下一节](05-Flux%20%E4%B8%AD%E7%9A%84%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%E4%B8%8E%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9C%EF%BC%88LSTM%EF%BC%89.md)
