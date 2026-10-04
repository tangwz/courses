---
course: "getting-started-with-pytorch"
chapter: "introduction-common-architectures"
lesson: "cnn-input-output-shapes"
sourceId: 2209
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/cnn-input-output-shapes"
title: "理解CNN层的输入/输出形状"
description: "根据输入尺寸、卷积核尺寸、步长和填充计算卷积层和池化层的输出维度。"
order: 3
plots: []
sourceHash: "ffe899ed56c75d4cdcc57e40fe4ac030daa32d5f4caf1a4351d80b0e43980867"
sourceCorrections: []
---

当你开始构建卷积神经网络 (neural network)（CNN）时，最常见的实际问题之一是确保一个层的输出形状能正确匹配下一个层的预期输入形状。与只需要考虑一个维度的简单全连接层不同，卷积层和池化层对多维网格状数据（如图像）进行操作，涉及高度、宽度和通道维度。了解这些维度如何变化对于构建有效的CNN架构非常重要。

让我们看一个用于二维CNN层（如`nn.Conv2d`或`nn.MaxPool2d`）的典型输入张量。它通常有四个维度：$(N, C_{in}, H_{in}, W_{in})$

- $N$: 批大小（同时处理的样本数量）。
- $C_{in}$: 输入通道数（例如，RGB图像为3，灰度图像为1）。
- $H_{in}$: 输入特征图的高度。
- $W_{in}$: 输入特征图的宽度。

批维度 $N$ 通常保持不变。主要的变换发生在通道 ($C$)、高度 ($H$) 和宽度 ($W$) 上。

### 卷积层 (`nn.Conv2d`)

`torch.nn.Conv2d`层对由多个输入平面组成的输入信号应用二维卷积。影响输出形状最重要的参数 (parameter)是：

- `in_channels` ($C_{in}$): 必须与输入张量中的通道数匹配。
- `out_channels` ($C_{out}$): 决定卷积产生的通道数。这是该层学习的滤波器数量。
- `kernel_size`: 卷积核（滤波器）的大小。可以是一个整数用于方形卷积核（例如，3表示3x3），也可以是一个元组`(kH, kW)`用于指定高度和宽度。
- `stride`: 卷积核在输入特征图上滑动时的步长。默认为1。可以是一个整数或一个元组`(sH, sW)`。较大的步长会导致输出特征图尺寸更小。
- `padding`: 输入边缘添加的零填充量。默认为0。可以是一个整数或一个元组`(padH, padW)`。填充有助于控制输出的空间维度，并能保留边界信息。
- `dilation`: 卷积核元素之间的间距。默认为1。较大的空洞（扩张）允许卷积核覆盖输入更广的区域，而不会增加参数数量（空洞卷积）。

输出形状 $(N, C_{out}, H_{out}, W_{out})$ 如下确定：

1. **通道数 ($C_{out}$):** 这由`nn.Conv2d`层的`out_channels`参数直接设置。每个滤波器产生一个输出通道（特征图）。
2. **高度 ($H_{out}$) 和宽度 ($W_{out}$):** 这些取决于输入维度 ($H_{in}, W_{in}$) 和层的参数。计算输出高度的公式是：

   
   $$
   H_{out} = \lfloor \frac{H_{in} + 2 \times \text{填充}[0] - \text{空洞}[0] \times (\text{卷积核尺寸}[0] - 1) - 1}{\text{步长}[0]} + 1 \rfloor
   $$
   

   对于宽度 ($W_{out}$) 也是类似的：

   
   $$
   W_{out} = \lfloor \frac{W_{in} + 2 \times \text{填充}[1] - \text{空洞}[1] \times (\text{卷积核尺寸}[1] - 1) - 1}{\text{步长}[1]} + 1 \rfloor
   $$
   

   注意：如果`padding`、`dilation`、`kernel_size`或`stride`被指定为单个整数，它们将应用于高度和宽度两个维度（例如，`padding[0] = padding[1] = padding`）。符号 $\lfloor \cdot \rfloor$ 表示向下取整函数（向下舍入到最接近的整数）。

让我们看一个当`dilation = 1`的常见情况。公式简化为：


$$
H_{out} = \lfloor \frac{H_{in} + 2 \times \text{填充}[0] - \text{卷积核尺寸}[0]}{\text{步长}[0]} + 1 \rfloor
$$


$$
W_{out} = \lfloor \frac{W_{in} + 2 \times \text{填充}[1] - \text{卷积核尺寸}[1]}{\text{步长}[1]} + 1 \rfloor
$$


**示例：**

假设我们有一个形状为`(16, 3, 32, 32)`的输入张量（批=16，通道=3，高=32，宽=32）。我们将其通过一个定义如下的`nn.Conv2d`层：

```python
import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# 输入: N=16, Cin=3, Hin=32, Win=32
input_tensor = torch.randn(16, 3, 32, 32)

# 参数: K=3, S=1, P=1, D=1 (默认)
# H_out = floor((32 + 2*1 - 1*(3-1) - 1)/1 + 1) = floor((32 + 2 - 2 - 1)/1 + 1) = floor(31/1 + 1) = 32
# W_out = floor((32 + 2*1 - 1*(3-1) - 1)/1 + 1) = floor((32 + 2 - 2 - 1)/1 + 1) = floor(31/1 + 1) = 32
# 简化公式 (D=1):
# H_out = floor((32 + 2*1 - 3)/1 + 1) = floor(31/1 + 1) = 32
# W_out = floor((32 + 2*1 - 3)/1 + 1) = floor(31/1 + 1) = 32

output_tensor = conv_layer(input_tensor)
print(output_tensor.shape)
# 预期输出: torch.Size([16, 64, 32, 32])
```

在此示例中，使用`kernel_size=3`、`stride=1`和`padding=1`是一种常见组合，它保持了输入的高度和宽度（`32x32` -> `32x32`），同时将通道数从3变为64。这有时被称为“相同”填充，尽管PyTorch不像其他一些框架那样有明确的`'same'`选项；你可以通过正确设置参数来实现。

如果我们将步长改为2（`stride=2`），输出维度将会减小：

```python
conv_layer_s2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
# H_out = floor((32 + 2*1 - 3)/2 + 1) = floor(31/2 + 1) = floor(15.5 + 1) = floor(16.5) = 16
# W_out = floor((32 + 2*1 - 3)/2 + 1) = floor(31/2 + 1) = floor(15.5 + 1) = floor(16.5) = 16

output_tensor_s2 = conv_layer_s2(input_tensor)
print(output_tensor_s2.shape)
# 预期输出: torch.Size([16, 64, 16, 16])
```

### 池化层 (`nn.MaxPool2d`)

池化层，例如`nn.MaxPool2d`，用于减小特征图的空间维度（下采样），使表示更紧凑且对小的平移具有鲁棒性。它们在每个通道上独立操作。

影响形状的主要参数 (parameter)与`nn.Conv2d`相似，但没有`out_channels`这个参数，因为池化不会改变通道数量：

- `kernel_size`: 池化窗口的大小。
- `stride`: 窗口的步长。通常设置为与`kernel_size`相等，以实现不重叠的池化（默认值是`kernel_size`）。
- `padding`: 添加的零填充量。
- `dilation`: 控制池化元素之间的间距。

输出形状 $(N, C_{out}, H_{out}, W_{out})$ 中 $H_{out}$ 和 $W_{out}$ 的计算遵循与卷积层完全相同的公式：


$$
H_{out} = \lfloor \frac{H_{in} + 2 \times \text{填充}[0] - \text{空洞}[0] \times (\text{卷积核尺寸}[0] - 1) - 1}{\text{步长}[0]} + 1 \rfloor
$$


$$
W_{out} = \lfloor \frac{W_{in} + 2 \times \text{填充}[1] - \text{空洞}[1] \times (\text{卷积核尺寸}[1] - 1) - 1}{\text{步长}[1]} + 1 \rfloor
$$


**重要区别：** 池化层**不**改变通道数量。因此，$C_{out} = C_{in}$。

**示例：**

让我们使用我们第一个`conv_layer`的输出（形状为`[16, 64, 32, 32]`）并将其通过一个常见的最大池化层：

```python
# 来自前一个卷积层的输入: N=16, Cin=64, Hin=32, Win=32
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 常见设置

# 参数: K=2, S=2, P=0, D=1 (默认)
# H_out = floor((32 + 2*0 - 1*(2-1) - 1)/2 + 1) = floor((32 - 1 - 1)/2 + 1) = floor(30/2 + 1) = floor(15 + 1) = 16
# W_out = floor((32 + 2*0 - 1*(2-1) - 1)/2 + 1) = floor((32 - 1 - 1)/2 + 1) = floor(30/2 + 1) = floor(15 + 1) = 16

pooled_output = pool_layer(output_tensor)
print(pooled_output.shape)
# 预期输出: torch.Size([16, 64, 16, 16])
```

在此，具有2x2卷积核和步长为2的池化层将高度和宽度维度减半（`32x32` -> `16x16`），而通道数量保持不变（64）。

> 张量维度通过一个示例卷积和池化层序列的流向。

### 实际中跟踪形状

在构建复杂的CNN时，手动计算形状会变得繁琐且容易出错。这里有一些实用建议：

1. **打印形状：** 在初始开发阶段，在层之后添加`print(x.shape)`语句以验证维度。
2. **使用虚拟输入：** 创建一个具有预期形状的虚拟输入张量，并将其逐步或逐层地通过你的网络定义，以查看形状如何变化。
3. **辅助函数：** 编写一个小的辅助函数，以层和输入形状作为参数 (parameter)，并使用公式计算输出形状。
4. **库/工具：** 一些库或工具（如`torchinfo`或`pytorch-summary`）可以自动总结你的模型，显示给定输入尺寸下每个层的输出形状。

掌握形状计算是在设计和调试CNN时必要的一步。通过理解`kernel_size`、`stride`、`padding`和`dilation`如何影响空间维度，以及`out_channels`如何决定深度，你可以放心地堆叠层来构建有效的深度学习 (deep learning)模型。

## 参考资料

- [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) — PyTorch Documentation (2023)
  Publisher: PyTorch Foundation
  提供`nn.Conv2d`层的官方细节和计算公式，涵盖参数和输出形状计算。
- [MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) — PyTorch Documentation (2023)
  提供`nn.MaxPool2d`层的官方细节和计算公式，涵盖参数和输出形状计算。
- [Deep Learning (Chapter 9: Convolutional Networks)](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  为卷积网络提供全面的理论基础，包括卷积和池化操作的数学原理。
- [Convolutional Neural Networks (CNNs) for Visual Recognition](http://cs231n.github.io/convolutional-networks/) — Andrej Karpathy and Fei-Fei Li (2023)
  Journal: Stanford CS231n Course Notes
  解释CNN的入门材料，包含卷积层和池化层的空间维度计算的实用方面。
