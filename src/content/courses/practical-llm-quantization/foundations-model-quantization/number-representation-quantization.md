---
course: "practical-llm-quantization"
chapter: "foundations-model-quantization"
lesson: "number-representation-quantization"
sourceId: 4598
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-1-foundations-model-quantization/number-representation-quantization"
title: "数值表示：浮点数与定点数"
description: "比较与量化相关的浮点数和定点数表示。"
order: 3
plots: []
sourceHash: "f66d79fe9f3f851743b765aee166cec00a80fe2c565781460e14963d4b77f978"
sourceCorrections: []
---

为了明白量化 (quantization)如何实现模型压缩与加速，我们首先需要了解数值在计算机中如何以数字形式表示。大型语言模型，像多数深度学习 (deep learning)模型一样，通常使用浮点数进行计算。量化包含将这些数值转换为需要更少内存的格式，通常是定点数或整数表示。

### 浮点数表示

浮点数是计算机表示实数（带小数部分的数值）的标准方式。它们提供宽广的动态范围，这意味着它们能表示极小和极大的数值，并且具有不错的精度。机器学习 (machine learning)中常见的格式包含：

1. **FP32（单精度）：** 每个数值使用32位。这是许多框架中训练和基准推理 (inference)的默认格式。它包含1位用于符号，8位用于指数（决定数值大小），以及23位用于尾数或有效数字（决定精度）。
2. **FP16（半精度）：** 使用16位。与FP32相比，它将内存使用量减少一半。它有1个符号位，5个指数位，和10个尾数位。较小的指数范围使得当数值跨越非常宽广的范围时，它更容易出现下溢（数值变为零）或上溢（数值变为无穷大）问题。
3. **BF16（BFloat16 - Brain Floating-Point）：** 也使用16位。它保持与FP32相同的指数范围（8位），但将尾数减少到7位（加上1个符号位）。这保留了FP32的动态范围，使其与FP16相比不易出现上溢/下溢问题，但其精度较低。

> 深度学习 (deep learning)中常见浮点格式的结构。

虽然浮点数提供了灵活性，但其存储和计算需求可能相当可观，特别是对于拥有数十亿参数 (parameter)的大型语言模型。对32位浮点数的操作通常比对较小整数类型的操作更慢且更耗能。

### 定点数表示

定点数表示是表示实数的一种替代方法。与浮点数不同，浮点数中小数点（或二进制小数点）的位置可以“浮动”，而在定点数表示中，二进制小数点的位置是固定的。

一个定点数本质上是一个由预设因子隐式缩放的整数。例如，您可能决定使用一个8位带符号整数（范围从-128到127）来表示-1.0到+1.0之间的数值。为此，您需要定义一个缩放因子。如果您选择缩放因子为 $2^7 = 128$，那么：

- 整数值127将表示 $127 / 128 \approx 0.992$。
- 整数值-128将表示 $-128 / 128 = -1.0$。
- 整数值0将表示 $0 / 128 = 0.0$。

分配给整数部分和分数部分的位数，决定了范围和精度。

- **整数位越多：** 范围越广，精度越低。
- **小数位越多：** 范围越窄，精度越高。

与浮点数相比：

- **范围：** 定点数具有更有限的动态范围。超出预定范围的数值无法被精确表示。
- **精度：** 在可表示的范围内，精度是均匀的。
- **硬件：** 定点数的算术运算（加法、乘法）通常可以通过硬件上更简单、更快速的整数逻辑单元来实现。

### 量化 (quantization)中的整数表示

许多量化技术会将FP32权重 (weight)和/或激活值直接转换为低精度整数类型，例如：

- **INT8：** 8位整数。可以是带符号的（-128到127）或无符号的（0到255）。
- **INT4：** 4位整数。通常是带符号的（-8到7）或无符号的（0到15）。

这些整数类型可大幅节省内存（例如，INT8的内存用量是FP32的四分之一），并能让配备专用整数算术单元的硬件（常存在于现代CPU和GPU/TPU中）进行快得多的计算。

当我们在量化中使用INT8或INT4时，我们实际上是在使用定点数表示。从浮点数转换的过程涉及确定一个**比例因子**（类似于上述定点数示例）以及通常一个**零点**（或偏移量）。这些参数 (parameter)将原始浮点数范围映射到目标整数范围。

举例而言，要将浮点数范围 $[R_{min}, R_{max}]$ 映射到8位整数范围 $[0, 255]$，映射方式可能如下：

$\text{整数值} = \text{取整}\left( \frac{\text{浮点数值} - Z}{S} \right)$

其中：

- $S$ 是比例因子（一个浮点数）。
- $Z$ 是零点（通常对应于浮点数0.0，但在目标范围中表示为整数）。

将浮点数值映射到整数格式时，被量化的数值通常在一个特定范围之内。例如，如果考虑一个浮点数值0.5和比例因子(S)0.1，对应的量化整数将是5（0.5 / 0.1 = 5）。其中 S 是比例因子（浮点数），Z 是零点（整数）。S和Z的选择很重要，因为它们定义了从原始浮点数范围到整数范围的映射，影响着精度和准确性。

此映射使我们能够使用高效的整数算术进行计算，仅在必要时才转换回浮点数。浮点数与定点数/整数表示之间的选择涉及数值范围/精度与计算效率/内存使用之间的权衡。量化旨在利用低精度格式的效率，同时尽量减少对模型准确性的影响。后续章节将介绍此映射在实践中如何实现。

## 参考资料

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://dl.acm.org/doi/10.1145/103162.103163) — David Goldberg (1991)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery, Inc.; Volume: 23; Pages: 5-48; DOI: [10.1145/103162.103163](https://doi.org/10.1145/103162.103163)
  解释浮点运算的基本原理，包括IEEE 754标准，并讨论常见问题。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 2704-2713; DOI: [10.48550/arXiv.1712.05877](https://doi.org/10.48550/arXiv.1712.05877)
  介绍神经网络的关键量化技术，包括对高效整数运算推理至关重要的尺度和零点映射。
- [Computer Organization and Design RISC-V Edition: The Hardware/Software Interface](https://www.whsmith.co.uk/products/computer-organization-and-design-risc-v-edition--the-hardware-software-interface-paperbacksoftback/david-a-patterson/john-l-hennessy/paperback/9780128203316.html) — David A. Patterson, John L. Hennessy (2021)
  Publisher: Morgan Kaufmann; Pages: 736 pages
  第6版。这本教科书提供了计算机体系结构的概览，涵盖数字表示、算术单元和数据类型。
