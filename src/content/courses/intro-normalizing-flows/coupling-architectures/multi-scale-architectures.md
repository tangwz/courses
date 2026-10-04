---
course: "intro-normalizing-flows"
chapter: "coupling-architectures"
lesson: "multi-scale-architectures"
sourceId: 8270
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-4-coupling-architectures/multi-scale-architectures"
title: "多尺度架构"
description: "使用多尺度架构降低计算成本并获取层次化特征。"
order: 3
plots: []
sourceHash: "c4956ee6966b344ee9f53bfcf105b42f50ece651c5f45a49fda4a5ead1cc97f8"
sourceCorrections: []
---

处理高维数据（例如高分辨率图像）通过数十个归一化 (normalization)流层需要大量计算资源。在标准架构中，输入向量 (vector) $x$ 的每个维度都会经过模型的每一层。在处理具有数十万个独立分量的输入时，每一步都计算仿射耦合变换及其雅可比矩阵会成为严重的瓶颈。

为了减轻这种计算负担，我们采用多尺度架构。这种设计模式由 RealNVP 模型推广。多尺度架构不是在每一层都变换全部数据量，而是定期分出一部分变量。这些分出的变量直接发送到最终输出，而只有剩余的变量继续通过流网络的其余部分。

让我们从数学和结构上分析其工作原理。假设在第 $i$ 阶段有一个中间张量 $h_i$。我们对 $h_i$ 应用一系列流操作，然后将结果沿通道维度拆分为两半：

$z_i, h_{i+1} = \text{拆分}(\text{流}(h_i))$

张量 $z_i$ 代表被分出并完成处理的变量。我们不再对其进行进一步处理。张量 $h_{i+1}$ 包含传递到网络下一阶段的活动变量。

> 多尺度架构在中间阶段分出变量，以减少后续流模块的计算负荷。

这种结构选择对概率密度估计有三个显著优点。

首先，它直接降低了内存和计算需求。由于每次拆分都会舍弃一半变量，张量的空间或通道维度会缩小。后续耦合层中用于计算缩放和平移的神经网络 (neural network)现在的输入规模小得多。

其次，它产生了一种自然的学习特征层次结构。流的早期层在数据的完整空间分辨率上运行。这里分出的变量往往代表纹理和边缘等细粒度的局部细节。随着数据通过挤压操作和后续拆分，空间分辨率降低，但感受野有效地增加了。在网络末端分出的变量捕捉的是全局结构和语义信息。

第三，分出变量将损失函数 (loss function)的计算分散到整个网络中。在精确极大似然估计中，我们要优化基分布的对数概率与雅可比对数行列式的总和。通过在多个中间阶段对 $z_i$ 评估基分布，我们为早期层提供了更短的梯度路径。这大大提升了深层模型训练的稳定性。

多尺度模型中的总对数似然是所有分出组件似然的总和。如果一个模型有 $L$ 个尺度，目标函数会汇总所有 $z_i$ 输出的对数概率，加上沿活动路径计算的所有雅可比对数行列式之和。

$\log p(x) = \sum_{i=1}^{L} \log p(z_i) + \sum_{j} \log \left| \det \frac{\partial h_j}{\partial h_{j-1}} \right|$

要在 PyTorch 中实现这一点，需要一种切分张量并存储完成变量的机制。下面是一个展示拆分操作在前向传播中如何工作的实际示例。

```python
import torch
import torch.nn as nn

class MultiScaleSplit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 将输入张量沿通道维度 (dim=1) 拆分为两半
        channels = x.shape[1]
        z, h = torch.split(x, channels // 2, dim=1)

        # 由于拆分是具有恒等变换的线性操作，
        # 该特定步骤的雅可比对数行列式为零。
        log_det = torch.zeros(x.shape[0], device=x.device)

        return z, h, log_det

    def inverse(self, z, h):
        # 将分出的变量与活动变量重新拼接
        x = torch.cat([z, h], dim=1)
        return x
```

在用于生成新数据的逆向过程中，该流程是反向的。你从基分布（通常是标准正态分布）中采样所有独立的划分 $z_1, z_2, \dots, z_L$。你让最后的划分 $z_L$ 反向通过最后一个流模块。然后，将结果与 $z_{L-1}$ 拼接，让合并后的张量反向通过前一个流模块，并重复此过程，直到重构出全维度的输出 $x$。

这种架构确保了归一化流在处理高维数据集时依然可行。在扩展图像生成的耦合层时，你会频繁使用这些多尺度模式。
