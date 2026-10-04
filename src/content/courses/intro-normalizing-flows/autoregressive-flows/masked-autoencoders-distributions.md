---
course: "intro-normalizing-flows"
chapter: "autoregressive-flows"
lesson: "masked-autoencoders-distributions"
sourceId: 8264
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-3-autoregressive-flows/masked-autoencoders-distributions"
title: "分布估算遮罩自编码器"
description: "了解 MADE 架构如何利用二进制遮罩高效地强制执行自回归特性。"
order: 2
plots: []
sourceHash: "5fb643f8ba0c9510bd97958389184a1e1fde199a9cee066d756cb0b8ac97aedd"
sourceCorrections: []
---

使用传统的序列网络评估自回归 (autoregressive)模型的联合概率，需要逐一计算每个条件概率。对于一个 100 维的数据集，每个数据点需要 100 次序列操作。这种序列依赖性构成了计算瓶颈，使得在专为并行执行设计的现代硬件上训练速度较慢。

分布估算遮罩自编码器（MADE）为这个问题提供了一个巧妙的数学方案。MADE 不使用循环结构，而是通过修改标准的神经网络 (neural network)，在单次前向传播中同时处理所有维度。它通过在网络的权重 (weight)矩阵上应用二进制遮罩来实现这一点。这些遮罩将特定的连接清零，从而确保对维度 $i$ 的预测仅依赖于严格小于 $i$ 的维度。

### 遮罩机制

在标准的稠密层中，每个输出节点都与每个输入节点相连。这种全连接结构违反了自回归 (autoregressive)特性，因为代表 $p(x_i | x_{<i})$ 的输出能够获取来自 $x_i$ 以及 $x_{i+1}$ 等后续维度的信息。

MADE 通过为网络中的每个节点分配一个称为“度”（degree）的整数来解决这个问题。度决定了哪些连接是允许的，哪些连接必须断开。

分配遵循以下规则：

1. **输入节点：** 每个输入维度 $x_d$ 获得一个与其索引相等的度 $m(d)$，范围从 $1$ 到 $D$。
2. **隐藏节点：** 隐藏层中的每个节点获得一个从 $[1, D-1]$ 范围内随机抽取的度。范围止于 $D-1$ 是因为如果一个隐藏单元参与预测最终输出，那么它一定不能包含关于最后一个输入维度的信息。
3. **输出节点：** 每个输出节点对应一个特定的输入维度，并获得与该输入相同的度，范围从 $1$ 到 $D$。

分配好度之后，我们为网络权重 (weight)构建二进制遮罩矩阵。对于从第 $l-1$ 层的节点 $k$ 到第 $l$ 层的节点 $j$ 的连接，遮罩 $M$ 通过比较它们的度来定义。

对于隐藏层之间的连接，信息可以流向度数相等或更高的节点。遮罩计算如下：

$M^{l}_{j, k} = \begin{cases} 1 & \text{如果 } m^l(j) \ge m^{l-1}(k) \\ 0 & \text{否则} \end{cases}$

对于到最终输出层的连接，条件变为严格大于。这确保了 $x_i$ 的输出不会接收到关于 $x_i$ 本身的信息：

$M^{L}_{j, k} = \begin{cases} 1 & \text{如果 } m^L(j) > m^{l-1}(k) \\ 0 & \text{否则} \end{cases}$

在前向传播过程中，在计算线性变换之前，权重矩阵 $W$ 与遮罩 $M$ 进行逐元素相乘：

$h^l = \text{激活函数}((W^l \odot M^l) h^{l-1} + b^l)$

> 简单 MADE 架构的有向图。二进制遮罩限制了信息流，使得输出 $p(x_3)$ 仅依赖于包含来自 $x_1$ 和 $x_2$ 信息的隐藏节点。输出 $p(x_1)$ 不依赖任何输入，实际上是在建模边际分布。

### 在 PyTorch 中实现遮罩线性层

要构建 MADE 模型，第一步是创建一个自定义的 PyTorch 模块，将这些二进制遮罩应用于标准的线性层。继承 `torch.nn.Linear` 使我们能够使用优化的原生操作，同时注入遮罩逻辑。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        # 将遮罩注册为 buffer，这样 PyTorch 会管理其设备分配，
        # 且不会将其视为可训练参数。
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        """更新该层的二进制遮罩。"""
        self.mask.data.copy_(mask)

    def forward(self, x):
        # 将遮罩逐元素应用于权重矩阵
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)
```

通过将遮罩注册为 buffer，当对模型调用 `.to('cuda')` 时，PyTorch 会自动将其移动到 GPU，但优化器知道在反向传播 (backpropagation)期间不更新它。要完成 MADE 网络，你需要堆叠多个 `MaskedLinear` 层，为每层的节点分配度，使用度比较规则生成布尔矩阵，并为每层调用 `set_mask` 方法。

### 输出与密度估算

在归一化 (normalization)流的背景下，MADE 网络并不直接输出概率值。相反，它输出基础分布的参数 (parameter)。例如，如果假设每个条件分布都是高斯分布，MADE 网络将为每个维度 $i$ 输出两个值：均值 $\mu_i$ 和对数标准差 $\log \sigma_i$。

由于网络在单次前向传播中生成所有参数，计算训练批次的对数似然变得非常高效。将数据 $x$ 输入网络，接收 $\mu$ 和 $\log \sigma$，然后使用这些参数评估 $x$ 的概率密度。

这种单次传递的特性使 MADE 成为密度估算任务的优良组件。然而，生成新样本需要反向传递。采样是严格序列化的，因为必须先采样 $x_1$，将其反馈回网络以获得 $x_2$ 的参数，接着采样 $x_2$，以此类推。这一特性决定了在选择特定自回归 (autoregressive)流架构时需要面对的权衡。
