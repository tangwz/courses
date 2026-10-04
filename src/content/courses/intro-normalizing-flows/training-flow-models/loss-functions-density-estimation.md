---
course: "intro-normalizing-flows"
chapter: "training-flow-models"
lesson: "loss-functions-density-estimation"
sourceId: 8274
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-5-training-flow-models/loss-functions-density-estimation"
title: "密度估计的损失函数"
description: "实现包含雅可比行列式的负对数似然损失函数。"
order: 2
plots: []
sourceHash: "65e718643ed44bf76918dc5a001378202c4bf2e587ccbf9590f32caa2485651f"
sourceCorrections: []
---

在 PyTorch 中优化神经网络 (neural network)，需要将精确的对数似然方程转换为可计算的损失函数 (loss function)。由于深度学习 (deep learning)框架旨在通过梯度下降 (gradient descent)来最小化目标函数，因此最大似然估计问题通过对对数似然取负值，转变为一个最小化问题。这便得到了负对数似然（Negative Log-Likelihood, NLL）损失。

对于单个数据点 $x$、可逆变换 $f$ 和基分布 $Z$，负对数似然的数学表达式为：

$\mathcal{L}(x) = - \left( \log p_Z(f(x)) + \log \left| \det J_f(x) \right| \right)$

该损失函数由模型在训练期间必须平衡的两个不同部分组成。第一项是变换后的数据点在基分布下的对数概率。第二项是雅可比矩阵的对数行列式，它补偿了由变换引起的体积变化。

我们先来看基分布项。在实际应用中，我们几乎总是选择标准多元高斯分布作为基分布 $Z$。对于一个 $D$ 维向量 (vector) $z = f(x)$，标准正态分布的对数密度计算如下：

$\log p_Z(z) = -\frac{D}{2} \log(2\pi) - \frac{1}{2} \sum_{i=1}^{D} z_i^2$

最小化该项的负值会促使网络将数据点映射到隐空间中靠近原点的位置。如果我们只优化这一项，网络会简单地将每个输入 $x$ 都映射到 $z = 0$，以获得尽可能低的损失。

这就是雅可比对数行列式作为防止体积坍缩的精确正则项发挥作用的地方。如果网络试图将整个输入空间压缩到一个点，对数行列式项将对模型施加沉重惩罚，因为计算出的空间体积正在剧烈收缩。网络被迫寻找一种映射，使数据点根据高斯形状分布，而不会将它们重叠在同一个位置。

> 负对数似然损失的前向传播计算图。

在 PyTorch 中实现这一点时，归一化 (normalization)流架构应当设计为：前向传播同时返回变换后的输出张量 $z$ 以及一个包含 batch 中每个样本累积雅可比对数行列式的一维张量。

以下是编写自定义 PyTorch 模块以计算负对数似然损失的方法：

```python
import torch
import torch.nn as nn
import math

class FlowLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, log_det_J):
        # z 形状: (batch_size, D)
        # log_det_J 形状: (batch_size,)

        D = z.shape[1]

        # 计算标准正态分布的对数概率
        # log p(z) = - (D/2) * log(2*pi) - 0.5 * sum(z^2)
        log_p_z = -0.5 * D * math.log(2 * math.pi) - 0.5 * torch.sum(z ** 2, dim=1)

        # 计算精确的对数似然
        log_likelihood = log_p_z + log_det_J

        # 返回 batch 内的平均负对数似然
        loss = -torch.mean(log_likelihood)

        return loss
```

注意 `torch.sum` 操作中的参数 (parameter) `dim=1`。因为我们是以 batch 形式处理数据，变量 `z` 的形状为 `(batch_size, D)`。我们在特征维度上对平方值求和，以计算 batch 中每个独立样本的对数概率。变量 `log_det_J` 的预期形状为 `(batch_size,)`，代表每个对应样本的累积体积变化。

我们将这两个张量相加，得到每个样本的最终对数似然。最后，我们使用 `-torch.mean(...)` 取 batch 的平均值。使用平均值而非总和可以确保即使在训练过程中更改了 batch 大小，学习率也能保持稳定。

虽然原始的负对数似然足以进行梯度下降优化，但它的具体数值高度依赖于输入数据的维度。在对数字图像等高维数据集评估流模型时，研究人员通常使用一种称为单位维度比特数 (Bits Per Dimension, BPD) 的标准化指标来报告损失。要计算 BPD，我们将负对数似然除以总维度数，并将自然对数转换为以 2 为底的对数。

$\text{BPD} = \frac{\text{NLL}}{D \cdot \ln(2)}$

该公式提供了一个标准化的指标，用于比较不同模型在不同规模和分辨率数据集上的密度估计性能。较低的 BPD 意味着模型对底层数据分布的拟合效果更好。
