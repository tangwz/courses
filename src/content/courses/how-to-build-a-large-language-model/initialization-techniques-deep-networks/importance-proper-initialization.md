---
course: "how-to-build-a-large-language-model"
chapter: "initialization-techniques-deep-networks"
lesson: "importance-proper-initialization"
sourceId: 6006
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/importance-proper-initialization"
title: "恰当初始化的重要性"
description: "阐述初始化如何影响收敛速度和训练稳定性。"
order: 1
plots: []
sourceHash: "db809715b1b23437c739fbca750f04271009103d4da8002f61e9a2af2d597c12"
sourceCorrections: []
---

训练深度神经网络 (neural network)，特别是现代LLM中使用的巨型Transformer架构，需要仔细关注诸多细节。模型参数 (parameter)的初始状态，特别是其权重 (weight)，是影响训练过程的一个主要方面。简单地将权重初始化为零或从朴素分布中抽取的微小随机数，常导致严重的训练难题。恰当的初始化不只是一种启发式方法；它是使深度网络能够有效学习的基本要求。

核心问题源于信号（包括前向传播时的激活和反向传播 (backpropagation)时的梯度）如何在网络层间传递。设想单层内的计算，通常涉及矩阵乘法后接非线性激活函数 (activation function)。当多层这样堆叠时，这些操作会依序重复。

### 梯度消失

在反向传播 (backpropagation)过程中，损失函数 (loss function)对早期层中某个权重 (weight)的梯度，是使用链式法则计算的，即乘以所有后续层的梯度。如果这些梯度（或层变换的雅可比矩阵）的平均大小小于1，梯度信号在反向传播时会指数级减小。


$$
\frac{\partial L}{\partial W_l} \propto \left( \prod_{k=l+1}^{N} J_k \right) \frac{\partial L}{\partial a_N}
$$


这里，$J_k$ 表示第 $k$ 层的雅可比矩阵。如果平均而言 $\|J_k\| < 1$，乘积项会随着层数 $N-l$ 的增加而迅速减小。

这种现象被称为**梯度消失问题**，意味着早期层中的权重得到极其微小的更新。因此，这些层学习非常缓慢，有时甚至完全不学。这实际上阻止了网络学习依赖于其全部深度参数 (parameter)配合调整的复杂表示。历史上，这是训练深度网络的一大障碍，尤其是那些使用如sigmoid或tanh等激活函数 (activation function)的网络，这些函数在饱和区域的导数小于1。

### 梯度爆炸

反之，如果这些梯度（或雅可比矩阵）的平均大小大于1，梯度信号在反向通过各层时会指数级增长。


$$
\| \frac{\partial L}{\partial W_l} \| \rightarrow \infty \quad \text{as } N-l \rightarrow \infty \quad \text{if } \|J_k\| > 1
$$


这种**梯度爆炸问题**导致权重 (weight)更新过大。过大的更新会使优化过程变得不稳定，可能导致越过最佳点或剧烈震荡。在极端情况下，梯度变得如此之大，以至于导致数值溢出（表示为`NaN`或`Inf`值），使训练过程停滞。尽管梯度裁剪等方法（将在第17章讨论）可以在训练期间缓解梯度爆炸，但恰当的初始化旨在从一开始就防止它们频繁出现。

让我们看一个简化的模拟。设想一个具有10层的非常简单的线性网络。我们将权重初始化为略小或略大的值，并观察一个模拟梯度反向传播 (backpropagation)后的大小。

```python
import torch
import torch.nn as nn
import math

# 模拟信号传播（简化）
def check_grad_magnitude(init_scale, num_layers=10):
    """
    模拟通过线性层的反向梯度大小。
    """
    # 这里输入维度不那么重要，侧重于尺度
    layer_dim = 100
    layers = []
    for _ in range(num_layers):
        layer = nn.Linear(layer_dim, layer_dim, bias=False)
        # 使用特定的标准差初始化权重
        nn.init.normal_(layer.weight, mean=0.0, std=init_scale)
        layers.append(layer)

    network = nn.Sequential(*layers)

    # 模拟输入和输出梯度
    x = torch.randn(1, layer_dim)
    # 假设输出梯度大小为1
    output_grad = torch.ones(1, layer_dim)

    # 计算相对于输入的梯度，使用链式法则模拟
    current_grad = output_grad
    # 手动反向传播以观察大小变化
    with torch.no_grad():
        for i in range(num_layers - 1, -1, -1):
            # 层输入梯度 = 层输出梯度 @ W的转置
            current_grad = current_grad @ layers[i].weight
            # 为了演示，进行归一化以避免实际的爆炸/消失
            # 在实际情况中，这种归一化不会逐层发生
            # current_grad /= math.sqrt(layer_dim) # 示例归一化因子

    # 计算相对于第一层输入的梯度大小
    # 这模拟了梯度到达最早参数的情况
    input_grad_norm = torch.norm(current_grad)
    return input_grad_norm.item()

# 检查大小
num_layers = 10
small_init_scale = 0.05 # 可能导致梯度消失
large_init_scale = 0.5  # 可能导致梯度爆炸
ideal_init_scale = math.sqrt(1.0 / 100) # 类似于Xavier/He的缩放提示

grad_norm_small = check_grad_magnitude(small_init_scale, num_layers)
grad_norm_large = check_grad_magnitude(large_init_scale, num_layers)
grad_norm_ideal = check_grad_magnitude(ideal_init_scale, num_layers)

print(
    f"Initialization Scale: {small_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_small:.4e}"
)
print(
    f"Initialization Scale: {ideal_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_ideal:.4e}"
)
print(
    f"Initialization Scale: {large_init_scale:.3f}, "
    f"Final Gradient Norm: {grad_norm_large:.4e}"
)

# 预期（近似）输出：
# Initialization Scale: 0.050, Final Gradient Norm: 9.5367e-08
# Initialization Scale: 0.100, Final Gradient Norm: 1.0000e+00
# Initialization Scale: 0.500, Final Gradient Norm: 9.7656e+06
```

上述简单模拟（不含归一化 (normalization)和非线性，它们会使情况更复杂）说明了权重尺度如何直接影响经过多层反向传播后的梯度大小。较小的初始尺度会导致梯度范数消失，而较大的尺度则会导致其爆炸。一个“理想”的尺度（例如Xavier/Glorot初始化中提示的 $1/\sqrt{\text{fan\_in}}$）有助于使梯度范数保持在其原始大小附近。

### 前向传播的稳定性

初始化也影响前向传播。如果权重 (weight)过大，线性层的输出会大幅增长，可能将激活函数 (activation function)的输入推到饱和区域（例如，对于sigmoid或tanh），使得梯度接近零。这再次阻碍了学习。反之，如果权重过小，激活值可能会逐层减弱，导致表示趋近于零，并降低网络的有效容量。

因此，我们将要讨论的Xavier和Kaiming初始化等原则性权重初始化策略的目标，是根据层维度谨慎设置权重的初始尺度。目的是确保前向传播中的激活和反向传播 (backpropagation)中的梯度在整个网络中保持合理的方差，从而防止信号消失或爆炸，进而促进更快、更稳定的训练收敛。这对本课程中非常重要的深度Transformer模型尤为重要。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot and Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: Proceedings of Machine Learning Research; Volume: 9; Pages: 249-256; DOI: [10.5555/3172186.3172237](https://doi.org/10.5555/3172186.3172237)
  提出了Glorot（Xavier）初始化方法，通过分析信号传播来缓解深度网络中的梯度消失/爆炸问题。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 1026-1034; DOI: [10.1109/ICCV.2015.122](https://doi.org/10.1109/ICCV.2015.122)
  提出了He（Kaiming）初始化方法，专为ReLU激活函数设计，基于稳定梯度流的原则。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面介绍了深度学习基础知识，包括梯度消失/爆炸和初始化策略的详细解释。
