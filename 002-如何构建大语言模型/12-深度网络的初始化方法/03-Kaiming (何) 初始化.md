# Kaiming (何) 初始化

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/kaiming-he-initialization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Xavier初始化旨在平衡使用$tanh$等对称激活函数 (activation function)的层的方差。然而，随着修正线性单元（ReLU）激活函数的普及，情况发生了显著变化。ReLU函数定义为$f(x) = max(0, x)$，引入了非对称性：它对所有负输入输出零。这意味着，平均而言，从ReLU单元输出的激活值可能有一半为零。Xavier初始化的原理是为对称激活函数设计的，不能完全处理这种行为。因此，将Xavier初始化与ReLU结合使用时，信号在前向传播过程中仍可能导致方差逐渐减小，这可能会减慢训练速度，或在非常深的神经网络 (neural network)中导致梯度消失。

认识到这种不匹配，何凯明等人G在其论文《Exploring Rectifiers: Surpassing Human-Level Performance on ImageNet Classification》中提出了一种专门为ReLU及其变体设计的初始化方案。其核心思想是在计算初始权重 (weight)的合适方差时，明确考虑ReLU引入的非线性。

### 推导 Kaiming 方差

让我们考虑一个线性层 $y = Wx + b$。假设输入 $x$ 的均值为零，并且权重 (weight) $W$ 独立初始化且均值为零，单个神经元 $i$ 的输出 $y_i$（激活前）的方差由下式给出：


$$
Var(y_i) = \sum_{j=1}^{n_{in}} Var(W_{ij} x_j)
$$


假设 $W_{ij}$ 和 $x_j$ 相互独立，且 $E[x_j]=0$：


$$
Var(W_{ij} x_j) = E[W_{ij}^2 x_j^2] - (E[W_{ij} x_j])^2
$$


$$
Var(W_{ij} x_j) = E[W_{ij}^2] E[x_j^2] - (E[W_{ij}] E[x_j])^2
$$


$$
Var(W_{ij} x_j) = Var(W_{ij}) Var(x_j)
$$


因此，对 $n_{in}$ 个输入（即扇入）求和：


$$
Var(y_i) = n_{in} Var(W) Var(x)
$$


现在，令 $z = f(y)$ 为应用ReLU激活函数 (activation function) $f$ 后的输出。何凯明等人的观点是ReLU如何影响方差。如果 $y$ 是均值为零的线性层的输出，它会对称地分布在零附近。ReLU将负值设为零。如果我们假设 $x$ 来自之前的ReLU层，则方差计算需要调整。然而，如果我们将注意力放在通过当前层并应用ReLU激活函数 $f$ 的前向传播上，我们有 $z_i = max(0, y_i)$。如果 $y_i$ 的均值为零且对称，则 $E[z_i^2] = \frac{1}{2} E[y_i^2]$。由于 $E[y_i]=0$，所以 $E[y_i^2] = Var(y_i)$。因此，$Var(z_i) \approx E[z_i^2] = \frac{1}{2} Var(y_i)$。

替换 $Var(y_i)$ 的表达式：


$$
Var(z_i) \approx \frac{1}{2} n_{in} Var(W) Var(x)
$$


为了保持信号传播的稳定性，我们希望激活函数输出的方差 ($Var(z_i)$) 大致等于层输入的方差 ($Var(x)$)。这要求：


$$
Var(z_i) = Var(x) \implies 1 \approx \frac{1}{2} n_{in} Var(W)
$$


解出所需的权重 $W$ 的方差：


$$
Var(W) = \frac{2}{n_{in}}
$$


这是Kaiming初始化针对ReLU激活函数在考虑前向传播（扇入模式）时的基本结果。类似地，考虑反向传播 (backpropagation)（梯度流）的推导会得出 $Var(W) = \frac{2}{n_{out}}$。

### Kaiming 初始化公式

基于此推导出的方差，我们可以使用正态分布或均匀分布来初始化权重 (weight)。

1. **Kaiming 正态初始化：** 权重从正态分布 $\mathcal{N}(0, \sigma^2)$ 中采样，其中标准差 $\sigma$ 为：

   - 扇入模式：$\sigma = \sqrt{\frac{2}{n_{in}}}$
   - 扇出模式：$\sigma = \sqrt{\frac{2}{n_{out}}}$
2. **Kaiming 均匀初始化：** 权重从均匀分布 $\mathcal{U}(-bound, bound)$ 中采样，其中边界值根据所需的方差计算得出：

   - $\mathcal{U}(-b, b)$ 的方差为 $\frac{b^2}{3}$。
   - 将 $\frac{b^2}{3} = \frac{2}{n_{mode}}$（其中 $n_{mode}$ 为 $n_{in}$ 或 $n_{out}$）设定，得到 $b^2 = \frac{6}{n_{mode}}$。
   - 因此，$bound = \sqrt{\frac{6}{n_{mode}}}$。
   - 扇入模式：$bound = \sqrt{\frac{6}{n_{in}}}$
   - 扇出模式：$bound = \sqrt{\frac{6}{n_{out}}}$

“扇入”模式通常更受青睐，因为它在前向传播过程中保持方差。

### PyTorch 中的实现

PyTorch 在 `torch.nn.init` 模块中提供了方便的 Kaiming 初始化函数。

```python
import torch
import torch.nn as nn
import math

# Transformer FFN 中典型的线性层示例
fan_in = 2048  # d_model 示例
fan_out = 8192
# 前馈维度示例（通常为 4*d_model）
linear_layer = nn.Linear(fan_in, fan_out, bias=False)
# 偏置通常初始化为零

# --- Kaiming 正态初始化（扇入模式，针对 ReLU）---
# 'a=0' 是 ReLU 的默认值。对于 Leaky ReLU，请使用不同的值作为
# 斜率。
# 'mode=fan_in' 在前向传播中保持方差。
# 'nonlinearity=relu' 指定了适用于 ReLU 的增益计算。
# ReLU。
nn.init.kaiming_normal_(
    linear_layer.weight,
    mode='fan_in',
    nonlinearity='relu'
)

print("Kaiming 正态初始化权重（形状，样本）：")
print(linear_layer.weight.data.shape)
print(linear_layer.weight.data[0, :5])
actual_var_normal = linear_layer.weight.data.var()
expected_var = 2.0 / fan_in
print(f"\n方差（正态）：{actual_var_normal:.6f}")
print(f"预期方差 (2 / fan_in)：{expected_var:.6f}")

# --- Kaiming 均匀初始化（扇入模式，针对 ReLU）---
linear_layer_uniform = nn.Linear(fan_in, fan_out, bias=False)
nn.init.kaiming_uniform_(
    linear_layer_uniform.weight,
    mode='fan_in',
    nonlinearity='relu'
)

print("\nKaiming 均匀初始化权重（形状，样本）：")
print(linear_layer_uniform.weight.data.shape)
print(linear_layer_uniform.weight.data[0, :5])
actual_var_uniform = linear_layer_uniform.weight.data.var()
# 预期方差仍为 2 / fan_in
print(f"\n方差（均匀）：{actual_var_uniform:.6f}")
print(f"预期方差 (2 / fan_in)：{expected_var:.6f}")

# --- 偏置初始化 ---
# 偏置通常初始化为零
bias_tensor = torch.zeros(fan_out)
print("\n偏置初始化（示例）：")
print(bias_tensor[:5])
```

在代码中，`nonlinearity='relu'` 告诉函数使用与ReLU相关的增益因子，即 $\sqrt{2}$。这个因子直接来自推导中需要 $Var(W) = 2/n_{in}$。如果您使用带有负斜率 `a` 的 Leaky ReLU，您将设置 `nonlinearity='leaky_relu'`，并可能调整 `a` 参数 (parameter)，它会相应地调整增益计算。`mode='fan_in'` 确保方差计算使用输入特征的数量 ($n_{in}$)。

### Transformer 中的适用性

Kaiming初始化是Transformer中位置前馈网络（FFN）内权重 (weight)矩阵的通用选择，因为这些网络通常使用ReLU或其近似函数，如GeLU或SwiGLU。尽管GeLU和SwiGLU并非严格意义上的ReLU，但Kaiming初始化通常是一个良好的起始点。对于嵌入 (embedding)层和注意力机制 (attention mechanism)中的线性投影，可能会采用不同的策略（通常更接近Xavier正态初始化或简单的缩放标准正态分布），但对于由类ReLU函数激活的核心FFN层，Kaiming初始化对于实现深度堆叠网络的训练非常重要。

通过专门针对ReLU激活函数 (activation function)的特性，Kaiming初始化为主要使用这些单元的深层网络提供了一种方法。它在阻止信号方差迅速衰减方面起到重要作用，从而有助于大型模型（如现代Transformer）的稳定高效训练。

## 参考资料

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 1026-1034; DOI: [10.1109/ICCV.2015.123](https://doi.org/10.1109/ICCV.2015.123)
  介绍He初始化（Kaiming初始化）的基础论文，提供了其在深度网络中使用ReLU激活时的推导和经验验证。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本全面涵盖深度学习基础知识的教科书，包括Xavier和He等多种权重初始化策略及其理论基础。
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Developers (2022)
  Publisher: PyTorch Foundation
  PyTorch权重初始化函数的官方文档，包括`kaiming_normal_`和`kaiming_uniform_`，详细说明了其用法和参数。
- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 9; Pages: 249-256
  介绍了Xavier初始化，它是Kaiming初始化的前身，专为对称激活函数设计，为Kaiming解决的问题提供了背景。

---

[上一节](02-Xavier%EF%BC%88Glorot%EF%BC%89%E5%88%9D%E5%A7%8B%E5%8C%96.md) · [下一节](04-Transformer%E7%BB%84%E4%BB%B6%E4%B8%AD%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96.md)
