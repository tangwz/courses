# Xavier（Glorot）初始化

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-12-initialization-techniques-deep-networks/xavier-glorot-initialization)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如前所述，初始化深度网络的权重 (weight)需要仔细考虑。简单地从标准正态或均匀分布中抽取权重通常会导致训练过程不稳定。梯度在反向传播 (backpropagation)时可能呈指数级缩小（梯度消失），或呈指数级增大（梯度爆炸），阻碍模型有效学习。

Xavier 初始化方法由 Xavier Glorot 和 Yoshua Bengio 于 2010 年提出，旨在解决深度网络训练中梯度消失或爆炸等常见挑战，其原理是使激活值和梯度的方差在各层之间大致保持不变。保持方差一致有助于确保信号（前向传播时的激活值和反向传播时的梯度）在穿过网络时不会消失或爆炸。

### 核心思想：方差保持

考虑神经网络 (neural network)中的一个线性层，执行操作 $y = Wx + b$，其中 $W$ 是权重 (weight)矩阵，$x$ 是输入向量 (vector)，$b$ 是偏置 (bias)向量，$y$ 是激活函数 (activation function)前的输出。Xavier 初始化分析了方差如何通过该层传播。

核心假设是：

1. 输入 $x$ 是独立同分布 (i.i.d.) 的，均值为零，方差为 $Var(x)$。
2. 权重 $W$ 是独立同分布 (i.i.d.) 的，均值为零，方差为 $Var(W)$。
3. 权重和输入相互独立。
4. 激活函数 $f$ 关于零对称（例如 $tanh$），并且其在零附近的导数近似为 1 ($f'(0) \approx 1$)。

在这些假设下，层输出 $y_i$（对于单个神经元 $i$）的方差可以与输入方差和权重方差相关联：

$Var(y_i) = n_{in} Var(W_{ij}) Var(x_j)$

这里，$n_{in}$ 表示神经元的输入连接数（“扇入”）。为了使输出方差 $Var(y_i)$ 等于输入方差 $Var(x_j)$，我们需要：

$n_{in} Var(W_{ij}) = 1$
$Var(W_{ij}) = \frac{1}{n_{in}}$

这个条件确保了激活值的方差在前向传播过程中不会显著改变。

同样，考虑到反向传播 (backpropagation)，我们分析梯度的方差。梯度相对于前一层激活值的方差取决于扇出 $n_{out}$（当前层输出连接到的神经元数量）。为了保持梯度方差，条件变为：

$n_{out} Var(W_{ij}) = 1$
$Var(W_{ij}) = \frac{1}{n_{out}}$

Xavier 初始化通过对这些约束进行平均，在这两个条件（前向传播时保持激活方差和反向传播时保持梯度方差）之间寻求一种折衷：

$Var(W_{ij}) = \frac{2}{n_{in} + n_{out}}$

此方差构成了初始化策略的依据。

### Xavier 初始化公式

权重 (weight)通常从均匀分布或正态分布中抽取，并根据此推导出的方差进行缩放。

1. **Xavier 均匀初始化：** 权重从均匀分布 $U[-a, a]$ 中采样，其中 $a$ 的选择使得方差为 $\frac{2}{n_{in} + n_{out}}$。$U[-a, a]$ 的方差是 $\frac{(2a)^2}{12} = \frac{a^2}{3}$。将其设置为目标方差：
   $\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}}$
   $a^2 = \frac{6}{n_{in} + n_{out}}$
   $a = \sqrt{\frac{6}{n_{in} + n_{out}}}$

   因此，权重使用以下方式初始化：
   $W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$
2. **Xavier 正态初始化：** 权重从均值为 0、目标方差为 $\sigma^2$ 的正态分布 $N(0, \sigma^2)$ 中采样：
   $\sigma^2 = \frac{2}{n_{in} + n_{out}}$

   因此，权重使用以下方式初始化：
   $W \sim N\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$

在两种情况下，$n_{in}$ 是层的输入数量（扇入），$n_{out}$ 是层的输出数量（扇出）。偏置 (bias)通常初始化为零。

### 适用性和实现

Xavier 初始化适用于后面跟着在零附近大致线性且对称的激活函数 (activation function)（如 $tanh$ 和逻辑 Sigmoid）的层。假设 $f'(0) \approx 1$ 对于这些函数是比较成立的。然而，它不太适合整流线性单元 (ReLU) 及其变体，这些函数是非对称的，并且对负输入导数为 0。这一局限促成了 Kaiming 初始化的发展，我们接下来会讨论。

在 PyTorch 中，您可以使用 `torch.nn.init` 模块轻松应用 Xavier 初始化。

```python
import torch
import torch.nn as nn

# 定义示例层维度
fan_in, fan_out = 512, 256 # 线性层的示例维度

# 创建一个线性层
linear_layer = nn.Linear(fan_in, fan_out, bias=True)

# --- Xavier 均匀初始化 ---
# 对权重应用 Xavier 均匀初始化
nn.init.xavier_uniform_(linear_layer.weight)

# 通常将偏置初始化为零
if linear_layer.bias is not None:
    nn.init.constant_(linear_layer.bias, 0)

print(f"层: Linear({fan_in}, {fan_out})")
print("初始化: Xavier 均匀")
print(f"权重均值: {linear_layer.weight.mean():.4f}, 标准差: {linear_layer.weight.std():.4f}")
# 检查均匀分布的理论标准差
# Var = (sqrt(6/(fan_in+fan_out)))^2 / 3 = 2 / (fan_in+fan_out)
# Std = sqrt(2 / (fan_in+fan_out))
theoretical_std_uniform = (2.0 / (fan_in + fan_out))**0.5
print(f"理论标准差（基于均匀）: {theoretical_std_uniform:.4f}\n")

# --- Xavier 正态初始化 ---
# 重新创建层或重新初始化
linear_layer_norm = nn.Linear(fan_in, fan_out, bias=True)

# 对权重应用 Xavier 正态初始化
nn.init.xavier_normal_(linear_layer_norm.weight)

# 将偏置初始化为零
if linear_layer_norm.bias is not None:
    nn.init.constant_(linear_layer_norm.bias, 0)

print("初始化: Xavier 正态")
print(
    f"权重均值: {linear_layer_norm.weight.mean():.4f}, "
    f"标准差: {linear_layer_norm.weight.std():.4f}"
)
# 正态分布的理论标准差是 sqrt(2 / (fan_in+fan_out))
theoretical_std_normal = (2.0 / (fan_in + fan_out))**0.5
print(f"理论标准差（正态）: {theoretical_std_normal:.4f}")
```

> 示例输出显示了 Xavier 初始化后的权重 (weight)统计信息。初始化权重的实际标准差应接近从扇入和扇出推导出的理论值。

Xavier 初始化提供了一种合理的方法来设置初始权重，有助于深度网络中更稳定的信号传播，特别是使用对称激活函数的网络。尽管它比朴素的随机初始化有了重要的改进，但其假设并非完全符合所有现代网络架构，尤其是那些高度依赖 ReLU 激活的网络。这引出了我们的下一种方法，Kaiming 初始化，它专门为此类情况而设计。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: JMLR.org; Volume: 9; Pages: 249-256; DOI: [10.5555/2078696.2078720](https://doi.org/10.5555/2078696.2078720)
  介绍Xavier初始化的基础论文，解释了梯度消失/爆炸问题和方差保持原则。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 291-297 (Chapter 8: Optimization for Training Deep Models, Section 8.4: Initializing the Parameters)
  一本内容全面的教材，解释了深度学习概念，包括关于Xavier等权重初始化技术的详细章节。
- [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Development Team (2022)
  Publisher: PyTorch
  PyTorch初始化模块的官方文档，提供了在PyTorch中实现Xavier（Glorot）初始化的实践细节。

---

[上一节](01-%E6%81%B0%E5%BD%93%E5%88%9D%E5%A7%8B%E5%8C%96%E7%9A%84%E9%87%8D%E8%A6%81%E6%80%A7.md) · [下一节](03-Kaiming%20%28%E4%BD%95%29%20%E5%88%9D%E5%A7%8B%E5%8C%96.md)
