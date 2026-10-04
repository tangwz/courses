---
course: "intro-normalizing-flows"
chapter: "coupling-architectures"
lesson: "implementing-coupling-layer-practice"
sourceId: 8272
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-4-coupling-architectures/implementing-coupling-layer-practice"
title: "实现耦合层实践"
description: "编写代码构建功能完备的仿射耦合层，并将其集成到神经网络中。"
order: 5
plots: []
sourceHash: "f91c3265a643447762009a0fbbe5113708ef9443c1202d062e03e237cfe27799"
sourceCorrections: []
---

从头开始构建仿射耦合层（Affine Coupling Layer）可以将数学原理转化为功能性的 PyTorch 代码。此实现展示了输入划分如何同时实现高性能采样和密度估计。

### 仿射耦合层设计

仿射耦合层通过将输入张量分成两半来运行。前半部分保持完全不变，并作为神经网络 (neural network)的输入。该网络计算用于转换输入后半部分的缩放和平移参数 (parameter)。

前向变换依赖于以下数学运算：

$y_1 = x_1$

$y_2 = x_2 \odot \exp(s(x_1)) + t(x_1)$

这里，$x$ 表示输入数据，$y$ 表示输出，函数 $s$ 和 $t$ 对应缩放和平移神经网络。符号 $\odot$ 表示逐元素相乘。

由于 $s$ 和 $t$ 网络仅处理 $x_1$，因此在前向和逆向过程中都可以轻松计算缩放因子 $\exp(s(x_1))$。逆向过程与前向过程密切对应：

$x_1 = y_1$

$x_2 = (y_2 - t(y_1)) \odot \exp(-s(y_1))$

下图展示了数据通过仿射耦合层的流程。

> 仿射耦合层的数据流架构，展示了输入拆分和参数化过程。

### 在 PyTorch 中实现网络

我们将构建一个继承自 `torch.nn.Module` 的 `AffineCouplingLayer` 类。对于 $s$ 和 $t$ 函数，我们可以使用一个单一的多层感知机，它输出一个维度为所需维度两倍的张量。然后我们将这个输出张量拆分为缩放和平移参数 (parameter)。

为了确保训练期间的数值稳定性，通常会在对缩放参数 $s$ 进行指数运算之前对其应用双曲正切（tanh）激活函数 (activation function)。这可以防止指数函数产生极大的数值，从而导致梯度爆炸。

```python
import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        # 网络处理一半的输入维度
        self.half_dim = input_dim // 2

        # 一个简单的多层感知机来计算 s 和 t
        self.st_net = nn.Sequential(
            nn.Linear(self.half_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 输出维度是 half_dim 的两倍，以产生 s 和 t
            nn.Linear(hidden_dim, self.half_dim * 2)
        )

    def forward(self, x):
        # 沿特征维度将输入对半拆分
        x1, x2 = x.chunk(2, dim=-1)

        # 从 x1 计算缩放和平移参数
        st_params = self.st_net(x1)
        s, t = st_params.chunk(2, dim=-1)

        # 限制缩放参数以保证数值稳定性
        s = torch.tanh(s)

        # 对 x2 应用仿射变换
        y1 = x1
        y2 = x2 * torch.exp(s) + t

        # 重新组合输出组件
        y = torch.cat([y1, y2], dim=-1)

        # 计算雅可比行列式的对数 (log-determinant)
        # 它是受限缩放参数的总和
        log_det_jacobian = s.sum(dim=-1)

        return y, log_det_jacobian

    def inverse(self, y):
        # 将输出对半拆分
        y1, y2 = y.chunk(2, dim=-1)

        # 从 y1 重新计算缩放和平移参数
        st_params = self.st_net(y1)
        s, t = st_params.chunk(2, dim=-1)

        # 对 s 应用完全相同的限制
        s = torch.tanh(s)

        # 逆转仿射变换
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        # 重新组合形成原始输入
        x = torch.cat([x1, x2], dim=-1)

        return x
```

注意在 `chunk` 和 `cat` 操作中使用了 `dim=-1`。这指定了拆分和拼接应始终沿张量的最后一个维度进行。这种设计允许该层交替接收单个数据点或批量数据。

### 验证精确可逆性

在构建归一化 (normalization)流时，逆方法中的微小错误或不匹配的激活函数 (activation function)都会破坏模型准确估计概率的能力。你应该始终测试你的可逆层，以确保数据通过前向方法后再通过逆方法能够完全恢复原始输入。

我们可以实例化新层并运行一个快速的验证测试。

```python
# 为 8 维输入实例化耦合层
layer = AffineCouplingLayer(input_dim=8, hidden_dim=32)

# 创建一批随机虚拟数据
original_x = torch.randn(4, 8)

# 通过前向方法传递数据
y, log_det = layer(original_x)

# 使用逆方法恢复数据
reconstructed_x = layer.inverse(y)

# 衡量原始数据与重构数据之间的最大绝对误差
max_error = torch.max(torch.abs(original_x - reconstructed_x))
print(f"最大重构误差: {max_error.item():.6e}")
```

如果实现正确，最大重构误差应该是一个接近于零的极小数字。由于计算机硬件中浮点精度的固有局限，误差不会完全为零，但 `1e-6` 或更小数量级的误差可以确认该层运行符合预期。
