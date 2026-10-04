---
course: "intro-normalizing-flows"
chapter: "autoregressive-flows"
lesson: "building-autoregressive-models-practice"
sourceId: 8267
sourceUrl: "https://apxml.com/zh/courses/intro-normalizing-flows/chapter-3-autoregressive-flows/building-autoregressive-models-practice"
title: "构建自回归模型实践"
description: "在 PyTorch 中构建掩码线性层，作为自回归流的基础。"
order: 5
plots: ["plots/8267-0.json"]
sourceHash: "678b061284a11edb815ffd6e68b1d714d294c7bbc09422f40af2d16c24a15756"
sourceCorrections: []
---

自回归 (autoregressive)模型通过因子分解来确保概率分布的有效性和行列式计算的高效性。在 PyTorch 中，这类架构核心组件的实现依赖于掩码线性层。分布估计掩码自动编码器 (MADE) 的这一基础组件通过对标准全连接层的权重 (weight)应用二值掩码，直接通过矩阵乘法而非顺序循环来强制执行自回归特性。

为了实现这一点，我们需要一个由 0 和 1 组成的矩阵。1 表示允许连接，而 0 表示阻止连接。对于处理变量序列 $x_1, x_2, ..., x_D$ 的自回归模型，预测 $x_i$ 的输出只能依赖于之前的输入 $x_1$ 到 $x_{i-1}$。这种严格的依赖关系自然地形成了一种下三角矩阵结构。



![自回归二值掩码](plots/8267-0.json)



> 输入层严格下三角二值掩码的热力图。蓝色方块代表激活的权重，灰色方块代表被遮盖的权重。

我们将编写一个继承自 `torch.nn.Linear` 的自定义 PyTorch 模块。实现过程需要将二值掩码注册为缓冲区 (buffer)。在 PyTorch 中，缓冲区是属于模块状态的一部分、但不会被视为可训练参数 (parameter)的张量。这确保了掩码能与模型权重一起移动到相应的设备（如 GPU）上，但在训练过程中不会被优化器修改。

线性变换计算输出 $y$ 的公式为：

$y = x W^T_{	ext{已掩码}} + b$

其中，掩码权重矩阵是通过对可学习权重 $W$ 和二值掩码 $M$ 进行哈达玛积（逐元素相乘）计算得出的：

$W_{	ext{已掩码}} = W \odot M$

以下是掩码线性层的实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        # 初始化标准线性层属性
        super().__init__(in_features, out_features, bias)

        # 将掩码注册为持久缓冲区
        self.register_buffer('mask', mask)

    def forward(self, x):
        # 权重与掩码进行逐元素相乘
        masked_weight = self.weight * self.mask

        # 使用掩码权重执行线性变换
        return F.linear(x, masked_weight, self.bias)
```

接下来，我们需要一个函数来生成掩码张量本身。对于网络的第一层（输入层），掩码必须是严格下三角的。对角线必须设为 0，因为输出变量不能依赖于其对应的输入变量。对于后续的隐藏层，掩码可以是标准的下三角矩阵，允许具有相同自回归度的节点进行连接。

```python
def create_autoregressive_mask(in_features, out_features, is_input_layer=False):
    """
    为自回归模型生成下三角二值掩码。
    """
    if is_input_layer:
        # 严格下三角（对角线为 0）
        mask = torch.tril(torch.ones(out_features, in_features), diagonal=-1)
    else:
        # 标准下三角（对角线为 1）
        mask = torch.tril(torch.ones(out_features, in_features), diagonal=0)

    return mask
```

让我们验证一下自定义层的行为。我们将定义一个较小的维度，生成输入掩码，并检查生成的层权重。通过将原始随机初始化的权重与掩码相乘，我们可以确认自回归约束已经生效。

```python
# 定义输入和输出维度
D = 5

# 为输入层创建掩码
mask = create_autoregressive_mask(in_features=D, out_features=D, is_input_layer=True)

# 实例化掩码线性层
masked_layer = MaskedLinear(in_features=D, out_features=D, mask=mask)

# 打印掩码权重，确认上三角和对角线已归零
print("有效权重:\n", masked_layer.weight * masked_layer.mask)
```

完成了 `MaskedLinear` 组件后，你就拥有了构建完整 MADE 架构的基础。通过堆叠多个掩码层并在其中加入 ReLU 等非线性激活函数 (activation function)，你就可以对极其复杂的联合概率分布进行建模。在实践中，完整 MADE 模型中的隐藏层需要为每个神经元分配特定的连接索引，以确保在多次变换中仍能保持严格的数学顺序。这种堆叠架构是掩码自回归流 (MAF) 和逆自回归流 (IAF) 模型底层的密度估计器。
