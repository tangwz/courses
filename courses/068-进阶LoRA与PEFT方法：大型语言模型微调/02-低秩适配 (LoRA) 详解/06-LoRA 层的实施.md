# LoRA 层的实施

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/implementing-lora-layers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

LoRA (低秩适应) 的理论依据是适应发生在低维子空间中的假设，其数学公式为 $\Delta W \approx BA$。在神经网络 (neural network)层中实施这项技术，其主要想法不是直接*改变*原始权重 (weight) $W_0$，而是通过矩阵 $A$ 和 $B$ 计算出的低秩更新来增加层的计算量。

### 修改前向传播

线性（全连接）层的标准前向传播通常表示为：
$y = x W_0^T + b$
其中 $x$ 是输入张量，$W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ 是权重 (weight)矩阵，$b$ 是可选的偏置 (bias)向量 (vector)，$y$ 是输出张量。这些维度假设输入批次 $x \in \mathbb{R}^{N \times d_{in}}$ 产生输出 $y \in \mathbb{R}^{N \times d_{out}}$。

使用 LoRA 时，我们保持 $W_0$ 和 $b$ 不变。适应项 $\Delta W = BA$（其中 $B \in \mathbb{R}^{d_{out} \times r}$，$A \in \mathbb{R}^{r \times d_{in}}$，$r$ 是秩）被添加到计算中。修改后的前向传播变为：
$y = x W_0^T + x (\Delta W)^T + b$
代入 $\Delta W = BA$：
$y = x W_0^T + x (BA)^T + b$
$y = x W_0^T + x A^T B^T + b$

LoRA 论文引入了一个缩放因子 $\alpha/r$，应用于低秩更新。这种缩放有助于控制适应项相对于原始权重的量级，尤其是在改变秩 $r$ 时。最终的 LoRA 前向传播为：
$y = x W_0^T + (x A^T B^T) \frac{\alpha}{r} + b$

这种公式表示很重要：

1. **基础权重不变**: $W_0$ 和 $b$ 在微调 (fine-tuning)期间不进行更新。它们的梯度不被计算，与完全微调相比，这节省了大量的内存和计算。
2. **可训练的适配器**: 只有 $A$ 和 $B$ 的参数 (parameter)被训练。由于 $r \ll \min(d_{in}, d_{out})$，可训练参数的数量大大减少。
3. **无推理 (inference)延迟（合并后）**: 尽管在前向传播在训练期间涉及额外的计算路径，但训练后学习到的 $\Delta W = BA$ 可以*合并*回 $W_0$ ($W_{adapted} = W_0 + BA$) 以进行部署，这与原始模型相比不会增加额外的延迟。合并将在第 4 章中进行讨论。

### 在 PyTorch 中实现 LoRA 层

我们来概述一下如何在 PyTorch 中实现 `LoRALinear` 层。这通常涉及创建一个自定义模块，该模块封装或替换现有 `nn.Linear` 层。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALinear(nn.Module):
    """ 将标准 nn.Linear 层替换为 LoRA 适应版本。 """
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        lora_dropout_p: float = 0.0,
        ):
        super().__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha

        # 将原始权重和偏置注册为不可训练的参数
        self.weight = nn.Parameter(original_layer.weight.detach().clone())
        self.weight.requires_grad = False

        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.detach().clone())
            self.bias.requires_grad = False
        else:
            # 使用 register_parameter 确保 'bias' 属性存在，即使它为 None
            self.register_parameter('bias', None)

        # 创建并初始化 LoRA 矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.Tensor(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.Tensor(self.out_features, rank))

        # LoRA 路径的可选 dropout 层
        if lora_dropout_p > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout_p)
        else:
            self.lora_dropout = nn.Identity() # 作为一个直通层

        # 缩放因子
        if rank > 0:
            self.scaling = self.alpha / self.rank
        else:
            self.scaling = 1.0 # 如果秩为 0，避免除以零

        # 初始化 LoRA 参数
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        """ 初始化 LoRA 矩阵 A 和 B。 """
        if self.rank > 0:
            # 使用 Kaiming uniform 初始化 A，以获得更好的梯度流动
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # 将 B 初始化为零，以便初始适应项为零
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 执行修改后的前向传播。 """
        # 计算原始（不变的）线性变换
        # 使用 F.linear 可以避免在混合张量时出现设备放置问题
        result = F.linear(x, self.weight, self.bias)

        # 如果 rank > 0，计算 LoRA 调整
        if self.rank > 0:
            # 在 LoRA 矩阵之前对输入 x 应用 dropout
            x_lora = self.lora_dropout(x)

            # 计算 x @ A^T
            # 输入 x_lora (N, d_in), 权重 lora_A (r, d_in) -> 输出 (N, r)
            after_A = F.linear(x_lora, self.lora_A)

            # 计算 (x @ A^T) @ B^T
            # 输入 after_A (N, r), 权重 lora_B (d_out, r) -> 输出 (N, d_out)
            lora_adjustment = F.linear(after_A, self.lora_B)

            # 将缩放后的 LoRA 调整添加到原始结果中
            result += lora_adjustment * self.scaling

        return result

    def train(self, mode: bool = True):
        """ 确保原始权重在训练期间保持不变。 """
        super().train(mode)
        # 在模式更改后显式设置 requires_grad 为 False
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # 确保 LoRA 参数可训练（它们默认是可训练的）
        # self.lora_A.requires_grad = True
        # self.lora_B.requires_grad = True

    def extra_repr(self) -> str:
        """ 向模块表示添加 LoRA 特定信息。 """
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}')
```

该实现的重要方面：

- **初始化**: `lora_A` 通常使用 Kaiming uniform 等标准方案进行初始化，而 `lora_B` 则初始化为零。将 `lora_B` 初始化为零可以确保初始 $\Delta W = BA$ 为零，这意味着适应后的模型在训练开始前与预训练 (pre-training)模型完全相同。
- **冻结**: 原始的 `weight` 和 `bias` 参数 (parameter)的 `requires_grad` 设置为 `False`。建议在 `train()` 方法的重写中重新确认这一点，以防止在模型模式切换不当时代替意外解冻。
- **前向计算**: 前向传播明确地分别计算原始路径和 LoRA 路径，然后将它们相加。使用 `torch.nn.functional.linear` (别名为 `F.linear`) 是一种常见做法，用于应用线性变换，而无需在前向传播本身中包含完整的 `nn.Linear` 模块开销。
- **Dropout**: Dropout 层可以选择性地专门应用于 LoRA 路径，通常在矩阵 A 乘法之前，正如某些实现中所建议的。这有助于正则化 (regularization)适应。
- **缩放**: 缩放因子 $\alpha/r$ 在添加 LoRA 调整之前应用。

### LoRA 前向路径的可视化

LoRA 层内部的计算流可以按如下方式可视化：

> 一张图表，说明了 LoRA 的前向传播。原始路径使用不变的权重 (weight)（$W_0$，可选 $b$），而并行的 LoRA 路径引入了可训练的低秩矩阵（$A$， $B$），其输出经过缩放后添加到原始结果中。

### 将 LoRA 应用于线性层

虽然 `nn.Linear` 层是 Transformer 中 LoRA 的最常见目标（特别是在自注意力 (self-attention)机制 (attention mechanism)和前馈网络块中），但低秩适应的底层原理也可以应用于其他层类型：

- **卷积层 (`nn.Conv2d`)**: 卷积层中的权重 (weight)张量也具有低秩适应的可能。$\Delta W$ 将是一个 4D 张量，可以应用将其分解为一系列较小卷积或使用低秩张量分解（如 Tucker 或 CP）的技术。相关实现存在，但不如线性层那样标准化。
- **嵌入 (embedding)层 (`nn.Embedding`)**: 嵌入层的权重矩阵本质上是一个查找表（$V \times d_{model}$，其中 $V$ 是词汇量大小）。$\Delta W$ 是一个相同形状的矩阵，使其可以直接适用于 LoRA（$BA$，其中 $B \in \mathbb{R}^{V \times r}, A \in \mathbb{R}^{r \times d_{model}}$），类似于线性层。

然而，在实际中，将 LoRA 主要应用于 `Linear` 层，尤其是注意力机制（查询、键、值、输出投影）和前馈网络中的那些层，通常能为大型语言模型提供参数 (parameter)效率和性能之间的良好平衡。

### 参数 (parameter)效率计算

我们来量化 (quantization)一下参数节省情况。考虑一个具有 $d_{in}$ 个输入特征和 $d_{out}$ 个输出特征的标准 `nn.Linear` 层。

- $W_0$ 中的原始参数：$d_{in} \times d_{out}$
- 完整 $\Delta W$ 中的参数：$d_{in} \times d_{out}$
- LoRA 矩阵 $A$ ($r \times d_{in}$) 和 $B$ ($d_{out} \times r$) 中的参数：$r \times d_{in} + d_{out} \times r = r(d_{in} + d_{out})$

示例：对于大型模型中 $d_{in} = d_{out} = 4096$ 的层。

- 完整 $\Delta W$ 参数：$4096 \times 4096 = 16,777,216$
- LoRA 参数（秩 $r=8$）：$8 \times (4096 + 4096) = 8 \times 8192 = 65,536$

这表示该单一层的可训练参数**减少了约 256 倍**（$16,777,216 / 65,536 \approx 256$）。当应用于多个层时，累积节省的参数量非常可观，显著减少了训练期间优化器状态和梯度的内存占用。

有了对如何实施单个 LoRA 层的这种理解，下一步就是将这些修改后的层集成到完整的 Transformer 架构中，我们将在本章后面讨论。选择合适的秩 $r$ 和缩放因子 $\alpha$ 等实际因素对于成功应用 LoRA 也非常重要，并将很快进行讨论。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  这篇论文介绍了低秩适配（LoRA）方法，详细阐述了其理论基础、数学公式和通过实验验证的用于高效微调大型语言模型的结果。
- [\`torch.nn.Module\`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Authors (2024)
  Publisher: PyTorch Foundation
  `torch.nn.Module` 的官方文档提供了在 PyTorch 中构建神经网络层和模型的核心信息，包括参数管理和前向传播定义。
- [PEFT: Parameter-Efficient Fine-tuning (Official Documentation)](https://huggingface.co/docs/peft/en/index) — Hugging Face Team (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档提供了将 LoRA 等参数高效微调方法应用于预训练模型的实用指南和示例。

---

[上一节](05-%E7%BC%A9%E6%94%BE%E5%8F%82%E6%95%B0%20Alpha.md) · [下一节](07-%E5%B0%86%20LoRA%20%E8%9E%8D%E5%85%A5%20Transformer%20%E6%9E%B6%E6%9E%84.md)
