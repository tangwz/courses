# 将 LoRA 融入 Transformer 架构

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-2-lora-in-depth/lora-integration-transformers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将低秩适应 (LoRA) 集成到 Transformer 模型中，需要在其复杂架构内理解其实际应用。LoRA 使用低秩矩阵 $B$ 和 $A$ 近似权重 (weight)更新 $\Delta W$。主要的考量不是 LoRA 如何独立运作，而是具体如何以及在哪里将其有效地融入现有的 Transformer 结构。

Transformer 由堆叠的模块构成，通常包含多头自注意力 (self-attention) (MHA) 机制和位置前馈网络 (FFN)。MHA 和 FFN 都非常依赖线性变换，这些变换由大型权重矩阵表示。它们是 LoRA 适应的主要目标。

### LoRA 适应的特定层选择

LoRA 允许采用选择性的方法，而不是修改模型中的每个权重 (weight)矩阵。最常见的方法是调整涉及以下内容的权重矩阵：

1. **自注意力 (self-attention)机制 (attention mechanism)：** 用于计算查询 ($W_q$)、键 ($W_k$) 和值 ($W_v$) 矩阵的线性投影是主要的选择。通常，在组合注意力头之后的输出投影矩阵 ($W_o$) 也会进行适应。
2. **前馈网络 (FFN)：** FFN 通常由两个线性层组成，中间有一个激活函数 (activation function)。这两个线性层都可以作为 LoRA 适应的目标。

这样做的原因是，这些层捕获了微调 (fine-tuning)过程中所需的许多特定任务知识。我们假设，通过将 LoRA 适应集中于此，我们可以在仅修改总参数 (parameter)一小部分的情况下，实现与完整微调相近的性能。

### 将 LoRA 整合到线性层中

回想一下，线性层的标准前向传播是 $h = Wx + b$。应用 LoRA 时，原始权重 (weight)矩阵 $W_0$ 被冻结。适应是通过在前向传播期间添加低秩更新来实现的。修改后的输出 $h_{LoRA}$ 计算方式如下：


$$
h_{LoRA} = W_0 x + \Delta W x = W_0 x + B A x
$$


通常，一个缩放因子 $s$（通常是 $\alpha/r$，这里 $\alpha$ 是 LoRA 缩放超参数 (parameter) (hyperparameter)，$r$ 是秩）会应用于 LoRA 更新：


$$
h_{LoRA} = W_0 x + s \cdot B A x
$$


这里，$W_0 \in \mathbb{R}^{d \times k}$ 保持冻结，而 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 是可训练的低秩矩阵。偏置 (bias)项 $b$（如果存在）通常仍然会被训练或根据具体实现进行处理。重要的是，在反向传播 (backpropagation)期间只有 $A$ 和 $B$ 更新，这极大地减少了可训练参数的数量。

### Transformer 模块中 LoRA 整合的可视化

此图显示了 LoRA 适配器 ($BA$) 通常在标准 Transformer 编码器模块中的插入位置。

> 此图显示了一个典型的 Transformer 模块。LoRA 适配器（蓝色平行四边形）与多头注意力 (multi-head attention) (Q, K, V, O) 和前馈网络 (FFN) 组件中的原始线性层（黄色方框）并行添加。原始权重 (weight)被冻结，且只有 LoRA 权重被训练。

### 实现考量

实现 LoRA 整合涉及用 LoRA 增强版本替换标准线性层。Hugging Face 的 `peft`（参数 (parameter)高效微调 (fine-tuning)）等库提供了高级抽象来自动化此过程。

您可能会定义一个 `LoRALinear` 层来包装一个标准 `Linear` 层。

```python
# 示例（简化版）
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer # 原始的、被冻结的线性层
        self.rank = rank
        self.alpha = alpha

        # 冻结原始层
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
             self.linear.bias.requires_grad = False # 通常偏置项仍然被训练，取决于配置

        # 创建 LoRA 矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(rank, linear_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear_layer.out_features, rank))

        # 初始化 LoRA 矩阵（例如，A 使用 Kaiming 均匀分布，B 使用零）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / self.rank

    def forward(self, x):
        # 原始的前向传播（冻结权重）
        result = self.linear(x)

        # LoRA 适应
        lora_update = (self.lora_B @ self.lora_A) * self.scaling
        result += torch.matmul(x, lora_update.T) # 应用更新: x @ (B*A)^T = x @ A^T @ B^T

        return result

# 用法：
# original_layer = nn.Linear(in_features=512, out_features=512)
# lora_layer = LoRALinear(original_layer, rank=8, alpha=16)
# 现在在 Transformer 模型中使用 lora_layer 替换 original_layer
```

这个例子说明了核心机制：冻结原始层，并将可训练低秩矩阵 ($B$ 和 $A$) 的缩放积添加到输出中。

### 配置选项

在整合 LoRA 时，您需要做出几个选择：

- **目标模块：** 决定 Transformer 的哪些部分进行适应。常见的选择包括仅注意力权重 (weight) ($W_q, W_k, W_v, W_o$)、仅 FFN 权重，或两者都选。更广泛地应用 LoRA 会增加可训练参数 (parameter)，但可能会提高性能。
- **秩 ($r$)：** 一个主要超参数 (hyperparameter)。更高的秩允许更强的适应表现力，但会增加可训练参数的数量 ($d \times r + r \times k$)。典型值范围为 4 到 64，通常从小值（例如 8 或 16）开始并进行试验。
- **Alpha ($\alpha$)：** 缩放因子。它控制 LoRA 适应相对于原始权重的幅度。通常设置为等于秩，但可以独立调整。

通常需要进行实验，为特定任务和模型找到最佳配置。一个常见的起点是仅使用低秩（例如 $r=8$）适应查询和值矩阵 ($W_q, W_v$)。

通过有策略地将这些低秩适应模块引入 Transformer 架构，LoRA 实现了高效的微调 (fine-tuning)，仅修改总参数的一小部分，同时保留了冻结权重中编码的预训练 (pre-training)知识。接下来的章节将考察实际实现细节，包括秩选择和缩放参数 $\alpha$ 的作用。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  描述Transformer架构的奠基性论文，LoRA在此架构上进行适应。
- [LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/peft/en/conceptual_guides/lora) — Hugging Face (2024)
  Hugging Face `peft` 库中关于 LoRA 集成和使用的官方指南。

---

[上一节](06-LoRA%20%E5%B1%82%E7%9A%84%E5%AE%9E%E6%96%BD.md) · [下一节](08-%E5%AE%9E%E9%99%85%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%BA%94%E7%94%A8%E5%9F%BA%E7%A1%80LoRA.md)
