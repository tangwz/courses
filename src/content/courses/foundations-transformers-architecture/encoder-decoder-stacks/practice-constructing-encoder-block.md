---
course: "foundations-transformers-architecture"
chapter: "encoder-decoder-stacks"
lesson: "practice-constructing-encoder-block"
sourceId: 2033
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-5-encoder-decoder-stacks/practice-constructing-encoder-block"
title: "动手实践：构建编码器块"
description: "实现一个完整的Transformer编码器块，整合多头注意力、前馈网络、残差连接和层归一化。"
order: 11
plots: []
sourceHash: "7875ba23018f6277af9e42757b20ea2df0e83550bdbe2e198c2ff4c998f34cf0"
sourceCorrections: []
---

Transformer编码器包含多个组件，其中包括用于捕捉语境关系的多头自注意力 (self-attention)、用于非线性转换的位置维度前馈网络、用于梯度流动的残差连接以及用于稳定激活的层归一化 (normalization)。这些组件被组合成一个功能性的`EncoderBlock`。构建`EncoderBlock`呈现了这些子层在一个编码器层中如何协同工作，从而形成堆叠以创建完整编码器的可重复单元。

我们的目标是实现一个标准的Transformer编码器块模块，通常使用像PyTorch或TensorFlow这样的框架。本例中，我们将使用PyTorch语法。

### 编码器块结构回顾

回顾单个编码器块内部的数据流：

1. 输入序列嵌入 (embedding)（结合了位置编码 (positional encoding)）首先通过多头自注意力 (self-attention)机制 (attention mechanism)。
2. 一个残差连接将原始输入添加到注意力子层的输出。
3. 结果随后由层归一化 (normalization)处理。
4. 这个归一化后的输出进入一个位置维度前馈网络。
5. 另一个残差连接将输入*到前馈子层*的输入添加到其输出。
6. 最后，应用第二次层归一化。

Dropout通常应用于自注意力和前馈子层之后，*在*残差连接和归一化*之前*。

> 标准Transformer编码器块内部的数据流（后归一化变体）。虚线表示残差连接。

### 在PyTorch中的实现

让我们为`EncoderBlock`定义一个PyTorch `nn.Module`。我们假设你已经有了`MultiHeadAttention`（如第3章所述）和`PositionwiseFeedForward`（本章前面已讨论）的实现。

```python
import torch
import torch.nn as nn

# 假设 MultiHeadAttention 和 PositionwiseFeedForward 类已在其他地方定义
# class MultiHeadAttention(nn.Module): ...
# class PositionwiseFeedForward(nn.Module): ...

class EncoderBlock(nn.Module):
    """
    实现一个Transformer编码器块。

    此块遵循“Attention Is All You Need”中描述的结构：
    输入 -> 自注意力 -> 加法 & 归一化 -> 前馈网络 -> 加法 & 归一化 -> 输出
    使用后置层归一化（先加法，后归一化）。
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_prob: float = 0.1):
        """
        参数：
            d_model: 输入和输出嵌入的维度（模型维度）。
            num_heads: 注意力头的数量。
            d_ff: 前馈网络的内部维度。
            dropout_prob: 在注意力和FFN后应用的dropout概率。
        """
        super().__init__()
        if d_model % num_heads != 0:
             raise ValueError(f"'d_model' ({d_model}) 必须能被 'num_heads' ({num_heads}) 整除")

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        将输入通过编码器块。

        参数：
            x: 输入张量，形状为 (batch_size, seq_len, d_model)。
            mask: 自注意力层的可选掩码。通常用于填充。
                  形状应能广播到 (batch_size, num_heads, seq_len, seq_len)。

        返回：
            输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        # 计算注意力输出。对于自注意力，Q=K=V=x。
        attn_output = self.self_attn(x, x, x, mask)
        # 对注意力输出应用dropout，然后添加残差连接（输入 x），
        # 最后应用层归一化。
        x = self.norm1(x + self.dropout(attn_output))

        # 存储第一个子层（注意力 + 加法&归一化）的输出
        # 这将是第二个残差连接的输入。
        sublayer1_output = x

        # 2. 前馈网络 + 残差连接 + 层归一化
        # 计算前馈网络输出
        ff_output = self.feed_forward(sublayer1_output)
        # 对FFN输出应用dropout，然后添加残差连接
        # （使用第一个子层的输出），并应用层归一化。
        x = self.norm2(sublayer1_output + self.dropout(ff_output))

        return x
```

### 理解代码

- **初始化 (`__init__`)**：我们实例化了必要的子模块：`MultiHeadAttention`、`PositionwiseFeedForward`、两个`LayerNorm`层和一个`Dropout`层。`LayerNorm`层对特征维度（`d_model`）进行归一化 (normalization)。为了多头注意力 (multi-head attention)机制 (attention mechanism)，代码进行了一个检查，确保`d_model`能够被`num_heads`整除。
- **前向传播 (`forward`)**：
  - 第一部分处理多头自注意力 (self-attention)子层。输入`x`作为查询、键和值。输出`attn_output`使用dropout进行正则化 (regularization)。
  - 第一个残差连接将原始输入`x`添加到（经过dropout修改的）注意力输出。这个和随后通过第一个层归一化（`self.norm1`）。结果更新变量`x`。
  - 第二部分处理位置维度前馈子层。第一个归一化（`x`）的输出通过前馈网络（`self.feed_forward`）。
  - 它的输出（`ff_output`）使用dropout进行正则化。
  - 第二个残差连接将*到前馈层*的输入（即`self.norm1`的输出，在优化后的代码中临时存储为`sublayer1_output`）添加到（经过dropout修改的）前馈输出。
  - 这个和通过第二个层归一化（`self.norm2`）。
  - 最终的张量，经过两个子层以及残差和归一化处理后，被返回。

### 实例化与使用示例

以下是如何创建和使用`EncoderBlock`，包括子模块的占位符定义，以使示例可运行：

```python
# 示例参数
batch_size = 4
seq_len = 50
d_model = 512 # 模型维度
num_heads = 8   # 注意力头数量
d_ff = 2048   # 前馈网络内部维度
dropout_prob = 0.1

# 创建虚拟输入张量 (batch_size, seq_len, d_model)
dummy_input = torch.rand(batch_size, seq_len, d_model)

# --- 假设 MultiHeadAttention 和 PositionwiseFeedForward 已定义 ---
# 这部分只是为了让示例能够独立运行。
# 请替换为你在前面章节/部分中的实际实现。
class MultiHeadAttention(nn.Module):
    # 占位符实现
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
             raise ValueError("d_model 必须能被 num_heads 整除")
        self.head_dim = d_model // num_heads
        # 通常，这些是 Q, K, V 的投影
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model) # 最终输出投影
    def forward(self, query, key, value, mask=None):
        # 简化占位符：投影输入，应用线性输出层
        # 在实际实现中，这会执行多头缩放点积注意力，并拼接结果。
        batch_size = query.shape[0]
        # 仅模拟最终输出投影以保持形状一致性
        # 为简化起见，此处忽略实际注意力计算
        projected_q = self.fc_q(query) # 示例投影
        return self.fc_out(projected_q) # 返回形状 (batch_size, seq_len, d_model)

class PositionwiseFeedForward(nn.Module):
    # 标准实现
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # Dropout 通常也应用于此处
        self.linear2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x))))
# ------------------------------------------------------------------------

# 使用（占位符）模块实例化编码器块
encoder_block = EncoderBlock(d_model, num_heads, d_ff, dropout_prob)

# 将输入通过该块
output = encoder_block(dummy_input)

print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output.shape}")

# 验证形状
# 预期输出：
# 输入形状: torch.Size([4, 50, 512])
# 输出形状: torch.Size([4, 50, 512])
```

输出张量保持形状`(batch_size, seq_len, d_model)`，这非常重要。这使得一个`EncoderBlock`的输出可以直接作为输入传入堆叠中的下一个`EncoderBlock`，从而能够构建深度编码器模型。

### 配置与变体

- **超参数 (parameter) (hyperparameter)**：`d_model`、`num_heads`、`d_ff`和`dropout_prob`的选择显著影响模型的容量、计算成本和泛化能力。原始的Transformer使用了`d_model=512`、`num_heads=8`、`d_ff=2048`和`dropout_prob=0.1`。更大的模型通常使用更大的值。
- **归一化 (normalization)位置（前置归一化）**：此实现使用后置归一化（层归一化在残差加法*之后*）。另一种方法是前置归一化，它在自注意力 (self-attention)和前馈子层*之前*应用层归一化，残差连接在之后添加。前置归一化通常带来更稳定的训练，尤其对于更深的模型，并且需要修改`forward`方法的结构。我们将在第6章讨论前置归一化与后置归一化的权衡。

这个实践示例提供了一个编码器块的具体实现，结合了前面讨论的理论组件。通过理解如何构建这个基础单元，你将能够构建整个编码器堆栈，并理解Transformer架构内发生的数据转换。

## 参考资料

- [torch.nn - PyTorch documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  PyTorch官方文档，提供构建神经网络的API参考和使用指南，包括nn.Module、nn.LayerNorm和nn.Dropout等模块，这些模块直接用于本节的实现。
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu (2020)
  Journal: ICML 2020; DOI: [10.48550/arXiv.2002.04745](https://doi.org/10.48550/arXiv.2002.04745)
  一篇研究论文，分析了Transformer架构中不同层归一化位置（Pre-LN与Post-LN）对训练稳定性和性能的影响，本节明确提到了该主题。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本全面的自然语言处理教材草案，详细解释了Transformer架构及其组成部分，适合更深入的学习。
