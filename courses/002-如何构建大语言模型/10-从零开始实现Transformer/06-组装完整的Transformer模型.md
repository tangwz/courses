# 组装完整的Transformer模型

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-10-implementing-transformer-from-scratch/assembling-full-transformer-model)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了构建完整的 Transformer 模型，`EncoderLayer` 和 `DecoderLayer` 作为独立的组成部分被使用。构建过程包括堆叠这些层，添加必需的嵌入 (embedding)层，加入位置编码 (positional encoding)，以及定义最终的输出投影。

回顾一下，Transformer 架构采用编码器-解码器结构。编码器处理输入序列并生成富含上下文 (context)信息的表示（通常称为 `memory`）。解码器随后将此 `memory` 与目标序列（训练期间）或之前生成的标记 (token)（推理 (inference)期间）一同使用，以生成输出序列。

我们来定义一个用于完整 Transformer 的 PyTorch `nn.Module`。

```python
import torch
import torch.nn as nn
import math

# 假设以下模块已在之前的章节/文件中定义：
# from .encoder import EncoderLayer
# from .decoder import DecoderLayer
# from .attention import MultiHeadAttention
# from .feed_forward import PositionwiseFeedForward
# from .embeddings import PositionalEncoding, Embeddings # 假设是组合的嵌入 + 位置编码

# 所需类的占位符定义（请替换为实际导入）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # 示例占位符
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential( # 示例占位符
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # 占位符的简化前向传播
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        # 示例占位符
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # 示例占位符
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.feed_forward = nn.Sequential( # 示例占位符
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 占位符的简化前向传播
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2, _ = self.multihead_attn(
            tgt, memory, memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.feed_forward(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # 使用 register_buffer 以便 pe 不作为模型参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数：
            x: 张量，形状为 [batch_size, seq_len, embedding_dim]
        """
        # 将位置编码调整为批处理格式 [batch_size, seq_len, embedding_dim]
        # 原始 pe 形状为 [max_len, 1, embedding_dim]。我们需要 [1, seq_len, embedding_dim] 或兼容形状。
        # 对当前序列长度切片 pe 并转置。
        # 形状变为 [1, seq_len, embedding_dim]
        pe_for_seq = self.pe[:x.size(1), :].permute(1, 0, 2)
        x = x + pe_for_seq
        return self.dropout(x)

# --- 主 Transformer 类 ---

class TransformerModel(nn.Module):
    """
    完整的 Transformer 模型实现。
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int, nhead: int, num_encoder_layers: int,
                 num_decoder_layers: int, dim_feedforward: int,
                 dropout: float = 0.1, max_len: int = 5000):
        """
        参数：
            src_vocab_size: 源词汇表大小。
            tgt_vocab_size: 目标词汇表大小。
            d_model: 嵌入和模型层的维度。
            nhead: 注意力头数量。
            num_encoder_layers: 堆叠编码器层的数量。
            num_decoder_layers: 堆叠解码器层的数量。
            dim_feedforward: 前馈网络隐藏层的维度。
            dropout: Dropout 比率。
            max_len: 位置编码的最大序列长度。
        """
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        # 对层列表使用 nn.ModuleList
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 最终的线性层，用于将解码器输出投影到词汇表大小
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # 可选：目标嵌入和最终线性层之间的权重绑定
        # self.tgt_tok_emb.weight = self.generator.weight # 需要相同维度

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化 Transformer 模型中的参数。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将源序列通过编码器堆栈。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            src_mask: 源序列注意力掩码（src_seq_len, src_seq_len）。
                      如果需要，可防止注意力集中于未来位置（编码器通常不需要）。
            src_key_padding_mask: 源序列中填充标记的掩码（batch_size, src_seq_len）。

        返回：
            编码器输出张量（batch_size, src_seq_len, d_model）。
        """
        # 嵌入标记并添加位置编码
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        # 通过每个编码器层
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_mask=src_mask,
                           src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor = None,
               memory_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None,
               memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        将目标序列和编码器 memory 通过解码器堆栈。

        参数：
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            memory: 编码器输出张量（batch_size, src_seq_len, d_model）。
            tgt_mask: 目标序列自注意力掩码（tgt_seq_len, tgt_seq_len）。
                      防止注意力集中于未来位置。
            memory_mask: 编码器-解码器注意力掩码（tgt_seq_len, src_seq_len）。
                         通常不需要，除非需要特定的交叉注意力掩码。
            tgt_key_padding_mask: 目标序列中填充标记的掩码（batch_size, tgt_seq_len）。
            memory_key_padding_mask: 源序列中填充标记的掩码，用于编码器-解码器注意力（batch_size, src_seq_len）。

        返回：
            解码器输出张量（batch_size, tgt_seq_len, d_model）。
        """
        # 嵌入标记并添加位置编码
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # 通过每个解码器层
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None,
                memory_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Transformer 模型的完整前向传播。

        参数：
            src: 源序列张量（batch_size, src_seq_len）。
            tgt: 目标序列张量（batch_size, tgt_seq_len）。
            src_mask: 源序列注意力掩码。
            tgt_mask: 目标序列自注意力掩码。
            memory_mask: 编码器-解码器注意力掩码。
            src_key_padding_mask: 源序列的填充掩码。
            tgt_key_padding_mask: 目标序列的填充掩码。
            memory_key_padding_mask: 用于交叉注意力的源序列填充掩码。

        返回：
            输出对数张量（batch_size, tgt_seq_len, tgt_vocab_size）。
        """
        memory = self.encode(src, src_mask, src_key_padding_mask)
        decoder_output = self.decode(tgt, memory, tgt_mask, memory_mask,
                                     tgt_key_padding_mask,
                                     memory_key_padding_mask)
        logits = self.generator(decoder_output)
        return logits
```

### 集成各部分

1. **初始化 (`__init__`)**:

   - 我们为源词汇表 (vocabulary)和目标词汇表初始化独立的嵌入 (embedding)层（`nn.Embedding`）。这些嵌入的大小为 `d_model`。
   - 创建一个单独的 `PositionalEncoding` 模块。由于其计算与词汇表无关，因此可以共享。请注意，在添加位置编码 (positional encoding)之前，令牌嵌入会乘以 `sqrt(d_model)`，这与原始论文中保持一致。
   - 我们使用 `nn.ModuleList` 来存放 `EncoderLayer` 和 `DecoderLayer` 实例的堆叠。这确保了层被正确注册为子模块。
   - 一个最终的 `nn.Linear` 层，`self.generator`，将解码器堆栈的 `d_model` 维度输出投影到目标词汇表（`tgt_vocab_size`）上的对数。
   - 应用了使用 Xavier 均匀初始化进行权重 (weight)初始化（`_reset_parameters`），这是 Transformer 的常见做法。
2. **编码 (`encode`)**:

   - 此方法包含编码器逻辑。它接收源序列（`src`），应用嵌入和位置编码，然后按顺序将结果通过堆栈中的每个 `EncoderLayer`。
   - 它接受掩码（`src_mask`，`src_key_padding_mask`），这些掩码会被传递到每个 `EncoderLayer` 内部的底层注意力机制 (attention mechanism)。`src_key_padding_mask` 对于防止对填充标记 (token)进行注意力计算很重要。
   - 编码器堆栈的最终输出（`memory`）表示已处理的源序列上下文 (context)。
3. **解码 (`decode`)**:

   - 此方法处理解码器逻辑。它接收目标序列（`tgt`）、来自编码器的 `memory` 以及各种掩码。
   - 与编码器类似，它对目标序列应用嵌入和位置编码。
   - 结果按顺序通过每个 `DecoderLayer`。每个 `DecoderLayer` 对目标序列执行自注意力 (self-attention)（使用 `tgt_mask` 和 `tgt_key_padding_mask`），并与编码器 `memory` 执行交叉注意力（使用 `memory_mask` 和 `memory_key_padding_mask`）。`tgt_mask` 在此尤其重要，用于阻止解码器在训练期间关注未来的标记（因果掩码）。
   - 最终输出是解码器对目标序列的表示，以源序列为条件。
4. **前向传播 (`forward`)**:

   - 主要的 `forward` 方法组织整个过程。
   - 它首先调用 `encode` 来获取 `memory`。
   - 然后，它使用目标序列和 `memory` 调用 `decode`。
   - 最后，它将 `generator` 线性层应用于解码器输出，以生成目标序列中每个位置的最终对数。这些对数随后可在训练期间与损失函数 (loss function)（如交叉熵）一同使用，或在推理 (inference)期间进行进一步处理（例如，使用 argmax 或采样）。

### 数据流图

以下图表展示了组装后的 Transformer 模型中的高层数据流：

> Transformer 模型中的高层数据流，展示了从输入标记 (token)经过嵌入 (embedding)、编码器、解码器和最终输出投影的路径。

这个完整的 `TransformerModel` 类提供了一个功能实现，基于之前开发的组件。它包含核心架构，可随时集成到训练循环中。后续章节将在此基础上进行构建，研究如何扩展此架构，在大规模数据集上高效训练它，并针对各种任务优化它。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构、其编码器-解码器设计、自注意力机制和位置编码的基础论文。
- [Transformer API Reference](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) — PyTorch Documentation (2024)
  Publisher: PyTorch Foundation
  PyTorch内置Transformer模块的官方文档，提供了实现细节和API用法。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Pages: Chapter 10 (Encoder-Decoder Models and Attention)
  一本全面的教材章节，详细介绍了Transformer架构，包括其组件和在自然语言处理中的应用。

---

[上一节](05-%E6%9E%84%E5%BB%BA%E7%BC%96%E7%A0%81%E5%99%A8%E5%B1%82%E5%92%8C%E8%A7%A3%E7%A0%81%E5%99%A8%E5%B1%82.md) · [下一节](../11-Transformer%E6%A8%A1%E5%9E%8B%E8%A7%84%E6%A8%A1%E5%8C%96%EF%BC%9A%E6%9E%B6%E6%9E%84%E9%80%89%E6%8B%A9/01-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E7%9A%84%E7%BC%A9%E6%94%BE%E5%AE%9A%E5%BE%8B.md)
