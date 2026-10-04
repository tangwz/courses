---
course: "introduction-to-transformer-models"
chapter: "training-implementing-transformers"
lesson: "practice-assembling-transformer"
sourceId: 3036
sourceUrl: "https://apxml.com/zh/courses/introduction-to-transformer-models/chapter-4-training-implementing-transformers/practice-assembling-transformer"
title: "实践：组装一个基本Transformer"
description: "将已实现的组件（编码器、解码器）组合起来，形成一个可用的Transformer模型结构。"
order: 8
plots: []
sourceHash: "ee53634321c7598eb020ea345dea48b7d0055230af2b2e132210b145269e35b9"
sourceCorrections: []
---

好的，现在我们来整合之前仔细研究过的各个组件。在前面的章节中，我们学习了这些构成元素：输入嵌入 (embedding)、位置编码 (positional encoding)、多头注意力 (multi-head attention)（包括自注意力 (self-attention)变体和缩放点积注意力）、残差连接与层归一化 (normalization)、逐位置前馈网络，以及独立的编码器和解码器层的结构。我们也讨论了如何准备数据、创建带掩码的批次、选择损失函数 (loss function)以及选择优化方法。

本练习旨在组装一个完整的Transformer模型结构，整合其主要组件。这些组件包含输入嵌入、位置编码、多头注意力（涵盖自注意力变体和缩放点积注意力）、Add & Norm 层、逐位置前馈网络，以及独立的编码器和解码器层的结构。我们将定义一个主要的`Transformer`类，用于编排数据流经编码器和解码器堆栈。此实践还将涉及数据准备、创建带掩码的批次、选择损失函数和优化策略。

我们将假设你已经能够使用以下组件的实现（可能来自于之前的实践练习或提供的辅助模块）：

- `InputEmbeddings`：将输入token ID转换为密集向量 (vector) (dense vector)。
- `PositionalEncoding`：向嵌入中添加位置信息。
- `EncoderLayer`：包含多头自注意力和前馈子层。
- `DecoderLayer`：包含带掩码的多头自注意力、编码器-解码器注意力以及前馈子层。
- `MultiHeadAttention`：核心注意力机制 (attention mechanism)。
- `PositionwiseFeedForward`：全连接前馈网络。
- `OutputProjection`：一个最终的线性层，用于将解码器输出映射到词汇表 (vocabulary)概率。

现在我们来构建主要的`Transformer`类。为了清晰起见，我们将使用类似PyTorch的伪代码风格，侧重于架构本身。

### 定义Transformer类

主类将初始化所有必需的层，并定义`forward`（前向）传递过程，该过程接收源序列和目标序列（以及它们的掩码），并生成最终的输出对数。

```python
import torch
import torch.nn as nn
import copy

# 假设 EncoderLayer、DecoderLayer、InputEmbeddings、PositionalEncoding、
# OutputProjection 已根据前几节/章的内容在其他地方定义。

class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 d_model: int, # 嵌入维度
                 n_head: int,  # 注意力头数量
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_ff: int,    # 前馈层的维度
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()

        self.d_model = d_model

        # 嵌入和位置编码
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        # --- 编码器堆栈 ---
        # 创建一个EncoderLayer实例
        encoder_layer = EncoderLayer(d_model, n_head, d_ff, dropout)
        # 使用深拷贝创建 N 个独立的层
        self.encoder_stack = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_encoder_layers)])

        # --- 解码器堆栈 ---
        # 创建一个DecoderLayer实例
        decoder_layer = DecoderLayer(d_model, n_head, d_ff, dropout)
        # 使用深拷贝创建 N 个独立的层
        self.decoder_stack = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])

        # 最终输出层
        self.output_projection = OutputProjection(d_model, tgt_vocab_size)

        # 初始化参数（对稳定训练很重要）
        self._initialize_parameters()

    def _initialize_parameters(self):
        # 对线性层使用Xavier均匀初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """通过编码器堆栈处理源序列。"""
        # 应用嵌入和位置编码
        src_emb = self.positional_encoding(self.src_embedding(src))
        # 通过每个编码器层
        encoder_output = src_emb
        for layer in self.encoder_stack:
            encoder_output = layer(encoder_output, src_mask)
        return encoder_output

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: torch.Tensor, src_tgt_mask: torch.Tensor) -> torch.Tensor:
        """通过解码器堆栈处理目标序列和编码器输出。"""
        # 应用嵌入和位置编码
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        # 通过每个解码器层
        decoder_output = tgt_emb
        for layer in self.decoder_stack:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask, src_tgt_mask)
        return decoder_output

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor, # 用于编码器中的自注意力
                tgt_mask: torch.Tensor, # 用于解码器中的带掩码自注意力
                src_tgt_mask: torch.Tensor # 用于解码器中的编码器-解码器注意力
               ) -> torch.Tensor:
        """
        Transformer模型的主要前向传递。
        src: (批次大小, 源序列长度)
        tgt: (批次大小, 目标序列长度)
        src_mask: (批次大小, 1, 1, 源序列长度)
        tgt_mask: (批次大小, 1, 目标序列长度, 目标序列长度)
        src_tgt_mask: (批次大小, 1, 1, 源序列长度)
        """
        # 1. 源序列通过编码器
        encoder_output = self.encode(src, src_mask) # (批次大小, 源序列长度, d_model)

        # 2. 目标序列和编码器输出通过解码器
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask) # (批次大小, 目标序列长度, d_model)

        # 3. 将解码器输出映射到词汇空间
        logits = self.output_projection(decoder_output) # (批次大小, 目标序列长度, 目标词汇表大小)

        return logits
```

### 组装的考虑因素

1. **模块化：** `Transformer`类作为一个容器。它不自己实现注意力或前馈逻辑，而是将这些任务委托给`EncoderLayer`和`DecoderLayer`模块。这有利于代码重用和清晰性。
2. **参数 (parameter)共享（或缺乏共享）：** 请注意在创建编码器和解码器堆栈时使用了`copy.deepcopy`。尽管堆栈中每个层的*架构*是相同的，但*参数*（权重 (weight)和偏置 (bias)）通常在层之间*不*共享。每个层学习自己的变换。
3. **层堆叠：** 我们使用`nn.ModuleList`（或在其他框架中等效的结构）来保存编码器和解码器层的堆栈。这确保了所有层都被正确注册为子模块，并且在训练模型时它们的参数都被包含在内。
4. **辅助方法：** 我们定义了独立的`encode`和`decode`方法。这使得主要的`forward`方法更简洁，更易于理解。如果特定应用需要（例如使用编码器输出来生成句子嵌入 (embedding)），这也有助于仅使用编码器或解码器部分。
5. **掩码处理：** `forward`方法明确要求我们之前讨论过的不同掩码（`src_mask`、`tgt_mask`、`src_tgt_mask`）。这些对于处理填充和防止解码器关注未来token非常重要。在数据准备阶段正确生成这些掩码是一个重要的步骤。
6. **参数初始化：** 包含了一个简单的参数初始化策略（Xavier均匀初始化）。适当的初始化对于有效训练深度网络很重要。

### 可视化数据流

以下图表展示了组装后的`Transformer`类在`forward`传递过程中的高级数据流：

> 组装后的Transformer模型内部的高级数据流，展示了输入、嵌入 (embedding)、编码器/解码器堆栈以及最终的输出投影。掩码在相关阶段提供。

### 实例化模型

定义好类后，你可以通过指定超参数 (parameter) (hyperparameter)来创建一个实例：

```python
# 示例超参数
num_layers = 6
d_model = 512
n_head = 8
d_ff = 2048 # 通常是 4 * d_model
src_vocab = 10000 # 示例源词汇表大小
tgt_vocab = 12000 # 示例目标词汇表大小
dropout_rate = 0.1
max_len = 500

# 实例化模型
transformer_model = Transformer(num_encoder_layers=num_layers,
                                num_decoder_layers=num_layers,
                                d_model=d_model,
                                n_head=n_head,
                                src_vocab_size=src_vocab,
                                tgt_vocab_size=tgt_vocab,
                                d_ff=d_ff,
                                dropout=dropout_rate,
                                max_seq_len=max_len)

print(f"Transformer model instantiated with {num_layers} layers, d_model={d_model}.")
# 你可以在这里添加一个使用模拟输入的检查：
# dummy_src = torch.randint(0, src_vocab, (2, 10)) # 批次大小 2，序列长度 10
# dummy_tgt = torch.randint(0, tgt_vocab, (2, 12)) # 批次大小 2，序列长度 12
# ... 创建模拟掩码 ...
# output = transformer_model(dummy_src, dummy_tgt, dummy_src_mask, dummy_tgt_mask, dummy_src_tgt_mask)
# print(f"Output shape: {output.shape}") # 应该是 (2, 12, 目标词汇表大小)
```

这个组装好的`Transformer`类提供了完整的结构。下一步是设置训练循环：输入批次数据（源序列、目标序列和相应的掩码），计算模型输出对数与实际目标序列之间的损失（例如，交叉熵），并使用优化器（如带有特定学习率调度策略的Adam）通过反向传播 (backpropagation)来更新模型参数。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  提出Transformer架构的基础论文，详细介绍了其编码器-解码器结构以及多头注意力、位置编码等核心组件。
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) — Alexander Rush (2018)
  一个有影响力且广泛使用的原始Transformer模型的PyTorch实现，提供逐行解释，与本节讨论的组装过程一致。
- [torch.nn - PyTorch 2.3 documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Contributors (2024)
  PyTorch神经网络模块的官方文档，提供了`nn.Module`、`nn.ModuleList`以及构建Transformer等模型所需的其他关键组件的详细信息。
- [Dive into Deep Learning](https://d2l.ai/chapter_attention-mechanisms/transformer.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2024)
  Publisher: Cambridge University Press
  一本全面的交互式深度学习书籍，包含PyTorch实现，其中有关于Transformer架构及其组件的详细章节。
