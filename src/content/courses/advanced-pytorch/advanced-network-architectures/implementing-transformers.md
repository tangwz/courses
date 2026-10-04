---
course: "advanced-pytorch"
chapter: "advanced-network-architectures"
lesson: "implementing-transformers"
sourceId: 2684
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-2-advanced-network-architectures/implementing-transformers"
title: "从组件构建Transformer模型"
description: "逐步构建Transformer模型，了解自注意力、多头注意力及位置编码。"
order: 1
plots: []
sourceHash: "e0ca73cb8d0733a6ff1fe19ee1fda8eaa26051cae75c7900d065c583c3da13f2"
sourceCorrections: []
---

Transformer模型彻底改变了序列建模，在自然语言处理、计算机视觉及其他方面取得了最先进的成果。其效果源于自注意力 (self-attention)机制 (attention mechanism)，该机制使模型在生成输出时，能够衡量不同输入元素的重要性，而无论它们之间的距离如何。使用PyTorch从基本组件构建Transformer模型，有助于充分了解其工作原理。假定熟悉基本的PyTorch模块（`nn.Module`、`nn.Linear`等）和深度学习 (deep learning)原理。

### Transformer架构：概览

最初的Transformer模型由Vaswani等人于2017年在《Attention Is All You Need》中提出，采用编码器-解码器结构，适用于机器翻译等序列到序列的任务。

> 编码器-解码器Transformer架构的高级视图。编码器处理输入序列，而解码器则使用编码器的输出和先前生成的目标序列来生成下一个元素。

许多成功模型仅使用编码器堆栈（例如，BERT用于语言理解）或解码器堆栈（例如，GPT用于语言生成）。我们将侧重于实现所有变体共有的核心构成模块。

### 输入表示：嵌入 (embedding)和位置编码 (positional encoding)

神经网络 (neural network)处理数字，而非原始文本。因此，第一步是将输入词元 (token)（单词、子词 (subword)或字符）转换为数值向量 (vector)。

#### 词元嵌入

这通常通过使用嵌入层完成，该嵌入层本质上是一个查找表。词汇表 (vocabulary)中的每个唯一词元都被分配一个固定大小的稠密向量，$d_{model}$。在PyTorch中，这可使用`torch.nn.Embedding`直接实现。

```python
import torch
import torch.nn as nn
import math

# 示例参数
vocab_size = 10000 # 词汇表大小
d_model = 512      # 嵌入维度

embedding = nn.Embedding(vocab_size, d_model)

# 示例用法：2个序列的批次，长度为10
input_tokens = torch.randint(0, vocab_size, (2, 10)) # (批次大小, 序列长度)
input_embeddings = embedding(input_tokens)          # (批次大小, 序列长度, d_model)

print("输入形状:", input_tokens.shape)
print("嵌入形状:", input_embeddings.shape)
```

#### 位置编码

接下来将要考察的自注意力 (self-attention)机制 (attention mechanism)同时处理序列元素。它本质上不考虑词元的顺序或位置。如果没有位置信息，注意力机制在嵌入后会将“the cat sat on the mat”和“the mat sat on the cat”视为相同。

为了解决这个问题，Transformer模型将每个词元的*位置*信息注入其嵌入中。原始论文提出使用固定的正弦函数：


$$
PE_{(位置, 2i)} = \sin(位置 / 10000^{2i/d_{model}})
$$


$$
PE_{(位置, 2i+1)} = \cos(位置 / 10000^{2i/d_{model}})
$$


这里，$pos$ 是词元在序列中的位置，$i$ 是嵌入向量内的维度索引（$0 \le 2i < d_{model}$）。位置编码的每个维度对应于不同频率的正弦曲线。这种选择使模型能够更容易地获取相对位置信息，因为 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。

或者，可以使用可学习的位置嵌入（类似于词元嵌入，但查找的是位置索引）。我们在这里将实现正弦版本。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置索引 (0 到 max_len - 1)
        position = torch.arange(max_len).unsqueeze(1) # 形状: (max_len, 1)

        # 计算正弦和余弦参数的除数项
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 形状: (d_model / 2)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model) # 形状: (max_len, d_model)

        # 对偶数索引应用sin，对奇数索引应用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加批次维度并注册为缓冲区（非模型参数）
        pe = pe.unsqueeze(0) # 形状: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 张量, 形状 [批次大小, 序列长度, d_model]
        返回:
            张量, 形状 [批次大小, 序列长度, d_model]
        """
        # 将位置编码添加到输入嵌入
        # self.pe 形状为 (1, max_len, d_model)。我们取 x 的序列长度范围内的切片。
        # x 的形状为 (批次大小, 序列长度, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 示例用法
pos_encoder = PositionalEncoding(d_model, dropout=0.1)
final_input = pos_encoder(input_embeddings * math.sqrt(d_model)) # 在添加位置编码前对嵌入进行缩放

print("位置编码后的形状:", final_input.shape)
# 注意：原始论文在添加位置编码前，会按 sqrt(d_model) 缩放嵌入。
```

第一个Transformer层的最终输入是词元嵌入（可选缩放）和位置编码之和。

### 核心组件：多头自注意力 (self-attention)

注意力机制 (attention mechanism)使模型在处理特定元素时能够关注输入序列的相关部分。自注意力将*单个*序列的不同位置关联起来，以计算该序列的表示。

#### 缩放点积注意力

基本构成块是缩放点积注意力。对于序列中的每个元素，我们计算三个向量 (vector)：查询（$Q$）、键（$K$）和值（$V$）。这些通常通过将输入嵌入 (embedding)（加上位置编码 (positional encoding)）乘以可学习的权重 (weight)矩阵 $W_Q$、$W_K$ 和 $W_V$ 来获得。

想象您正在处理句子“making Transformer models more interpretable”中的单词“making”。

- “making”的**查询**向量提问：“在当前语境中，句子的哪些部分与理解我的意思相关？”
- 句子中的每个词都生成一个向量，本质上是说：“这是我所代表的信息。”
- 每个词还会生成一个**值**向量：“如果您认为我相关，这是我将提供的信息。”

一个词（例如，“making”）的查询与另一个词（例如，“Transformer”）的键之间的注意力分数是使用点积计算的。这些分数确定“making”应该对“Transformer”投入多少注意力。

分数通过向量维度（$d_k$）的平方根进行缩放，以防止点积变得过大，这可能使softmax函数饱和并导致梯度消失。然后，softmax函数将这些分数转换为总和为1的概率（权重）。

最后，查询词（“making”）的输出是序列中所有值向量的加权和，其中权重是计算出的概率。

公式如下：


$$
\text{注意力}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


其中 $Q$、$K$ 和 $V$ 是包含序列中所有词元 (token)的查询、键和值的矩阵。

#### 多头注意力 (multi-head attention)

多头注意力不同于使用$d_{model}$维度的Q、K、V向量进行单次注意力计算，它将输入Q、K、V向量通过$h$次（其中$h$是头数）不同的可学习线性投影（权重矩阵）投影到维度$d_q$、$d_k$、$d_v$（通常$d_q = d_k = d_v = d_{model} / h$）。

缩放点积注意力随后独立应用于这些投影版本（每个“头”）。这使模型能够同时关注来自不同表示子空间和不同位置的信息。这就像同时针对输入提出多个不同的问题（查询）一样。

所有$h$个头的输出被连接起来，然后通过一个最终线性层（$W_O$），以生成多头注意力层的最终输出。

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """计算缩放点积注意力"""
    d_k = q.size(-1) # 获取最后一个维度（K的嵌入维度）
    # Q与K转置的矩阵乘法: (..., 查询序列长度, d_k) x (..., 键序列长度, d_k) -> (..., 查询序列长度, 键序列长度)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用掩码（如果提供），将掩码位置设置为一个非常小的数字 (-1e9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 应用softmax以获取注意力权重
    attn_weights = torch.softmax(scores, dim=-1) # (..., 查询序列长度, 键序列长度)

    # 权重与V的矩阵乘法: (..., 查询序列长度, 键序列长度) x (..., 值序列长度, d_v) -> (..., 查询序列长度, d_v)
    # 注意: 键序列长度 == 值序列长度
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # 每个头的键/查询维度

        # Q、K、V投影的线性层（应用于所有头）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 连接后的最终线性层
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (批次大小, 序列长度, d_model)
        batch_size, seq_len, _ = x.size()
        # 重塑为 (批次大小, 序列长度, 头数, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # 转置为 (批次大小, 头数, 序列长度, d_k) 以进行注意力计算
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: (批次大小, 头数, 序列长度, d_k)
        batch_size, _, seq_len, _ = x.size()
        # 转置回 (批次大小, 序列长度, 头数, d_k)
        x = x.transpose(1, 2).contiguous() # 确保转置后内存连续
        # 重塑为 (批次大小, 序列长度, d_model)
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q, k, v: (批次大小, 序列长度, d_model)
        # 掩码: (批次大小, 1, 查询序列长度, 键序列长度) 或类似的可广播形状

        # 1. 应用线性投影
        q = self.W_q(q) # (批次大小, 查询序列长度, d_model)
        k = self.W_k(k) # (批次大小, 键序列长度, d_model)
        v = self.W_v(v) # (批次大小, 值序列长度, d_model) (键序列长度 == 值序列长度)

        # 2. 分割成多个头
        q = self.split_heads(q) # (批次大小, 头数, 查询序列长度, d_k)
        k = self.split_heads(k) # (批次大小, 头数, 键序列长度, d_k)
        v = self.split_heads(v) # (批次大小, 头数, 值序列长度, d_k)

        # 3. 应用缩放点积注意力
        # 输出: (批次大小, 头数, 查询序列长度, d_k)
        # 注意力权重: (批次大小, 头数, 查询序列长度, 键序列长度)
        attention_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # 4. 合并头
        output = self.combine_heads(attention_output) # (批次大小, 查询序列长度, d_model)

        # 5. 最终线性层
        output = self.W_o(output) # (批次大小, 查询序列长度, d_model)

        return output # 通常我们只需要输出，不需要权重，用于下一层

# 示例用法
mha = MultiHeadAttention(d_model=512, num_heads=8)
# 在自注意力中，Q、K和V最初通常是同一个张量
query = key = value = final_input # 形状: (批次大小, 序列长度, d_model)
attention_result = mha(query, key, value, mask=None) # 掩码对填充/解码很重要

print("多头注意力输出形状:", attention_result.shape)
```

#### 掩码

掩码在两种情况中不可或缺：

1. **填充掩码：** 为了防止模型关注批次中不同长度序列的填充词元。此掩码在编码器和解码器自注意力中都应用。它通常具有类似 `(batch_size, 1, 1, seq_len_k)` 的形状，填充位置包含0，否则为1。
2. **前瞻掩码：** 在解码器的*自*注意力层中，这阻止一个位置关注后续位置。这确保位置 $i$ 的预测只能依赖于小于 $i$ 的位置的已知输出。这通过掩盖（在softmax前设置为 `-1e9`）注意力分数矩阵的上三角部分来实现。

### 加和归一化 (normalization)层

Transformer中的每个子层（如多头注意力 (multi-head attention)或前馈网络）都具有残差连接，然后是层归一化。

#### 残差连接（加和）

子层的输出被添加到子层的*输入*中：`output = x + Sublayer(x)`。这种技术借鉴自残差网络（ResNets），有助于缓解深度网络中的梯度消失问题，使梯度在反向传播 (backpropagation)过程中更直接地流过网络。它也使得训练更深层的模型成为可能。

#### 层归一化（归一化）

层归一化（`nn.LayerNorm`）独立地对批次中每个单独数据样本（词元 (token)）的*特征维度*（即 $d_{model}$ 维度）上的激活进行归一化。这与批归一化形成对比，后者对批次维度进行归一化。层归一化在自然语言处理和Transformer模型中通常更受青睐，原因如下：

- 它不依赖于批次大小，使其在小批次或可变序列长度下也能保持稳定。
- 它为每个序列元素独立地提供一致的归一化。

加和归一化步骤通常实现为：`output = LayerNorm(x + Dropout(Sublayer(x)))`。Dropout通常在残差加法和归一化之前应用于子层的输出。

```python
class AddNorm(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer_output: torch.Tensor) -> torch.Tensor:
        # 应用残差连接和Dropout，然后是层归一化
        return self.layer_norm(x + self.dropout(sublayer_output))

# 示例：在多头注意力后应用加和归一化
dropout_rate = 0.1
add_norm1 = AddNorm(d_model, dropout_rate)
# 'final_input' 是MHA层的输入
normed_attention_output = add_norm1(final_input, attention_result)

print("加和归一化输出形状:", normed_attention_output.shape)
```

### 位置维前馈网络（FFN）

在注意力子层（及其加和归一化 (normalization)）之后，每个位置的表示都通过一个相同且独立的前馈网络（FFN）。该网络通常由两个线性变换和一个非线性激活函数 (activation function)组成，通常是ReLU或GeLU（高斯误差线性单元）。


$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \quad (\text{使用ReLU})
$$


维度通常在第一个线性层中增加（例如，到 $d_{ff} = 4 \times d_{model}$），然后在第二个层中减少回 $d_{model}$。这种FFN使模型能够独立地处理通过注意力在每个位置获取的信息，从而增加了非线性建模能力。

```python
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU() # 或 nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (批次大小, 序列长度, d_model)
        x = self.linear1(x)     # (批次大小, 序列长度, d_ff)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)     # (批次大小, 序列长度, d_model)
        return x

# 示例用法
d_ff = d_model * 4 # 常见做法
ffn = PositionWiseFeedForward(d_model, d_ff, dropout_rate)
ffn_output = ffn(normed_attention_output)

# 应用第二个加和归一化层
add_norm2 = AddNorm(d_model, dropout_rate)
# 'normed_attention_output' 是FFN的输入
encoder_layer_output = add_norm2(normed_attention_output, ffn_output)

print("FFN输出形状:", ffn_output.shape)
print("编码器层输出形状:", encoder_layer_output.shape)
```

### 层的组合

有了这些组件，我们可以定义一个完整的编码器层和解码器层。

#### 编码器层

一个编码器层包含：

1. 多头自注意力 (self-attention)。
2. 加和归一化 (normalization)。
3. 位置维前馈网络。
4. 加和归一化。

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 自注意力子层
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.add_norm1(x, attn_output) # 残差连接 + 归一化

        # 前馈子层
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output) # 残差连接 + 归一化
        return x
```

#### 解码器层

一个解码器层稍微复杂一些，包含两种注意力机制 (attention mechanism)：

1. **带掩码的多头自注意力：** 使用前瞻掩码，关注解码器的输入序列（目前已生成的目标序列）。
2. 加和归一化。
3. **多头编码器-解码器注意力：** 关注编码器堆栈的输出（通常称为`记忆`）。查询（$Q$）来自上一个解码器子层的输出，而键（$K$）和值（$V$）来自编码器输出。这使解码器在生成*输出*序列时能够考虑*输入*序列的相关部分。这里可能需要来自编码器输入的填充掩码。
4. 加和归一化。
5. 位置维前馈网络。
6. 加和归一化。

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # 1. 带掩码的自注意力子层
        # Q=x, K=x, V=x; 使用前瞻掩码
        self_attn_output = self.masked_self_attn(q=x, k=x, v=x, mask=look_ahead_mask)
        x = self.add_norm1(x, self_attn_output)

        # 2. 编码器-解码器注意力子层
        # Q=x (来自上一层), K=编码器输出, V=编码器输出
        # 使用与编码器输出相关的填充掩码
        enc_dec_attn_output = self.encoder_decoder_attn(q=x, k=encoder_output, v=encoder_output, mask=padding_mask)
        x = self.add_norm2(x, enc_dec_attn_output)

        # 3. 前馈子层
        ffn_output = self.ffn(x)
        x = self.add_norm3(x, ffn_output)

        return x
```

### 构建完整的Transformer

最终的Transformer模型堆叠多个编码器层（例如 $N=6$）形成编码器，并堆叠多个解码器层（例如 $N=6$）形成解码器。`nn.ModuleList` 对此很方便。

```python
class Transformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 d_model: int, num_heads: int, d_ff: int,
                 input_vocab_size: int, target_vocab_size: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.encoder_embedding = nn.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.final_linear = nn.Linear(d_model, target_vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def create_padding_mask(self, seq: torch.Tensor, pad_token_idx: int = 0) -> torch.Tensor:
        # 序列形状: (批次大小, 序列长度)
        # 输出掩码形状: (批次大小, 1, 1, 序列长度)
        mask = (seq != pad_token_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def create_look_ahead_mask(self, size: int) -> torch.Tensor:
        # 创建一个上三角矩阵用于掩盖未来词元
        # 输出掩码形状: (1, 1, 大小, 大小)
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        # 我们希望在掩盖处为0，所以我们进行反转（如果注意力中使用0进行掩盖）
        # 或者如果注意力函数期望在掩盖处为True，则按原样返回
        # 假设 scaled_dot_product_attention 使用 masked_fill(mask == 0, -1e9) 或 masked_fill(mask == True, -1e9)，请相应调整。
        # 让我们假设后者 (True表示掩码)
        return ~mask.unsqueeze(0).unsqueeze(0) # 在掩盖处设置为False

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # 源: (批次大小, 源序列长度)
        # 源掩码: (批次大小, 1, 1, 源序列长度)
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_pos_emb = self.positional_encoding(src_emb)
        enc_output = self.dropout(src_pos_emb)

        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output # (批次大小, 源序列长度, d_model)

    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # 目标: (批次大小, 目标序列长度)
        # 编码器输出: (批次大小, 源序列长度, d_model)
        # 前瞻掩码: (批次大小, 1, 目标序列长度, 目标序列长度)
        # 填充掩码: (批次大小, 1, 1, 源序列长度) # 在编码器-解码器注意力中使用

        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_pos_emb = self.positional_encoding(tgt_emb)
        dec_output = self.dropout(tgt_pos_emb)

        for layer in self.decoder_layers:
            dec_output = layer(dec_output, encoder_output, look_ahead_mask, padding_mask)

        return dec_output # (批次大小, 目标序列长度, d_model)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # 源: (批次大小, 源序列长度)
        # 目标: (批次大小, 目标序列长度) 通常为训练目的而右移

        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt) # 如果目标也有填充，也需要
        look_ahead_mask = self.create_look_ahead_mask(tgt.size(1)).to(tgt.device)

        # 将前瞻掩码和目标填充掩码结合用于解码器自注意力
        # 确保两个掩码都可广播: (批次大小, 1, 目标序列长度, 目标序列长度)
        combined_look_ahead_mask = torch.logical_and(tgt_padding_mask.transpose(-2, -1), look_ahead_mask)

        encoder_output = self.encode(src, src_padding_mask)
        decoder_output = self.decode(tgt, encoder_output, combined_look_ahead_mask, src_padding_mask)

        # 最终线性投影
        output = self.final_linear(decoder_output) # (批次大小, 目标序列长度, 目标词汇表大小)
        return output # 通常在推理/损失计算期间，模型外部接着Softmax

# 示例实例化（参数仅作说明）
transformer_model = Transformer(
    num_encoder_layers=6, num_decoder_layers=6,
    d_model=512, num_heads=8, d_ff=2048,
    input_vocab_size=10000, target_vocab_size=12000,
    max_seq_len=500, dropout=0.1
)

# 用于形状检查的虚拟输入（假设批次大小为2）
src_dummy = torch.randint(1, 10000, (2, 100)) # (批次, 源长度)
tgt_dummy = torch.randint(1, 12000, (2, 120)) # (批次, 目标长度) - 例如右移的目标

# 如果GPU可用，将模型和数据移至GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# transformer_model.to(device)
# src_dummy = src_dummy.to(device)
# tgt_dummy = tgt_dummy.to(device)

output_logits = transformer_model(src_dummy, tgt_dummy)
print("最终输出形状 (logits):", output_logits.shape) # 应为 (2, 120, 12000)
```

通过从这些基本PyTorch模块组装Transformer，您将具体了解信息如何在模型中流动以及注意力机制 (attention mechanism)如何实现语境感知的序列处理。这种基于组件的实现也提供了一个灵活的构建方式，用于试验架构变体或使模型适应不同任务。请记住，高效训练此类模型需要仔细考虑优化、正则化 (regularization)和数据处理，这些主题将在后续章节中介绍。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.55989/nips.2017.p5998](https://doi.org/10.55989/nips.2017.p5998)
  这篇基础论文介绍了 Transformer 架构和自注意力机制，推动了序列建模领域的重大进展。
- [torch.nn API Reference](https://pytorch.org/docs/stable/nn.html) — PyTorch Core Team (2024)
  Publisher: PyTorch
  PyTorch 神经网络模块的官方文档，提供了 `nn.Linear`、`nn.Embedding` 和 `nn.LayerNorm` 等层的定义和使用示例。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这本教科书涵盖了深度学习的理论基础和实践方面，包括与 Transformer 相关的架构。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Stanford University (2024)
  Publisher: Stanford University
  这门高级大学课程提供了关于深度学习在自然语言处理中应用的详细讲座和资源，包括对 Transformer 模型的深入讨论。
