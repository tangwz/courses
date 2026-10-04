---
course: "advanced-tensorflow"
chapter: "implementing-advanced-architectures"
lesson: "building-attention-mechanisms"
sourceId: 2942
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/building-attention-mechanisms"
title: "从零构建注意力机制"
description: "实现多种形式的注意力机制，它们是许多现代架构的构建模块。"
order: 1
plots: []
sourceHash: "014b82ddaacbb6b2b2439ac6072e672987e73319766b0d8deebbdb857c1d49dd"
sourceCorrections: []
---

注意力机制 (attention mechanism)已成为许多先进深度学习 (deep learning)模型中的基本组成部分，特别是在Transformer等架构处理的序列建模任务中。
本质上，注意力机制允许模型在生成输出时，根据输入元素与当前处理步骤的相关性，动态地关注输入序列的不同部分，并分配不同的权重 (weight)或“注意力”。这与RNN等早期序列模型形成对比，后者通常按顺序处理信息，并可能因信息瓶颈而难以处理长距离依赖。

注意力机制不再仅仅依赖于编码器的最终隐藏状态，而是计算一个上下文 (context)向量 (vector)，作为输入表示的加权和。权重是根据当前输出步骤（由查询表示）与输入元素（由键表示）之间的关系动态确定的。

接下来，我们将使用TensorFlow从零开始实现最常见的注意力机制。

### 点积缩放注意力

这可以说是最广泛使用的注意力机制 (attention mechanism)，构成Transformer架构的核心。它对三个输入进行操作：查询（Q）、键（K）和值（V）。可以把它看作一个数据库检索系统：对于给定的查询（Q），我们计算它与每个可用键（K）的分数。这些分数决定了我们对相应的值（V）付出多少注意力（权重 (weight)）。输出是这些值的加权和。

公式如下：


$$
注意力(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$


这里：

- $Q$: 查询矩阵，形状为 $(..., N_q, d_k)$。$N_q$ 是查询的数量。
- $K$: 键矩阵，形状为 $(..., N_k, d_k)$。$N_k$ 是键/值的数量。
- $V$: 值矩阵，形状为 $(..., N_k, d_v)$。$N_k$ 必须与键的维度匹配。$d_v$ 是值的维度。
- $d_k$: 键和查询的维度。
- $\sqrt{d_k}$: 缩放因子。通过键维度平方根进行缩放，可以防止点积过大，这可能使softmax函数进入梯度非常小的区域，从而阻碍学习。

让我们使用TensorFlow实现它。我们将创建一个函数，接收Q、K、V和一个可选的掩码作为输入。掩码对于防止对某些位置（例如序列中的填充符或因果注意力（在解码器中使用）中的未来标记 (token)）的关注很重要。

```python
import tensorflow as tf

def scaled_dot_product_attention(q, k, v, mask=None):
    """计算注意力权重和输出。

    参数：
      q: 查询张量；形状 == (..., seq_len_q, depth)
      k: 键张量；形状 == (..., seq_len_k, depth)
      v: 值张量；形状 == (..., seq_len_v, depth_v)
         注意：seq_len_k == seq_len_v
      mask: 可选的浮点张量，其形状可广播到
            (..., seq_len_q, seq_len_k)。默认为None。

    返回：
      output: 值的加权和，形状 == (..., seq_len_q, depth_v)
      attention_weights: softmax后的注意力分数，
                         形状 == (..., seq_len_q, seq_len_k)
    """
    # 计算查询和键之间的点积。
    # q 形状: (..., seq_len_q, depth)
    # k 形状: (..., seq_len_k, depth)
    # matmul_qk 形状: (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # 将 matmul_qk 按深度 (dk) 的平方根缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 将掩码添加到缩放后的张量。
    # 掩码会将大的负值添加到不应被关注的位置。
    if mask is not None:
        # 掩码需要可广播到 scaled_attention_logits 的形状
        # 典型的掩码形状是 (batch_size, 1, 1, seq_len_k) 或 (batch_size, 1, seq_len_q, seq_len_k)
        scaled_attention_logits += (mask * -1e9) # 添加一个大的负数

    # 应用 softmax 获取注意力权重。
    # attention_weights 形状: (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 将权重乘以值以获得加权和。
    # attention_weights 形状: (..., seq_len_q, seq_len_k)
    # v 形状: (..., seq_len_v, depth_v)，其中 seq_len_v == seq_len_k
    # output 形状: (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

# 示例用法（说明性形状）
batch_size = 4
seq_len_q = 10 # 目标序列长度
seq_len_k = 12 # 源序列长度
depth = 64     # 键/查询维度
depth_v = 128  # 值维度

# 随机示例张量
queries = tf.random.normal([batch_size, seq_len_q, depth])
keys = tf.random.normal([batch_size, seq_len_k, depth])
values = tf.random.normal([batch_size, seq_len_k, depth_v]) # seq_len_k 与键匹配

# 可选掩码（例如，掩盖键中的填充符）
# 掩码形状 (batch_size, 1, 1, seq_len_k) 以正确广播
# 1 表示有效标记，0 表示填充/掩盖标记
# 我们需要 0 表示有效，1 表示掩盖以实现加法技巧
mask_values = tf.cast(tf.random.uniform((batch_size, 1, 1, seq_len_k)) > 0.5, tf.float32) # 示例掩码

# 计算注意力
attention_output, attention_weights = scaled_dot_product_attention(queries, keys, values, mask=mask_values)

print("注意力输出形状：", attention_output.shape)
print("注意力权重形状：", attention_weights.shape)
```

这个实现抓住了点积缩放注意力的要点。请注意，张量形状必须与矩阵乘法正确对齐 (alignment)。可选的掩码允许灵活控制哪些键/值对对每个查询的输出有所贡献。

### 加性注意力（Bahdanau注意力）

另一种有影响力的注意力机制 (attention mechanism)，常被称为Bahdanau注意力，比点积注意力稍早提出。它不是使用简单的点积，而是使用一个带有单个隐藏层的前馈网络来计算查询和键之间的对齐 (alignment)分数。

查询 $q$ 与键 $k_i$ 之间的分数（或能量）通常按如下方式计算：


$$
score(q, k_i) = v^T \tanh(W_q q + W_k k_i + b)
$$


在此，$W_q$、$W_k$ 和 $v$ 是学习到的权重 (weight)矩阵/向量 (vector)，$b$ 是偏置 (bias)项。所有键的分数随后通过softmax函数以获取注意力权重。

加性注意力可以处理不同维度的查询和键，因为投影 $W_q q$ 和 $W_k k_i$ 可以在$\tanh$激活之前将它们映射到公共维度。

以下是作为Keras层实现它的方式：

```python
import tensorflow as tf
from tensorflow.keras import layers

class AdditiveAttention(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # 全连接层，将查询和键投影到相同维度 (units)
        self.Wq = layers.Dense(units, use_bias=False, name='query_projection')
        self.Wk = layers.Dense(units, use_bias=False, name='key_projection')
        # 全连接层，从组合投影计算分数
        self.V = layers.Dense(1, name='score_projection') # 投影到单个分数

    def call(self, query, key, value, mask=None):
        """
        参数：
          query: 查询张量；形状 == (batch_size, seq_len_q, query_depth)
                 如果为单个查询，则为 (batch_size, query_depth)
          key: 键张量；形状 == (batch_size, seq_len_k, key_depth)
          value: 值张量；形状 == (batch_size, seq_len_k, value_depth)
          mask: 可选的掩码张量；形状可广播到
                (batch_size, seq_len_q, seq_len_k)。

        返回：
          context_vector: 值的加权和；形状 == (batch_size, seq_len_q, value_depth)
                           如果为单个查询，则为 (batch_size, value_depth)
          attention_weights: softmax后的注意力分数；
                             形状 == (batch_size, seq_len_q, seq_len_k)
                             如果为单个查询，则为 (batch_size, 1, seq_len_k)
        """
        # 如果查询是单个向量（例如，解码器状态），则添加时间维度
        # query 形状: (batch_size, 1, query_depth)
        if tf.rank(query) == 2:
            query = tf.expand_dims(query, 1) # 添加 seq_len_q 维度

        # 投影查询和键
        # query_proj 形状: (batch_size, seq_len_q, units)
        # key_proj 形状: (batch_size, seq_len_k, units)
        query_proj = self.Wq(query)
        key_proj = self.Wk(key)

        # 扩展维度以进行广播加法
        # query_proj 形状: (batch_size, seq_len_q, 1, units)
        # key_proj 形状: (batch_size, 1, seq_len_k, units)
        query_proj_expanded = tf.expand_dims(query_proj, 2)
        key_proj_expanded = tf.expand_dims(key_proj, 1)

        # 计算对齐分数（能量）
        # combined_proj 形状: (batch_size, seq_len_q, seq_len_k, units)
        # scores 形状: (batch_size, seq_len_q, seq_len_k, 1) -> (batch_size, seq_len_q, seq_len_k)
        scores = self.V(tf.nn.tanh(query_proj_expanded + key_proj_expanded))
        scores = tf.squeeze(scores, axis=-1) # 移除最后一个维度

        # 在 softmax 前应用掩码
        if mask is not None:
            # 如果查询最初是二维的，则掩码形状需要调整
            if tf.rank(mask) == 2: # 例如，形状 (batch_size, seq_len_k)
                mask = tf.expand_dims(mask, 1) # -> (batch_size, 1, seq_len_k)
            scores += (tf.cast(mask, tf.float32) * -1e9)

        # 使用 softmax 计算注意力权重
        # attention_weights 形状: (batch_size, seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # 计算上下文向量（值的加权和）
        # attention_weights 形状: (batch_size, seq_len_q, seq_len_k)
        # value 形状: (batch_size, seq_len_k, value_depth)
        # context_vector 形状: (batch_size, seq_len_q, value_depth)
        context_vector = tf.matmul(attention_weights, value)

        # 如果原始查询是二维的，则移除添加的维度
        if tf.rank(query) == 2:
            context_vector = tf.squeeze(context_vector, axis=1)
            # 如果需要，将 attention_weights 形状保持为 (batch_size, 1, seq_len_k) 以提高清晰度
            # attention_weights 形状保持为 (batch_size, seq_len_q, seq_len_k)，即 (batch_size, 1, seq_len_k)

        return context_vector, attention_weights

# 示例用法
batch_size = 4
seq_len_q = 1 # 示例：像解码器状态一样的单个查询
seq_len_k = 12 # 源序列长度
query_depth = 50
key_depth = 60
value_depth = 70
units = 32 # 加性注意力机制中的隐藏单元

# 示例张量
single_query = tf.random.normal([batch_size, query_depth])
keys = tf.random.normal([batch_size, seq_len_k, key_depth])
values = tf.random.normal([batch_size, seq_len_k, value_depth])
# 可选掩码（1表示掩码，0表示有效）-> 转换为布尔值以提高清晰度
mask = tf.cast(tf.random.uniform((batch_size, seq_len_k)) > 0.8, tf.bool) # 大约掩盖20%

additive_attention_layer = AdditiveAttention(units)
context, weights = additive_attention_layer(single_query, keys, values, mask=mask)

print("加性注意力上下文形状：", context.shape)
print("加性注意力权重形状：", weights.shape)

# 序列查询示例
seq_query = tf.random.normal([batch_size, 10, query_depth]) # seq_len_q = 10
context_seq, weights_seq = additive_attention_layer(seq_query, keys, values, mask=None) # 为简单起见不使用掩码
print("\n加性注意力上下文形状（序列查询）：", context_seq.shape)
print("加性注意力权重形状（序列查询）：", weights_seq.shape)
```

这个Keras层封装了逻辑，使其可重用。使用`Dense`层使得在模型训练期间学习投影权重（$W_q$，$W_k$，$V$）变得直接。

### 多头注意力 (multi-head attention)

多头注意力（MHA）通过并行执行多个注意力计算来增强基本注意力机制 (attention mechanism)，每个计算都关注输入的不同方面或“表示子空间”。

该过程涉及：

1. **线性投影：** 使用不同的学习到的线性投影（全连接层），将输入Q、K和V投影到多个低维子空间。这会创建 $h$ 组（Q'，K'，V'），其中 $h$ 是注意力头的数量。
2. **并行注意力：** 将选定的注意力函数（通常是点积缩放注意力）独立地应用于 $h$ 组投影集（Q'，K'，V'）中的每一个。
3. **拼接：** 拼接并行注意力层产生的 $h$ 个输出向量 (vector)。
4. **最终线性投影：** 对拼接后的输出应用最终线性投影（全连接层），以生成最终结果，通常将其映射回原始预期维度。

> 多头注意力的流程。输入被多次投影，每个头并行计算注意力，输出被拼接，并应用最终线性变换。

这使得模型能够同时捕获不同位置处不同类型的关系。让我们将其作为Keras层实现。

```python
import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        """
        参数：
            d_model: 模型的总维度（输出维度）。
                     必须能被 num_heads 整除。
            num_heads: 注意力头的数量。
        """
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        # 确保 d_model 能被 num_heads 整除
        assert d_model % self.num_heads == 0

        # 每个注意力头投影的深度
        self.depth = d_model // self.num_heads

        # 全连接层，用于Q、K、V的线性投影
        self.wq = layers.Dense(d_model, name='query_projection') # 投影到 d_model
        self.wk = layers.Dense(d_model, name='key_projection')   # 投影到 d_model
        self.wv = layers.Dense(d_model, name='value_projection') # 投影到 d_model

        # 用于最终线性投影的全连接层
        self.dense = layers.Dense(d_model, name='output_projection')

    def split_heads(self, x, batch_size):
        """将最后一个维度分割成 (num_heads, depth)。
        转置结果，使形状为 (batch_size, num_heads, seq_len, depth)

        参数：
          x: 输入张量；形状 == (batch_size, seq_len, d_model)
          batch_size: 批大小。

        返回：
          形状为 (batch_size, num_heads, seq_len, depth) 的张量
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        """
        参数：
          v: 值张量；形状 == (batch_size, seq_len_v, d_model)
          k: 键张量；形状 == (batch_size, seq_len_k, d_model)
          q: 查询张量；形状 == (batch_size, seq_len_q, d_model)
          mask: 可选的掩码。

        返回：
          output: 最终注意力输出；形状 == (batch_size, seq_len_q, d_model)
          attention_weights: 注意力权重；形状 ==
                             (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = tf.shape(q)[0]

        # 1. 线性投影
        # q, k, v 形状: (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 2. 分割成多个头
        # q, k, v 形状: (batch_size, num_heads, seq_len, depth)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # 3. 对每个头应用点积缩放注意力
        # scaled_attention 形状: (batch_size, num_heads, seq_len_q, depth)
        # attention_weights 形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # 4. 拼接头
        # 转置回: (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # 拼接: (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # 5. 最终线性投影
        # output 形状: (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

# 示例用法
batch_size = 4
seq_len = 15 # 为简单起见，这里假设Q、K、V序列长度相同
d_model = 128 # 模型维度
num_heads = 8

# 示例输入（自注意力中Q、K、V通常来自同一来源）
input_seq = tf.random.normal([batch_size, seq_len, d_model])

mha_layer = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# 自注意力示例：Q、K、V相同
output, weights = mha_layer(v=input_seq, k=input_seq, q=input_seq, mask=None)

print("多头注意力输出形状：", output.shape)
print("多头注意力权重形状：", weights.shape) # 注意 num_heads 维度
```

这个`MultiHeadAttention`层是Transformer编码器和解码器的基本构成单元。注意`split_heads`和拼接步骤，它们对于管理跨头的并行计算必不可少。

### 自注意力 (self-attention)

自注意力是注意力机制 (attention mechanism)（如点积缩放注意力或多头注意力 (multi-head attention)）的一种特定应用，其中查询、键和值都来源于*同一个*输入序列。这使得模型在计算每个标记 (token)的表示时，能够权衡*同一个序列*内不同词或标记的重要性。

例如，在句子“The animal didn't cross the street because it was too tired”（这只动物没有过马路，因为它太累了）中，自注意力可以帮助模型学习到“it”指代的是“the animal”而不是“the street”。

为了实现自注意力，您只需将相同的输入张量（或通过初始投影从其派生出的张量）作为查询、键和值参数 (parameter)传入您的注意力层（如上面所示的`MultiHeadAttention`）。`MultiHeadAttention`的示例用法已演示了这种情况。

这些构建模块——点积缩放注意力、加性注意力和多头注意力——为深度学习 (deep learning)模型融入上下文 (context)和关注提供了强大而灵活的方式。理解如何从零开始实现它们，对于定制架构和解释模型行为很有价值，也为实现Transformer等高级模型打下基础。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; Publisher: Curran Associates, Inc.; Pages: 5998-6008; DOI: [10.5555/3295222.3295349](https://doi.org/10.5555/3295222.3295349)
  介绍了Transformer架构以及缩放点积注意力、多头注意力和自注意力机制。这是许多现代注意力模型的来源。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR) 2015; Publisher: International Conference on Learning Representations; DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  提出了加性注意力机制，通常称为Bahdanau注意力，这是序列到序列模型中的一个重要进步。
- [tf.keras.layers.MultiHeadAttention](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention) — TensorFlow Developers (2024)
  官方Keras MultiHeadAttention层的文档，提供了其参数和在TensorFlow中使用的详细信息。
- [Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms/index.html) — Aston Zhang, Zack C. Lipton, Mu Li, Alex Smola (2023)
  Publisher: Cambridge University Press
  一本广泛使用的在线教科书中的章节，对各种注意力机制（包括缩放点积注意力和加性注意力）进行了教学性解释。
