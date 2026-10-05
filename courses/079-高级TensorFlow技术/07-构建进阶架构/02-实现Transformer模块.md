# 实现Transformer模块

来源：[原文](https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/implementing-transformer-blocks)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Transformer架构的核心构建单元，即编码器和解码器层，通过注意力机制 (attention mechanism)进行组合。Transformer 最初在论文《Attention Is All You Need》（Vaswani 等人，2017年）中提出，现已成为自然语言处理中许多任务的根本。实现这些模块的结构将主要在TensorFlow和Keras中进行，这要求您熟悉`tf.keras.layers.Layer`的子类化以及自注意力 (self-attention)原理。

Transformer架构通常由堆叠的相同编码器层和堆叠的相同解码器层组成。我们来搭建每种类型的层。

## Transformer编码器层

编码器层同时处理整个输入序列。它的主要目的是生成输入数据的丰富表示，编码序列中每个元素的上下文 (context)信息。每个编码器层包含两个主要的子层：

1. **多头自注意力 (self-attention)：** 允许输入序列中的每个位置关注序列中的所有位置（包括自身）。这捕获了不分距离的依赖关系。
2. **位置前馈网络（FFN）：** 一个简单的、全连接的前馈网络，独立应用于每个位置。

残差连接在两个子层周围使用，之后是层归一化 (normalization)。每个子层的输出是`LayerNorm(x + Sublayer(x))`，在此`Sublayer(x)`是子层本身实现的函数。

> 单个Transformer编码器层的结构。虚线表示残差连接。

### 实现位置前馈网络

FFN由两个线性变换组成，中间有一个激活函数 (activation function)。激活函数的常见选择是ReLU，GELU也常使用。内层维度（`dff`）通常大于模型的维度（`d_model`）。


$$
FFN(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$


以下是使用Keras层的一个简单实现：

```python
import tensorflow as tf

class PositionWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, activation='relu', **kwargs):
        """
        初始化位置前馈网络。

        参数：
            d_model: 模型的维度（输入/输出）。
            dff: 内层前馈网络的维度。
            activation: 内层的激活函数（'relu'或'gelu'）。
            **kwargs: 基层类的额外关键字参数。
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        # 使用 kernel_initializer 以获得更好的权重初始化
        initializer = tf.keras.initializers.GlorotUniform()

        self.dense_1 = tf.keras.layers.Dense(dff, activation=activation,
                                             kernel_initializer=initializer,
                                             name='ffn_dense_1')
        self.dense_2 = tf.keras.layers.Dense(d_model,
                                             kernel_initializer=initializer,
                                             name='ffn_dense_2')

    def call(self, x):
        """
        FFN的前向传播。

        参数：
            x: 输入张量，形状为 (batch_size, seq_len, d_model)。

        返回：
            输出张量，形状为 (batch_size, seq_len, d_model)。
        """
        x = self.dense_1(x) # Shape: (batch_size, seq_len, dff)
        x = self.dense_2(x) # Shape: (batch_size, seq_len, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'dff': self.dff,
            'activation': self.dense_1.activation.__name__ # Store activation name
        })
        return config
```

### 组装编码器层

现在，我们将多头注意力 (multi-head attention)层（假设我们已在上一节中定义了`MultiHeadAttention`层）和FFN组合成一个`EncoderLayer`。我们还加入了Dropout进行正则化 (regularization)，应用于每个子层之后，但在残差连接和归一化 (normalization)*之前*。

```python
# 假设MultiHeadAttention层在其他地方已定义或已导入
# from your_attention_module import MultiHeadAttention

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        """
        初始化一个Transformer编码器层。

        参数：
            d_model: 模型的维度。
            num_heads: 注意力头的数量。
            dff: 内层前馈网络的维度。
            rate: Dropout比率。
            **kwargs: 基层类的额外关键字参数。
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads, name='multi_head_attention')
        self.ffn = PositionWiseFeedForward(d_model, dff, name='position_wise_ffn')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')

        self.dropout1 = tf.keras.layers.Dropout(rate, name='dropout_1')
        self.dropout2 = tf.keras.layers.Dropout(rate, name='dropout_2')

    def call(self, x, training, mask):
        """
        编码器层的前向传播。

        参数：
            x: 输入张量，形状为 (batch_size, input_seq_len, d_model)。
            training: 布尔值，指示层是否应在训练模式下运行（应用dropout）。
            mask: 用于自注意力的填充掩码，形状为 (batch_size, 1, 1, input_seq_len)。

        返回：
            输出张量，形状为 (batch_size, input_seq_len, d_model)。
            注意力权重（可选，如果MultiHeadAttention返回它们）。
        """
        # 多头注意力模块
        # 注意：MHA通常接收query、key、value和mask。对于自注意力，query=key=value=x。
        attn_output, attn_weights = self.mha(x, x, x, mask, training=training) # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) # (batch_size, input_seq_len, d_model)

        # 前馈模块
        ffn_output = self.ffn(out1) # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output) # (batch_size, input_seq_len, d_model)

        # 返回输出和可选的注意力权重以供检查
        # 根据您的MultiHeadAttention实现，您可能只返回out2
        return out2 # 或者返回 out2, attn_weights

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config
```

请注意`training`参数 (parameter)的使用，它对于控制训练和推断期间的Dropout行为很重要。`mask`参数用于防止关注输入序列中的填充标记 (token)。

## Transformer解码器层

解码器层与编码器层相似，但引入了第三个子层来关注编码器堆栈的输出。其目的是一次生成输出序列的一个元素，以编码输入和先前生成的输出元素为条件。

每个解码器层包含三个主要的子层：

1. **带掩码多头自注意力 (self-attention)：** 允许*输出*序列中的每个位置关注*输出*序列中的先前位置（包括自身）。“掩码”操作防止关注未来位置，这在训练时是必要的，因为那时整个目标序列是可用的，但生成是按顺序进行的。
2. **多头交叉注意力（编码器-解码器注意力）：** 允许输出序列中的每个位置关注*输入*序列中的所有位置（特别是最终编码器层的输出）。这是输入序列的信息融入到输出生成中的地方。
3. **位置前馈网络（FFN）：** 结构上与编码器层中的FFN相同。

同样，残差连接和层归一化 (normalization)应用于每个子层之后。

> 单个Transformer解码器层的结构。虚线表示残差连接。请注意两种注意力类型和所需的掩码。

### 解码器中的掩码

解码器中通常使用两种类型的掩码：

1. **前瞻掩码：** 应用于*带掩码*的多头自注意力 (self-attention)层。这防止位置关注后续位置。对于长度为`T`的目标序列，它通常与填充掩码结合使用，生成形状为`(batch_size, 1, T, T)`的掩码。
2. **填充掩码：** 应用于*交叉注意力*（编码器-解码器注意力）层。这防止关注*编码器输出*中的填充标记 (token)。这种掩码通常具有类似`(batch_size, 1, 1, input_seq_len)`的形状。如果目标序列本身有填充，自注意力层也可能需要一个填充掩码，它将与前瞻掩码结合。

### 组装解码器层

该实现将这三个子层与残差连接、层归一化 (normalization)和Dropout结合起来。

```python
# 假设MultiHeadAttention和PositionWiseFeedForward已定义

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        """
        初始化一个Transformer解码器层。

        参数：
            d_model: 模型的维度。
            num_heads: 注意力头的数量。
            dff: 内层前馈网络的维度。
            rate: Dropout比率。
            **kwargs: 基层类的额外关键字参数。
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # 带掩码的自注意力
        self.mha1 = MultiHeadAttention(d_model, num_heads, name='masked_multi_head_attention')
        # 交叉注意力（编码器-解码器）
        self.mha2 = MultiHeadAttention(d_model, num_heads, name='cross_multi_head_attention')

        self.ffn = PositionWiseFeedForward(d_model, dff, name='position_wise_ffn')

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_1')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_2')
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm_3')

        self.dropout1 = tf.keras.layers.Dropout(rate, name='dropout_1')
        self.dropout2 = tf.keras.layers.Dropout(rate, name='dropout_2')
        self.dropout3 = tf.keras.layers.Dropout(rate, name='dropout_3')

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """
        解码器层的前向传播。

        参数：
            x: 输入张量（目标序列嵌入），形状为 (batch_size, target_seq_len, d_model)。
            enc_output: 来自编码器堆栈的输出，形状为 (batch_size, input_seq_len, d_model)。
            training: 布尔值，指示层是否应在训练模式下运行。
            look_ahead_mask: 用于自注意力的前瞻掩码和填充掩码的组合，
                             形状为 (batch_size, 1, target_seq_len, target_seq_len)。
            padding_mask: 用于交叉注意力中编码器输出的填充掩码，
                          形状为 (batch_size, 1, 1, input_seq_len)。

        返回：
            输出张量，形状为 (batch_size, target_seq_len, d_model)。
            自注意力权重（可选）。
            交叉注意力权重（可选）。
        """
        # 带掩码的多头自注意力模块
        # Q=K=V=x 用于自注意力
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        # 多头交叉注意力模块
        # Q=out1（来自解码器），K=V=enc_output（来自编码器）
        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2) # (batch_size, target_seq_len, d_model)

        # 位置前馈模块
        ffn_output = self.ffn(out2) # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output) # (batch_size, target_seq_len, d_model)

        # 返回输出和可选的注意力权重
        # 根据MHA实现，您可能只返回out3
        return out3 # 或者返回 out3, attn_weights_block1, attn_weights_block2

    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'rate': self.rate
        })
        return config
```

在这个`DecoderLayer`中，请注意`mha1`（自注意力 (self-attention)）如何使用`x`作为查询、键和值并应用`look_ahead_mask`，而`mha2`（交叉注意力）使用第一个子层的输出（`out1`）作为查询，将`enc_output`作为键和值，并应用对应于编码器输入的`padding_mask`。

## 堆叠层

一个完整的Transformer模型会堆叠多个编码器层（`N`次）和多个解码器层（`N`次）。最终编码器层的输出成为*每个*解码器层的交叉注意力机制 (attention mechanism)的`enc_output`输入。位置编码 (positional encoding)通常在输入嵌入 (embedding)进入第一个编码器或解码器层之前添加，以提供序列中标记 (token)位置的信息，因为自注意力 (self-attention)机制本身不能固有地捕获顺序。

通过使用TensorFlow的Keras API实现这些`EncoderLayer`和`DecoderLayer`组件，您将获得模块化和可重用的块。这些块可以组合起来构建完整的编码器和解码器堆栈，构成完整Transformer模型的支柱，适用于各种序列转换问题。接下来的步骤通常涉及添加输入嵌入、位置编码以及一个最终的线性层（以及用于分类/生成的softmax）来完成模型。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: arXiv; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构、多头注意力机制以及编码器-解码器结构的基础论文，这些是本节的核心内容。
- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Pearson
  一份全面的教材章节，详细解释了Transformer架构、注意力机制及其在自然语言处理中的应用。
- [Custom layers and models](https://www.tensorflow.org/guide/keras/custom_layers_and_models) — fchollet (2023)
  Publisher: tensorflow.org
  官方Keras指南，介绍如何通过子类化`tf.keras.layers.Layer`创建自定义层和模型，这对于构建Transformer块等自定义架构至关重要。

---

[上一节](01-%E4%BB%8E%E9%9B%B6%E6%9E%84%E5%BB%BA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md) · [下一节](03-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C%20%28GANs%29%20%E5%8E%9F%E7%90%86.md)
