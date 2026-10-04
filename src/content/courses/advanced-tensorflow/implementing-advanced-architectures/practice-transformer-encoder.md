---
course: "advanced-tensorflow"
chapter: "implementing-advanced-architectures"
lesson: "practice-transformer-encoder"
sourceId: 2952
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-7-implementing-advanced-architectures/practice-transformer-encoder"
title: "实践：实现Transformer编码器层"
description: "实现一个完整的Transformer编码器层，包括多头自注意力机制和前馈网络。"
order: 7
plots: []
sourceHash: "b16e80137e2725abbc0f64b703d465518074ed940b9c58a27289323a7e4a4763"
sourceCorrections: []
---

一个Transformer编码器层结合了Transformer架构的核心组件：多头自注意力 (self-attention)机制 (attention mechanism)和位置前馈网络。我们将使用TensorFlow的Keras API实现这个层。该层代表一个基本构建块，可以多次堆叠以形成Transformer模型的完整编码器部分。

请记住，单个编码器层执行两个主要操作：

1. 计算输入序列的自注意力。
2. 独立地对每个位置应用一个全连接前馈网络。

每个操作之后是残差连接和层归一化 (normalization)。层内还应用了Dropout用于正则化 (regularization)。

### 定义编码器层类

我们将通过继承`tf.keras.layers.Layer`来创建一个自定义Keras层。这使我们能够灵活地精确定义内部结构和前向传播计算。

```python
import tensorflow as tf

# 假设 MultiHeadAttention 和 PositionWiseFeedForwardNetwork
# 类已在前几节中定义。
# 为了完整性，这里是简化的占位符定义：

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.num_heads = num_heads
    # 简化：实际中包含用于 Q、K、V 和输出的 Dense 层
    print(f"占位符 MHA: d_model={d_model}, num_heads={num_heads}")

  def call(self, v, k, q, mask=None):
    # 简化：将输入查询作为注意力输出的占位符返回
    # 实际实现会计算缩放点积注意力
    print("占位符 MHA 调用")
    # 输出形状：(batch_size, seq_len_q, d_model)
    return q # Placeholder

class PositionWiseFeedForwardNetwork(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.dff = dff
    # 简化：实际中包含两个 Dense 层
    print(f"占位符 FFN: d_model={d_model}, dff={dff}")

  def call(self, x):
     # 简化：将输入作为占位符返回
     print("占位符 FFN 调用")
     # 输出形状：(batch_size, seq_len, d_model)
     return x # Placeholder

# --- 实际编码器层实现 ---

class TransformerEncoderLayer(tf.keras.layers.Layer):
  """
  实现一个具有多头注意力、前馈网络、层归一化和Dropout的单个Transformer编码器层。
  """
  def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
    """
    初始化Transformer编码器层。

    参数:
      d_model: 输入和输出的维度（嵌入维度）。
      num_heads: 注意力头的数量。
      dff: 前馈网络中内层的维度。
      dropout_rate: 介于0和1之间的浮点数。要丢弃的单元的比例。
    """
    super().__init__(**kwargs)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.dropout_rate = dropout_rate

    # 多头注意力子层
    self.mha = MultiHeadAttention(d_model, num_heads)

    # 位置前馈网络子层
    self.ffn = PositionWiseFeedForwardNetwork(d_model, dff)

    # 层归一化层
    # 为数值稳定性添加的epsilon
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    # Dropout层
    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, training, mask=None):
    """
    Transformer编码器层的前向传播。

    参数:
      x: 输入张量。形状：(batch_size, input_seq_len, d_model)
      training: 布尔值，指示层是应在训练模式下运行
                （应用dropout）还是推理模式下运行。
      mask: 注意力机制的可选掩码。

    返回:
      输出张量。形状：(batch_size, input_seq_len, d_model)
    """
    # 1. 多头注意力（带残差连接和归一化）
    # 自注意力：查询、键和值都是相同的输入 'x'
    attn_output = self.mha(x, x, x, mask)  # 形状：(batch_size, input_seq_len, d_model)

    # 对注意力输出应用dropout
    # Dropout仅在训练期间应用
    attn_output = self.dropout1(attn_output, training=training)

    # 添加残差连接并应用层归一化
    # out1 = x + attn_output
    out1 = self.layernorm1(x + attn_output) # 形状：(batch_size, input_seq_len, d_model)

    # 2. 前馈网络（带残差连接和归一化）
    ffn_output = self.ffn(out1) # 形状：(batch_size, input_seq_len, d_model)

    # 对FFN输出应用dropout
    ffn_output = self.dropout2(ffn_output, training=training)

    # 添加残差连接并应用层归一化
    # out2 = out1 + ffn_output
    out2 = self.layernorm2(out1 + ffn_output) # 形状：(batch_size, input_seq_len, d_model)

    return out2

  def get_config(self):
    """序列化层配置。"""
    config = super().get_config()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
        'dff': self.dff,
        'dropout_rate': self.dropout_rate
    })
    return config
```

在`__init__`方法中，我们实例化了必要的子层：`MultiHeadAttention`、`PositionWiseFeedForwardNetwork`、两个`LayerNormalization`层和两个`Dropout`层。超参数 (parameter) (hyperparameter)`d_model`、`num_heads`、`dff`和`dropout_rate`控制着层的行为和能力。

`call`方法定义了计算流程。

1. 输入`x`通过`MultiHeadAttention`层。由于这是编码器内的自注意力 (self-attention)，注意力机制 (attention mechanism)的查询、键和值输入都是相同的张量`x`。任何必要的填充掩码都会传递过去。
2. Dropout应用于注意力输出。请注意`training`参数的使用，它确保dropout仅在模型训练期间激活。
3. 添加一个残差连接（`x + attn_output`），之后是层归一化 (normalization)（`self.layernorm1`）。
4. 结果（`out1`）通过`PositionWiseFeedForwardNetwork`。
5. 再次应用Dropout（`self.dropout2`）。
6. 添加另一个残差连接（`out1 + ffn_output`），之后是第二个层归一化（`self.layernorm2`）。
7. 最终输出`out2`与输入`x`的形状相同。

我们还包含一个`get_config`方法，这是自定义Keras层的一个好习惯，使得层可以轻松地保存和加载。

### 编码器层结构的可视化

以下图表说明了`TransformerEncoderLayer`内的数据流。

> 单个Transformer编码器层内的数据流，展示了多头注意力 (multi-head attention)和前馈子层，每个都跟着Dropout、残差连接（加）和层归一化 (normalization)。

### 实例化并测试层

让我们创建`TransformerEncoderLayer`的一个实例，并通过它传递一些示例数据，以验证其操作和输出形状。

```python
# 定义超参数
batch_size = 64
input_seq_len = 50
d_model = 512  # 嵌入维度
num_heads = 8  # 注意力头的数量
dff = 2048     # FFN中的隐藏层大小
dropout_rate = 0.1

# 创建一个示例输入张量（例如，序列嵌入）
# 在实际场景中替换为真实数据
sample_input = tf.random.uniform((batch_size, input_seq_len, d_model))

# 实例化编码器层
# 注意：需要使用实际的 MHA 和 FFN 实现才能获得真实结果
# 上面使用的占位符版本只会打印消息并传递数据。
# 假设您有可用的功能性 MHA 和 FFN 类：
# encoder_layer = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)

# 用于占位符演示：
print("使用占位符实例化编码器层:")
encoder_layer = TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate, name="my_encoder_layer")
print("-" * 20)

# 通过层传递示例输入
# 设置 training=False 为推理模式（无dropout）
print("运行编码器层调用 (training=False):")
output_tensor = encoder_layer(sample_input, training=False)
print("-" * 20)

# 在训练模式下通过层传递示例输入
print("运行编码器层调用 (training=True):")
output_tensor_train = encoder_layer(sample_input, training=True)
print("-" * 20)

# 检查输出形状
print(f"输入形状: {sample_input.shape}")
print(f"输出形状 (推理): {output_tensor.shape}")
print(f"输出形状 (训练): {output_tensor_train.shape}")

# 验证输出形状是否与输入形状匹配（可能不包括批次大小）
assert sample_input.shape == output_tensor.shape
assert sample_input.shape == output_tensor_train.shape

print("\n编码器层创建并测试成功。")
# 您还可以检查层的配置
print("\n层配置:")
print(encoder_layer.get_config())
```

运行此代码（假设有功能性的`MultiHeadAttention`和`PositionWiseFeedForwardNetwork`类）将实例化编码器层并处理示例输入。输出形状应与输入形状`(batch_size, input_seq_len, d_model)`匹配，确认该层在处理序列的同时，保持了堆叠多层所需的维度。如果使用前面显示的简化版本，您还会看到占位符消息。

这个实践实现提供了对不同组件如何在Transformer编码器层内集成的具体理解。通常，一个完整的Transformer编码器由该层的多个实例按顺序堆叠而成，一个层的输出会作为下一个层的输入。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的奠基性论文，详细阐述了多头注意力、前馈网络、残差连接和层归一化等架构组件，这些是编码器层实现的基础。
- [Writing custom layers and models](https://www.tensorflow.org/guide/keras/custom_layers_and_models) — TensorFlow Developers (2024)
  Publisher: TensorFlow
  TensorFlow Keras关于如何通过继承`tf.keras.layers.Layer`创建自定义层的官方指南，这对于实现TransformerEncoderLayer是至关重要的。
- [Deep Learning with Python](https://www.oreilly.com/library/view/deep-learning-with/9781617296864/) — François Chollet (2021)
  Publisher: O'Reilly Media
  一本关于使用 Keras 和 TensorFlow 进行深度学习的实用教程，其中阐述了 Transformer 模型及其实现细节，并为文中代码提供了更全面的背景。
