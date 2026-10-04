---
course: "rnns-and-sequence-modeling"
chapter: "evaluating-tuning-sequence-models"
lesson: "regularization-techniques-rnns"
sourceId: 2650
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-10-evaluating-tuning-sequence-models/regularization-techniques-rnns"
title: "RNN的正则化方法"
description: "应用Dropout等正则化方法（包括循环Dropout变体）以防止过拟合。"
order: 6
plots: []
sourceHash: "1d31e75ebab1c14b0b8302ef57e57e98a11ab7bd31714a6fbed7d8bf5a739784"
sourceCorrections: []
---

与前馈网络类似，循环神经网络 (neural network) (RNN)，特别是更复杂的LSTM和GRU，可能出现过拟合 (overfitting)。当模型过度拟合训练数据，包括其中的噪声和特定模式，而无法应用于新的、未见过的数据时，就会发生过拟合。这通常表现为在训练集上准确率高，但在验证集或测试集上显著降低。考虑到序列数据有时可能有限，并且RNN的参数 (parameter)随时间步演变，它们特别容易记住训练序列，而不是学习一般的时间模式。

正则化 (regularization)方法旨在通过对模型或其训练过程添加限制来对抗过拟合，促使其学习更简单、更具普适性的模式。神经网络（包括RNN）中最常用且有效的一种正则化方法是 **Dropout**。

### 理解Dropout

Dropout背后的主要思想出人意料地简单。在每次训练迭代中，Dropout随机将层中一部分神经元（或单元）的输出设为零。要丢弃的神经元比例由 *dropout率* 控制，这是一个通常设在0.1到0.5之间的超参数 (parameter) (hyperparameter)。

通过随机“丢弃”单元，网络无法过度依赖任何单个神经元或一小群神经元共同适应以学习特定特征。这迫使网络学习冗余表示，并将学习分布到更多单元上，从而使学到的特征更可靠，对单个神经元的特定权重 (weight)不那么敏感。可以将其视为同时训练许多不同的稀疏网络并对它们的预测进行平均。

### 在RNN中应用Dropout：难题

虽然标准Dropout对前馈层效果良好，但在RNN的循环连接中直接应用它会带来问题。请记住，在时间步 $t$ 的隐藏状态 $h_t$ 是根据之前的隐藏状态 $h_{t-1}$ 和当前输入 $x_t$ 计算的。

如果你将标准Dropout应用于循环连接（即从 $h_{t-1}$ 到 $h_t$ 的连接），那么在每个时间步， $h_{t-1}$ 中可能都会有 *不同* 的单元被丢弃。这种有效循环连接的持续变化会严重干扰网络传播信息和在长序列上保持记忆的能力。这就像你试图进行一场对话，但你的短期记忆中随机的词汇在每一步都被不断抹去。这可能会阻止RNN、LSTM或GRU学习有意义的时间依赖关系。

### 循环Dropout：正确应用Dropout

为了有效地将Dropout应用于循环连接，而不妨碍时间动态的学习，通常使用一种被称为 **循环Dropout**（具体来说，是一种 *变分Dropout* 形式）的方法。

主要思想是在给定训练序列内的所有时间步中，对循环连接使用 **相同的Dropout掩码**。这意味着如果一个特定的循环单元连接在时间步 $t$ 被丢弃（设为零），那么在该序列的 *整个* 前向和反向传播 (backpropagation)过程中，它在时间步 $t+1, t+2, \dots$ 也被丢弃。对于下一个训练序列或批次，会生成一个新的Dropout掩码，并一致地应用于其所有时间步。

> 比较了应用于循环连接的标准Dropout（有问题）和循环（变分）Dropout（优选）之间的差异。循环Dropout在给定序列的时间步中应用一致的掩码。

这种一致性使得网络能够适当学习时间依赖性，同时仍能受益于Dropout的正则化 (regularization)效果。标准Dropout仍可应用于 *非循环* 连接，例如从输入 $x_t$ 到隐藏状态 $h_t$ 的连接，或从隐藏状态 $h_t$ 到输出层的连接，而不会引起同样的问题。

### 框架中的实现

TensorFlow（Keras API）和PyTorch等深度学习 (deep learning)框架提供了在LSTM和GRU层中实现标准Dropout和循环Dropout的方便方式。

**TensorFlow/Keras：**
`LSTM` 和 `GRU` 层通常有两个单独的参数 (parameter)：

- `dropout`：指定输入连接（从 $x_t$ 到 $h_t$）的Dropout率。
- `recurrent_dropout`：指定循环连接（从 $h_{t-1}$ 到 $h_t$）的Dropout率，实现上述变分Dropout方法。

```python
# TensorFlow/Keras使用示例
import tensorflow as tf

# 对输入应用20%的Dropout，对循环连接应用30%的循环Dropout
lstm_layer = tf.keras.layers.LSTM(
    units=64,
    dropout=0.2,
    recurrent_dropout=0.3,
    return_sequences=True
)

# 在模型中：
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=16, mask_zero=True),
    tf.keras.layers.LSTM(units=64, dropout=0.2, recurrent_dropout=0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Dropout在model.fit()期间自动应用，在model.predict()期间禁用
```

**PyTorch：**
PyTorch中的 `LSTM` 和 `GRU` 模块使用单个 `dropout` 参数。如果 `num_layers` 大于1，则此参数会将标准Dropout应用于每个层（*除了最后一层*）的输出。它 *不会* 像Keras的 `recurrent_dropout` 那样自动将变分Dropout应用于每个层内的循环连接。实现真正的变分Dropout通常需要自定义实现或第三方库，尽管在堆叠的RNN层之间使用标准Dropout是常见做法。

```python
# PyTorch使用示例（如果层堆叠，则在层间应用Dropout）
import torch
import torch.nn as nn

# 如果num_layers > 1，Dropout应用于层间
lstm_layer = nn.LSTM(
    input_size=16,
    hidden_size=64,
    num_layers=2, # 设置 > 1 以在层间启用Dropout
    dropout=0.3,   # 堆叠LSTM层之间的Dropout率
    batch_first=True
)

# model.train() 启用Dropout
# model.eval() 禁用Dropout
```

**重要提示：** Dropout应仅在 *训练* 阶段激活。在评估或预测（推理 (inference)）期间，应使用完整的网络（不丢弃单元）。当你调用训练函数（如Keras中的 `model.fit()` 或PyTorch中的 `model.train()`）与评估/预测函数（Keras中的 `model.evaluate()`、`model.predict()` 或PyTorch中的 `model.eval()`）时，深度学习框架会自动处理此切换。

### 选择Dropout率

最佳的Dropout率（包括标准和循环Dropout，如果适用）是需要调整的超参数 (parameter) (hyperparameter)。

- 常用值范围为0.1到0.5。
- 从小值（例如0.1或0.2）开始，如果过拟合 (overfitting)持续存在，则增加。
- 监测验证集上的性能。如果验证性能开始下降而训练性能持续提高，则可能发生过拟合，此时增加Dropout（或应用其他正则化 (regularization)方法）可能有所帮助。
- 如果训练和验证性能都很差或过早停滞，则Dropout率可能过高，阻碍了模型的学习能力（欠拟合 (underfitting)）。

### 其他正则化 (regularization)方法

虽然Dropout在RNN中非常普遍，但也可以使用其他方法，有时会结合使用：

- **权重 (weight)正则化（L1/L2）：** 根据网络权重的量级向损失函数 (loss function)添加惩罚项（L1促使稀疏性，L2促使权重变小）。在Keras中，这些可以通过RNN层中的 `kernel_regularizer`、`recurrent_regularizer` 和 `bias_regularizer` 参数 (parameter)应用。
- **早停：** 在训练期间监测验证损失或指标，并在验证性能停止提高或开始下降时停止训练过程。这可以阻止模型在后续周期中继续过拟合 (overfitting)训练数据。这种方法经常与其他正则化方法一起使用。

在训练LSTM和GRU时，应用适当的正则化，特别是循环Dropout，是防止过拟合并提高其对新序列数据泛化能力的标准做法。调整Dropout率是本章前面讨论的超参数 (hyperparameter)优化过程中的一个重要部分。

## 参考资料

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958
  介绍神经网络dropout正则化技术的原始学术论文。
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287) — Yarin Gal, Zoubin Ghahramani (2016)
  Journal: Advances in Neural Information Processing Systems 29; DOI: [10.48550/arXiv.1512.05287](https://doi.org/10.48550/arXiv.1512.05287)
  介绍变分dropout，为在RNN中将dropout应用于循环连接提供了一种不干扰时间动态的方法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  涵盖深度学习基本概念的权威教科书，包括正则化技术和循环神经网络。
- [tf.keras.layers.LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) — TensorFlow Team (2024)
  Keras LSTM层的官方文档，详细说明了`dropout`和`recurrent_dropout`等实现参数。
