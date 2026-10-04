---
course: "rnns-and-sequence-modeling"
chapter: "preparing-sequence-data"
lesson: "batching-sequential-data"
sourceId: 2624
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-8-preparing-sequence-data/batching-sequential-data"
title: "序列数据批处理"
description: "关于如何创建适用于循环神经网络训练的序列数据批次的方法。"
order: 8
plots: []
sourceHash: "d2cd58e1b64f1d2a07266fe3c3a481ec424de9be14f8c8847562245223934cba"
sourceCorrections: []
---

在使用分词 (tokenization)、整数编码以及可能的嵌入 (embedding)查找等方法将原始序列转换为数值表示，并通过填充和掩码处理确保统一性之后，下一步是将这些处理过的序列分组，形成批次。深度学习 (deep learning)模型通常以数据批次而非单个样本进行训练，这有几个原因：

1. **计算效率：** 批处理数据让GPU等硬件加速器更有效地并行计算，大幅提升训练速度。
2. **梯度稳定性：** 对一个批次的数据计算损失和梯度，相比于单个样本，能更准确地估算整个数据集的真实梯度。这使得模型权重 (weight)在训练期间的更新更稳定可靠。

### 序列数据批次结构

循环模型的输入数据通常期望为特定的三维张量格式。尽管具体顺序在不同框架（如TensorFlow或PyTorch）中可能略有差异，但常见结构为：

`(batch_size, time_steps, num_features)`

让我们分解说明：

- `batch_size`: 此批次中包含的序列数量。
- `time_steps`: 批次中序列的长度。很重要的一点是，由于填充处理，这个维度将等于该批次中最长序列的长度。所有较短的序列都会被填充到这个长度。
- `num_features`: 每个时间步表示的维度。
  - 如果您将整数编码的序列直接作为模型的第一层输入到嵌入 (embedding)层，`num_features` 将是 1（即整数ID）。
  - 如果您已在RNN之前通过嵌入层处理了整数，或者您的输入本身是多维的（如多元时间序列），`num_features` 将是嵌入向量 (vector)的维度或每个时间步的输入特征数量。

假设您有三个经过整数编码和填充（其中 0 是填充值）的序列：

- 序列 A (原始长度 3): `[10, 25, 31, 0, 0]`
- 序列 B (原始长度 5): `[15, 8, 99, 50, 2]`
- 序列 C (原始长度 4): `[5, 12, 18, 77, 0]`

如果这些序列形成一个批次（`batch_size = 3`），`time_steps` 维度必须适应最长的序列（序列 B，长度 5）。生成的输入批次张量（假设整数ID的`num_features = 1`）将如下所示：

```
[
  [[10], [25], [31], [ 0], [ 0]],  // 序列 A (已填充)
  [[15], [ 8], [99], [50], [ 2]],  // 序列 B
  [[ 5], [12], [18], [77], [ 0]]   // 序列 C (已填充)
]
```

此张量的形状为 $(3, 5, 1)$。

### 批次中掩码的作用

还记得我们在填充步骤中生成的掩码吗？它们在处理批次时变得非常重要。循环神经网络 (neural network) (RNN)层（以及后续层或损失函数 (loss function)）需要知道批次张量中的哪些元素对应真实数据，哪些只是填充。

以上述示例来说，对应的掩码张量将表示真实数据（1）与填充（0）：

```
[
  [1, 1, 1, 0, 0],  // 序列 A 的掩码
  [1, 1, 1, 1, 1],  // 序列 B 的掩码
  [1, 1, 1, 1, 0]   // 序列 C 的掩码
]
```

这个掩码张量，通常形状为 `(批次大小, 时间步)`，经常与输入批次一同传递给循环神经网络层，或在损失计算中使用。循环神经网络层的框架API通常有特定参数 (parameter)（例如 Keras/TensorFlow 中的 `mask`，或 PyTorch 中使用 packed sequences 隐式处理）来处理这一点，以确保与填充时间步相关的计算被忽略。

### 数据加载工具

手动创建这些填充后的批次和掩码可能很繁琐。幸运的是，深度学习 (deep learning)框架提供了高级工具来简化此过程：

- **TensorFlow:** `tf.data.Dataset` API 效率很高。您可以创建包含可变长度序列的数据集，然后使用 `padded_batch` 等方法。此方法会自动将元素分组为批次，确定每个批次中的最大长度，将该批次中的所有序列填充到该长度，并且通常可以隐式处理下游层的掩码。
- **PyTorch:** 使用 `torch.utils.data.Dataset` 和 `DataLoader` 类。您通常定义一个自定义 `Dataset` 来加载单个序列。`DataLoader` 处理批处理。要在批次内实现填充，您通常会向 `DataLoader` 提供一个自定义 `collate_fn`。此函数接受样本列表（您的序列），并手动填充它们（例如，使用 `torch.nn.utils.rnn.pad_sequence`）以形成批次张量，并可能生成掩码。

以下是根据可变长度序列创建批次的图示：

> 可变长度序列批处理流程。原始序列在堆叠成批次张量之前，会被填充到组内的最大长度（本例中为5）。相应的掩码标识出原始数据点。

### 批处理注意事项

- **批次大小：** 这是一个重要的超参数 (parameter) (hyperparameter)。较大的批次大小可以加快训练步骤（更好的硬件利用率）并获得更稳定的梯度，但会消耗更多内存。非常大的批次也可能平均需要更多的填充，略微降低每个样本的计算效率。较小的批次需要更少内存，有时可以帮助模型更好地泛化（因为梯度噪声更大），但训练时间更长。典型的批次大小范围为 16 到 256，具体取决于任务、模型大小和可用内存。
- **序列长度变异：** 如果您的数据集包含长度差异很大的序列（例如，句子长度从 3 个词到 100 个词不等），简单的批处理可能因过度填充而效率低下。一种常见于数据加载器中的方法是大致按序列长度排序数据集，然后从长度相似的序列创建批次。这可以最大限度地减少每个批次内的填充。然而，要确保在不同周期（epochs）之间充分打乱数据，以避免在训练过程中引入偏差。

通过正确地对填充和掩码处理后的序列进行批处理，您可以将数据准备成精确的格式，从而高效且有效地训练使用现代深度学习 (deep learning)库的循环神经网络 (neural network) (RNN)。这一步连接了预处理后的单个序列与模型训练循环的输入要求。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，讨论用于优化的迷你批次梯度下降，并介绍与RNN数据准备相关的序列建模概念。
- [Build TensorFlow input pipelines with tf.data](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2023)
  TensorFlow 中创建高效数据输入管道的官方指南，包括用于处理变长序列数据的 `padded_batch` 等方法。
- [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Sasank Chilamkurthy (2024)
  Publisher: PyTorch
  一个官方 PyTorch 教程，演示使用 `Dataset` 和 `DataLoader` 进行数据处理，包括自定义 `collate_fn` 用于批处理和填充变长序列。
