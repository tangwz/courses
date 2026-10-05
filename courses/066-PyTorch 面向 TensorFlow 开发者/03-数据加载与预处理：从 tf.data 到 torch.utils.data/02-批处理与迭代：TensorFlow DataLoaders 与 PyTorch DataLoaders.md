# 批处理与迭代：TensorFlow DataLoaders 与 PyTorch DataLoaders

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/dataloaders-pytorch-tensorflow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 PyTorch `Dataset` 定义如何访问单个数据项，是模型训练期间高效地以批次形式加载和迭代数据的基础。作为 TensorFlow 开发者，您习惯于使用 `tf.data.Dataset.batch()` 及相关方法来为训练循环准备数据。PyTorch 为此提供了一个强大而灵活的类，名为 `torch.utils.data.DataLoader`。

`DataLoader` 封装了一个 `Dataset`，并提供了一个可迭代对象，产生数据批次。它还处理重要的功能，如混洗、使用多个工作进程进行并行数据加载以及自定义批次整理。这种关注点分离，即 `Dataset` 定义如何获取单个数据点而 `DataLoader` 定义如何对这些点进行分组和迭代，是 PyTorch 中的一种常见模式。

### 从 `tf.data.batch()` 到 `DataLoader`

在 TensorFlow 中，您通常通过直接在 `tf.data.Dataset` 对象上链式调用方法来创建批处理数据集：

```python
# TensorFlow tf.data 示例
import tensorflow as tf

# 虚拟特征和标签
features_tf = tf.random.uniform(shape=(100, 10))
labels_tf = tf.random.uniform(shape=(100, 1), maxval=2, dtype=tf.int32)

# 创建一个 tf.data.Dataset
tf_dataset = tf.data.Dataset.from_tensor_slices((features_tf, labels_tf))

# 混洗、批处理和预取
batched_tf_dataset = tf_dataset.shuffle(buffer_size=100).batch(32).prefetch(tf.data.AUTOTUNE)

# 迭代批次
# for x_batch, y_batch in batched_tf_dataset:
#     # 您的 TensorFlow 训练代码在这里
#     pass
```

在这个 TensorFlow 片段中，`shuffle()`、`batch()` 和 `prefetch()` 都是 `tf.data.Dataset` 类的方法，用于转换数据集。

PyTorch 处理方式不同。您首先定义 `Dataset`，然后将其传递给 `DataLoader` 实例：

```python
# PyTorch DataLoader 示例
import torch
from torch.utils.data import TensorDataset, DataLoader

# 虚拟特征和标签
features_pt = torch.randn(100, 10)
labels_pt = torch.randint(0, 2, (100, 1))

# 创建一个 PyTorch Dataset (TensorDataset 是张量数据的一种便捷方式)
pytorch_dataset = TensorDataset(features_pt, labels_pt)

# 创建一个 DataLoader
# 我们很快会更详细地讨论 num_workers
pytorch_loader = DataLoader(pytorch_dataset, batch_size=32, shuffle=True, num_workers=0)

# 迭代批次
# for x_batch, y_batch in pytorch_loader:
#     # 您的 PyTorch 训练代码在这里
#     # x_batch 的形状将是 [32, 10] (如果未丢弃，最后一个批次可能更小)
#     # y_batch 的形状将是 [32, 1]
#     pass
```

`DataLoader` 接收您的 `pytorch_dataset` 并内部处理批处理和混洗。

### DataLoader 参数 (parameter)

我们来看看 `DataLoader` 的重要参数以及它们与您的 TensorFlow 经验的关系：

- `dataset` (Dataset)：这是用于加载数据的 `Dataset` 对象。它等同于您开始使用的 `tf.data.Dataset` 对象。
- `batch_size` (int, 可选, 默认值=1)：每个批次加载多少样本。这直接类似于 `tf.data.Dataset.batch()` 中的 `batch_size` 参数。
- `shuffle` (bool, 可选, 默认值=False)：设置为 `True` 以在每个周期重新混洗数据。

  - **比较**：在 TensorFlow 中，您使用 `tf_dataset.shuffle(buffer_size=...)`。TensorFlow 混洗的效果取决于 `buffer_size`；为了获得完美的混洗，`buffer_size` 理想情况下应大于或等于数据集大小。PyTorch 的 `DataLoader` 在 `shuffle=True` 时（与映射风格的数据集一起使用时）通常会在每个周期之前混洗所有索引，从而实现完全混洗。如果您使用的是可迭代风格的数据集，混洗行为将取决于该数据集如何实现其迭代。
- `num_workers` (int, 可选, 默认值=0)：这是一个重要的性能参数。它指定用于数据加载的子进程数量。

  - `0` 表示数据将在主进程中加载。
  - 一个正整数值将生成相应数量的工作进程。这些工作进程在后台加载数据，通过将数据预处理和加载与 GPU 上的模型计算重叠进行，可以显著加快您的训练流水线。
  - **比较**：TensorFlow 的 `tf.data` API 主要通过 `dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)` 和 `dataset.prefetch(tf.data.AUTOTUNE)` 实现并行。`num_parallel_calls` 使映射函数并行化，而 `prefetch` 则使数据预处理和模型执行重叠。PyTorch 的 `num_workers` 直接控制数据*加载*步骤本身的并行性，每个工作进程获取一个批次（或整理成批次的样本）。适当设置 `num_workers` 可以充分利用用于数据加载的 CPU 内核，防止数据瓶颈。一个常见的起点是将 `num_workers` 设置为可用的 CPU 内核数量，但最佳值可能因数据集、转换和系统硬件而异。
  - *注意*：在 Windows 上，使用 `num_workers > 0` 可能需要您的主脚本执行受 `if __name__ == '__main__':` 的保护。

以下图表说明了 `DataLoader` 如何使用多个工作进程来准备批次：

> 当 `num_workers` > 0 时使用 `DataLoader` 进行数据加载。工作进程从 `Dataset` 中获取单个样本，这些样本随后通常由 `collate_fn` 分组为批次，并放入队列中供主训练循环使用。

- `pin_memory` (bool, 可选, 默认值=False)：如果为 `True`，`DataLoader` 在返回张量之前会将它们复制到 CUDA 固定（页锁定）内存中。这可以加快 CUDA 兼容 GPU 的 CPU 到 GPU 数据传输速度。TensorFlow 的数据流水线通常更隐式地管理内存和 GPU 传输。对于 PyTorch 中的 GPU 训练，如果您的数据适合，通常建议设置 `pin_memory=True`。
- `drop_last` (bool, 可选, 默认值=False)：如果为 `True`，如果数据集大小不能被 `batch_size` 完全整除，`DataLoader` 将丢弃最后一个批次。如果为 `False`，则最后一个批次可能小于 `batch_size`。这类似于 `tf.data.Dataset.batch(..., drop_remainder=True)` 中的 `drop_remainder=True`。
- `collate_fn` (可调用, 可选)：此函数用于合并样本列表以形成张量的迷你批次。当自动批处理（通常使用 `torch.stack`）不起作用时，它特别有用，例如，如果您的 `Dataset` 返回不同大小（如不同长度的序列）或复杂数据结构的样本时。

  - **比较**：在 TensorFlow 中，您可以使用 `tf.data.Dataset.padded_batch()` 或在批处理前通过 `dataset.map()` 转换中实现自定义填充逻辑来处理不同的序列长度。PyTorch 的 `collate_fn` 为这种自定义批次组装逻辑提供了一个集中位置。默认的 `collate_fn` 在大多数常见情况下效果良好，即样本已经是相同形状的张量或可以转换为张量的情况。

### 自定义 `collate_fn` 的实际应用

假设您的 `Dataset` 生成表示为 token ID 张量的单个句子，并且这些句子的长度不同。默认的 `collate_fn` 将会失败，因为它无法堆叠不同维度的张量。以下是您如何使用自定义 `collate_fn` 与 `torch.nn.utils.rnn.pad_sequence` 来处理这种情况的示例：

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence # 用于填充

# 返回可变长度序列（作为张量）的示例数据集
class VariableLengthSentenceDataset(Dataset):
    def __init__(self, list_of_sentences_as_ids):
        # 将句子存储为张量
        self.sentences = [torch.tensor(s, dtype=torch.long) for s in list_of_sentences_as_ids]

    def __getitem__(self, index):
        return self.sentences[index] # 返回一个 1D 张量

    def __len__(self):
        return len(self.sentences)

# 示例数据：列表的列表（句子的 token ID）
raw_data = [[10, 25, 3], [40, 52], [60, 77, 81, 99]]
sentence_dataset = VariableLengthSentenceDataset(raw_data)

# 自定义 collate_fn，用于批次内序列填充
def pad_collate_sentences(batch_of_sentence_tensors):
    # 'batch_of_sentence_tensors' 是 1D 张量（句子）的列表
    # pad_sequence 期望一个张量列表，并将其填充到列表中最长的长度
    # batch_first=True 使输出形状为 (batch_size, max_seq_length)
    # padding_value=0 对于 token ID 很常见，假设 0 是一个填充 token
    sequences_padded = pad_sequence(batch_of_sentence_tensors, batch_first=True, padding_value=0)

    # 如果您的模型需要长度，例如对于 PackedSequence，您也可以返回长度
    # lengths = torch.tensor([len(s) for s in batch_of_sentence_tensors])
    # return sequences_padded, lengths
    return sequences_padded

# 使用自定义 collate_fn 的 DataLoader
# batch_size=2，因此如果 drop_last=False，我们期望两个批次
custom_loader = DataLoader(sentence_dataset,
                           batch_size=2,
                           shuffle=False, # 保持顺序以便演示
                           collate_fn=pad_collate_sentences)

print("使用 custom_loader 迭代：")
for i, batch_data in enumerate(custom_loader):
    print(f"批次 {i+1}：")
    print("  数据（填充后的句子）：\n", batch_data)
    print("  形状：", batch_data.shape)

# 预期输出：
# 使用 custom_loader 迭代：
# 批次 1：
#   数据（填充后的句子）：
#  tensor([[10, 25,  3],
#          [40, 52,  0]])
#   形状：torch.Size([2, 3])
# 批次 2：
#   数据（填充后的句子）：
#  tensor([[60, 77, 81, 99]])
#   形状：torch.Size([1, 4])
```

在这个示例中，`pad_collate_sentences` 接收一个由单个句子张量组成的列表（这些张量在整理前构成一个批次），并使用 `pad_sequence` 通过填充较短的句子来确保结果批次张量中的所有句子具有相同的长度。

### 在训练循环中迭代

将 `DataLoader` 整合到 PyTorch 训练循环中非常直接。您直接迭代 `DataLoader` 对象，它会生成数据批次。

```python
# 假设模型、准则（损失函数）和优化器已定义
# 假设 train_dataset 是您的 PyTorch Dataset 实例
# 假设设备是 'cuda' 或 'cpu'

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
# num_epochs = 10

# for epoch in range(num_epochs):
#     model.train() # 将模型设置为训练模式
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         # 将数据移动到目标设备
#         inputs, labels = inputs.to(device), labels.to(device)

#         # 将参数梯度归零
#         optimizer.zero_grad()

#         # 前向传播
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)

#         # 反向传播和优化
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
```

这种结构应该让您感到熟悉。与 TensorFlow `tf.data` 循环的主要区别在于，批处理和迭代逻辑封装在 `DataLoader` 对象中，该对象是您预先创建的。

通过了解 `DataLoader` 及其参数 (parameter)，您可以在 PyTorch 中构建高效灵活的数据输入流水线。通过 `num_workers` 对并行加载的明确控制以及通过 `collate_fn` 进行自定义批处理是强大的功能，使您能够根据特定需求调整数据处理，通常在正确配置时会提高训练性能。

## 参考资料

- [torch.utils.data](https://pytorch.org/docs/stable/data.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch数据加载工具的官方文档，包括DataLoader和Dataset类。
- [tf.data: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data) — TensorFlow Authors (2024)
  TensorFlow tf.data API的官方指南，有助于理解其与PyTorch相比的批处理和预处理功能。
- [Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) — PyTorch Documentation (2024)
  Publisher: PyTorch
  一个PyTorch教程，展示了如何使用自定义数据集、DataLoader和数据转换，包括自定义collate_fn的示例。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本全面书籍，详细介绍了PyTorch数据加载机制，提供背景知识和实际应用。

---

[上一节](01-%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%EF%BC%9Atf.data.Dataset%20%E4%B8%8E%20torch.utils.data.Dataset.md) · [下一节](03-%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%EF%BC%9ATensorFlow%20%E6%96%B9%E6%B3%95%E4%B8%8E%20torchvision.transforms.md)
