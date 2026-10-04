---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-data-loading-for-tf-users"
lesson: "efficient-data-pipelines-pytorch"
sourceId: 5756
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/efficient-data-pipelines-pytorch"
title: "在 PyTorch 中构建高效数据管道"
description: "在 PyTorch 中创建高性能数据加载管道的方法，包括并行加载。"
order: 6
plots: []
sourceHash: "99383c21e30cd0dbb6dd115c0d599c2f05347a0dd386c394af22975af9420ecf"
sourceCorrections: []
---

PyTorch 的 `Dataset` 用于封装数据，`DataLoader` 用于批量迭代数据。尽管这些工具提供了坚实的基础，但数据管道的效率会显著影响整体训练速度。慢速的数据管道可能导致 GPU 等待数据，从而未能充分利用其计算能力。在此，讨论了在 PyTorch 中构建高性能数据加载管道的各种方法，并结合 `tf.data` 中相关的优化技巧进行比较。

### 使用 `num_workers` 进行并行数据加载

训练中最常见的一个瓶颈是数据加载和预处理步骤。如果此过程在主训练程序中按顺序进行，您的 CPU 可能难以跟上 GPU 的速度，导致 GPU 闲置。PyTorch 的 `DataLoader` 提供了一个直接的解决方案：使用多个工作进程进行并行数据加载。

`DataLoader` 构造函数中的 `num_workers` 参数 (parameter)指定了将生成多少个独立进程来加载和预处理数据。当 `num_workers > 0` 时，数据获取和转换会在这些后台进程中执行。这使得主训练循环可以接收预先获取的数据批次，这些批次已为 GPU 准备就绪，从而最大程度地减少延迟。

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# 示例数据
data = torch.randn(1000, 3, 32, 32)
labels = torch.randint(0, 10, (1000,))
dataset = TensorDataset(data, labels)

# 使用多个工作进程的 DataLoader
# 仅作演示，实际最佳 num_workers 值取决于您的系统
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4, # 工作进程数量
    pin_memory=True # 稍后解释
)

# 在您的训练循环中：
# for batch_data, batch_labels in train_loader:
#     # 数据已预取并准备就绪
#     # 如有必要，将其移至 GPU 并继续训练
#     pass
```

**选择 `num_workers`**:
`num_workers` 的最佳值取决于您的 CPU、磁盘速度、批量大小以及数据转换的复杂程度。

- 一个常见的起点是可用 CPU 核心的数量。
- 设置 `num_workers = 0`（默认值）表示数据加载在主进程中进行。
- 过多的工作进程可能引入进程间通信开销或耗尽系统资源，从而可能降低速度。通常需要进行尝试。请监控您的 CPU 和 GPU 使用率以找到一个良好平衡。

如果您熟悉 `tf.data`，PyTorch `DataLoader` 中的 `num_workers` 与 `dataset.map()` 操作中设置 `num_parallel_calls` 或依赖 `tf.data.AUTOTUNE` 在 TensorFlow 中动态调整并行性具有类似的作用。

### 使用 `pin_memory` 加速 CPU 到 GPU 的数据传输

在 GPU 上训练时，`DataLoader` 加载的数据（通常位于标准 CPU 可分页内存中）需要传输到 GPU 的内存中。如果 CPU 内存是“锁页”（也称为页锁定内存），此传输可以更快。

在 `DataLoader` 构造函数中设置 `pin_memory=True` 会指示 PyTorch 将 `DataLoader` 返回的张量分配在锁页内存中。这使得使用 CUDA 可以更快地将数据异步传输到 GPU。

```python
# 使用 pin_memory=True 的 DataLoader
gpu_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True # 启用锁页内存
)

# 在您的训练循环中（假设 CUDA 可用）：
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for batch_data, batch_labels in gpu_loader:
#     batch_data = batch_data.to(device, non_blocking=True) # 非阻塞传输
#     batch_labels = batch_labels.to(device, non_blocking=True)
#     # ...其余的训练步骤
```

使用 `pin_memory=True` 时，您还可以在 `.to(device)` 调用中设置 `non_blocking=True`。这使得 CPU 可以在数据传输在后台进行时继续执行其他操作，从而可能重叠计算和数据传输。

**注意**：过度使用锁页内存会减少其他应用程序或操作系统可用的可分页内存量，如果您的 RAM 有限，这可能导致性能问题。当您的数据加载确实是瓶颈并且您正在 GPU 上训练时，它最有益。

### 优化批量大小

`DataLoader` 中的 `batch_size` 参数 (parameter)是一个基本超参数 (hyperparameter)。虽然它并非严格意义上的 `DataLoader` 优化功能，但它的选择会显著影响管道效率和训练动态：

- **更大的批量大小**可以带来更高的吞吐量 (throughput)，因为它们能更好地利用 GPU 的并行处理能力。对更大的张量进行操作通常效率更高。
- 然而，更大的批量大小需要更多的 GPU 内存。
- 非常大的批量大小有时可能导致泛化能力下降，尽管这取决于具体问题。

寻找最佳批量大小通常需要进行尝试，以平衡 GPU 内存限制与训练速度和模型性能。

### 高效数据增强和预处理

数据转换（包括增强）通常在 `Dataset` 的 `__getitem__` 方法中定义，或者通过 `transform` 参数 (parameter)使用 `torchvision.transforms.Compose` 应用。

- **在 CPU 工作进程上执行转换**：当 `num_workers > 0` 时，这些转换由工作进程并行应用。这通常是推荐的方法。
- **转换的复杂性**：即使使用多个工作进程，计算量非常大的转换仍可能减慢数据加载速度。如果您怀疑它们是瓶颈，请对您的转换进行性能分析。
- **转换的顺序**：某些转换如果按特定顺序应用，可能会更有效。例如，将图像转换为张量通常应在大多数图像处理转换之后进行。
- **GPU 加速转换**：对于某些特定且计算密集型的增强，NVIDIA DALI 或 Kornia 等库提供了 GPU 加速转换。这些是更高级的工具，通常在基于 CPU 的增强成为严重瓶颈时使用。

TensorFlow 用户会发现这与在 `tf.data` 中使用预处理函数应用 `map` 操作类似，其中并行性由 `num_parallel_calls` 处理。

### 管理磁盘 I/O

如果您的数据集很大并且每个 epoch 都直接从磁盘读取，那么磁盘 I/O 可能成为一个重要瓶颈，尤其是在使用较慢的硬盘时。

- **使用 SSD**：固态硬盘比传统机械硬盘提供快得多的读写速度。
- **将数据加载到内存中**：如果您的数据集足够小，可以完全放入 RAM，那么在开始时将其完全加载到内存中可以消除训练期间的磁盘 I/O。您的 `Dataset` 将直接从 RAM 访问数据。
- **高效的文件格式**：对于非常大的数据集，请考虑使用针对快速随机访问或大读取进行优化的文件格式，例如 HDF5、LMDB 或 WebDataset。您可以将原始数据预处理成这些格式一次。
- **顺序读取**：如果可能，请设计您的数据访问模式使其尽可能顺序化，因为这通常比从磁盘进行随机读取更快。

### 优化自定义 `collate_fn`

有时，`DataLoader` 提供的默认整理函数（它会堆叠张量）不适用，例如处理长度可变的序列时。在这种情况下，您需要提供一个自定义的 `collate_fn`。
一个编写效率不高的 `collate_fn` 可能成为瓶颈，因为它在工作进程获取数据后（如果 `num_workers > 0` 且整理发生在主进程中）或在每个工作进程返回批次之前（如果整理是工作进程任务的一部分）运行。

- 使您的 `collate_fn` 尽可能简单快速。
- 在 `collate_fn` 中进行填充或其他操作时，优先使用 PyTorch 张量操作而不是 Python 循环。

```python
# 变长序列自定义 collate_fn 的示例草图
def custom_collate_fn(batch):
    # 批次是 (序列, 标签) 元组的列表
    sequences, labels = zip(*batch)
    # 将序列填充到此批次中的最大长度
    # 使用 torch.nn.utils.rnn.pad_sequence 提高效率
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded_sequences, labels

# train_loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn, num_workers=2)
```

### 监控和识别瓶颈

为了有效优化数据管道，您首先需要判断它是否确实是一个瓶颈。

- **监控 CPU 和 GPU 使用率**：
  - 使用 `htop` 或任务管理器等工具检查 CPU 使用情况。如果 CPU 占用率很高而 GPU 经常空闲或使用率很低，您的数据管道很可能太慢了。
  - 使用 `nvidia-smi`（适用于 NVIDIA GPU）检查 GPU 使用率。
- **PyTorch Profiler**：PyTorch 包含一个内置的性能分析器 (`torch.profiler`)，它可以帮助找出数据加载与模型计算所花费的时间。我们将在第 6 章中更详细地了解这一点。一个简单的检查方法是计时在不执行任何模型训练步骤的情况下，遍历 `DataLoader` 一个 epoch 需要多长时间。

> CPU/GPU 使用模式和潜在解释的简化视图。高 CPU 和低 GPU 使用率通常表明数据管道存在瓶颈。

构建高效数据管道涉及这些方法的组合。最有影响力的设置通常是 `num_workers` 和 `pin_memory`，但其他因素也可能发挥作用，具体取决于您的数据集和硬件。务必进行测量和性能分析，以指导您的优化工作。通过确保数据快速馈送到模型，您可以让训练按照计算硬件的速度进行，而不是受限于数据访问。

## 参考资料

- [\`torch.utils.data\` API Reference](https://pytorch.org/docs/stable/data.html) — PyTorch Developers (2025)
  提供关于`Dataset`、`DataLoader`、`num_workers`、`pin_memory`和`collate_fn`的详细信息，以实现高效数据处理。
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — Szymon Migacz (2020)
  Publisher: PyTorch
  提供优化 PyTorch 模型训练的实用策略，包括数据加载和 GPU 利用的最佳实践。
- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Developers (2023)
  详细介绍了如何使用 TensorFlow 的 `tf.data` API 构建高性能输入管道，对从 TensorFlow 过渡的开发人员很有价值。
