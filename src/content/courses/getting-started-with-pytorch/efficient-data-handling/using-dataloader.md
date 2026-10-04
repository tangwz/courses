---
course: "getting-started-with-pytorch"
chapter: "efficient-data-handling"
lesson: "using-dataloader"
sourceId: 2182
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/using-dataloader"
title: "使用 `torch.utils.data.DataLoader`"
description: "使用 `DataLoader` 实现自动批处理、数据打乱和并行数据加载。"
order: 5
plots: []
sourceHash: "22f3e6841d800da9c814b6d2cd773df087a2e4ed7c35d987af0da7a85334dd39"
sourceCorrections: []
---

尽管 `Dataset` 类提供了一种清晰的方式来抽象对单个数据样本的访问，但对大型数据集逐个样本进行迭代对于训练深度学习 (deep learning)模型通常效率不高。训练通常受益于分批处理数据。在这种情况下，`torch.utils.data.DataLoader` 在高效数据处理中发挥着核心作用。

`DataLoader` 封装了一个 `Dataset`（无论是内置的还是你自定义的实现）并提供了对其的迭代接口。它的主要职责是：

1. **分批处理**：将单个样本分组为小批量。
2. **打乱数据**：在每个周期随机打乱数据，以防止模型学习到样本的顺序并提高泛化能力。
3. **并行数据加载**：使用多个子进程并发加载数据，这可能会显著加快数据处理流程。

### 基本用法和迭代

创建 `DataLoader` 很简单。你主要需要提供 `Dataset` 实例并指定所需的 `batch_size`。

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 假设 'YourCustomDataset' 已如前所示定义
# 或者使用内置数据集，例如 datasets.MNIST
# 为了演示，我们创建一个简单的虚拟数据集：
class DummyDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.features = torch.randn(num_samples, 10) # 示例：100 个样本，10 个特征
        self.labels = torch.randint(0, 2, (num_samples,)) # 示例：100 个二元标签

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 实例化数据集
dataset = DummyDataset(num_samples=105)

# 实例化 DataLoader
# batch_size: 每个批次的样本数量
# shuffle: 设置为 True 以在每个周期打乱数据（对训练很重要）
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# 迭代 DataLoader
print(f"Dataset size: {len(dataset)}")
print(f"DataLoader batch size: {train_loader.batch_size}")

for epoch in range(1): # 一个周期的示例
    print(f"\n--- Epoch {epoch+1} ---")
    for i, batch in enumerate(train_loader):
        # DataLoader 产生批次。每个 'batch' 通常是元组或列表
        # 包含特征和标签的张量。
        features, labels = batch
        print(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")
        # 在这里你通常会执行训练步骤：
        # model.train()
        # optimizer.zero_grad()
        # outputs = model(features)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
```

运行此代码将展示 `DataLoader` 如何产生数据批次。注意每个批次打印的形状反映了 `batch_size`（除了可能的最后一个批次）。

### 批次策略和 `drop_last`

默认情况下，如果 `Dataset` 中的总样本数不能被 `batch_size` 完全整除，最后一个批次将包含剩余样本，因此会更小。

在我们有 105 个样本、批次大小为 32 的示例中：

- 批次 1: 32 个样本
- 批次 2: 32 个样本
- 批次 3: 32 个样本
- 批次 4: 9 个样本 (105 - 3\*32 = 9)

有时，拥有可变批次大小，尤其是非常小的最后一个批次，可能会影响某些训练动态或特定层的要求（例如训练期间的 BatchNorm 层，尽管 PyTorch 处理得相当好）。如果你希望所有批次都具有精确的 `batch_size`，并丢弃较小的最后一个批次，你可以在创建 `DataLoader` 时设置 `drop_last=True`：

```python
# 如果数据集大小不能被批次大小整除，则丢弃最后一个不完整的批次
train_loader_drop_last = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=True)

print("\n--- DataLoader with drop_last=True ---")
for i, batch in enumerate(train_loader_drop_last):
    features, labels = batch
    print(f"Batch {i+1}: Features shape={features.shape}, Labels shape={labels.shape}")

# 预期输出：只有 3 个大小为 32 的批次。最后的 9 个样本被丢弃。
```

### 打乱数据以获得更好的训练

在训练期间强烈建议设置 `shuffle=True`。它告诉 `DataLoader` 在为每个周期创建批次之前重新打乱数据集的索引。这确保模型每次都以不同的顺序查看数据，减少过度拟合数据呈现顺序的风险并提高模型健壮性。对于验证或测试，通常禁用打乱数据（`shuffle=False`），以确保不同运行之间评估指标的一致性。

### 使用 `num_workers` 并行加载数据

数据加载和预处理（应用变换）有时可能会成为瓶颈，特别是当变换很复杂或数据读取涉及大量 I/O 操作时。CPU 可能会花费大量时间准备下一个批次，而 GPU 则空闲等待数据。

`DataLoader` 允许你通过使用多个工作进程并行加载数据来缓解这个问题。你可以使用 `num_workers` 参数 (parameter)来指定工作进程的数量：

```python
# 使用 4 个工作进程加载数据
# num_workers > 0 启用多进程数据加载
# 一个常见的起始点是 num_workers = 4 * num_gpus，但最优值取决于
# 系统（CPU 核心数、磁盘速度）和批次大小。通常需要通过实验确定。
fast_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)

# 迭代看起来相同，但数据加载发生在后台进程中
# for features, labels in fast_loader:
#     # 训练步骤...
#     pass
```

当 `num_workers > 0` 时，`DataLoader` 会生成指定数量的 Python 进程。每个工作进程独立加载一个批次。这使得后续批次的数据加载和变换可以并行发生，而主进程则对当前批次执行模型训练步骤，通常通过更有效地利用 GPU 来显著提高速度。

请注意，增加 `num_workers` 也会增加 CPU 使用率和内存消耗，因为每个工作进程都会加载数据。将其设置过高有时会导致资源争用和收益递减，甚至减慢速度。它通常是根据你的具体硬件和数据集进行调整的参数。

> `DataLoader` 的数据加载流程。`DataLoader` 封装了 `Dataset`，并且在 `num_workers > 0` 的情况下，使用工作进程获取并整理样本成批次，这些批次随后被训练循环使用。

### 使用 `pin_memory` 优化 GPU 传输

在 GPU 上训练时，由 `DataLoader` 加载的数据（位于标准 CPU 内存中）需要传输到 GPU 的内存中。这种传输需要时间。你通常可以通过在 `DataLoader` 中设置 `pin_memory=True` 来稍微加快速度。

```python
# 启用固定内存以加快 CPU 到 GPU 的传输
gpu_optimized_loader = DataLoader(dataset=dataset,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

# 在训练循环内部（假设你有 GPU）
# for features, labels in gpu_optimized_loader:
#     features = features.to('cuda') # 传输变得更快
#     labels = labels.to('cuda')
#     # ...其余训练步骤...
```

设置 `pin_memory=True` 指示 `DataLoader` 在 CPU 端将张量分配到“固定”（页面锁定）内存中。从固定 CPU 内存到 GPU 内存的传输通常比从标准可分页 CPU 内存的传输快。这在与 `num_workers > 0` 一起使用时最有效。请注意，使用固定内存会消耗更多的 CPU 内存。

总之，`DataLoader` 是 PyTorch 中一个核心的工具，它简化并优化了向模型提供数据的过程。通过处理批次化、打乱数据和并行加载，它使你能够专注于模型架构和训练逻辑，同时确保你的数据处理流程高效且可扩展。

## 参考资料

- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — PyTorch Contributors (2025)
  PyTorch DataLoader 的官方文档，解释了其参数和在高效批处理、数据混洗和并行数据加载方面的用法。
- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) — PyTorch Contributors (2025)
  Publisher: PyTorch Foundation
  PyTorch Dataset 类的官方文档，它是 DataLoader 使用的自定义数据集的基础。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  PyTorch 的综合指南，提供了构建和训练深度学习模型的实用知识，包括使用 Dataset 和 DataLoader 进行数据处理。
- [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Sasank Chilamkurthy (2024)
  Publisher: PyTorch Foundation
  一个官方 PyTorch 教程，演示了如何使用 Dataset 和 DataLoader 来准备数据管道，涵盖了常见的数据转换和效率考量。
