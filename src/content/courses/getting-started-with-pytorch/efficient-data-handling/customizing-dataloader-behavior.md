---
course: "getting-started-with-pytorch"
chapter: "efficient-data-handling"
lesson: "customizing-dataloader-behavior"
sourceId: 2184
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/customizing-dataloader-behavior"
title: "自定义DataLoader用法"
description: "探讨自定义采样器和collate函数等选项，以应对高级数据加载情况。"
order: 6
plots: []
sourceHash: "44ed53827474a0390b338bfb57f8a177c784bea4da97da5ff2afc2b32e72070a"
sourceCorrections: []
---

“尽管默认的`DataLoader`提供了方便的批处理和随机排列功能，但许多应用需要更精细地控制数据如何抽样和整理成批次。PyTorch通过自定义采样器和`collate`函数提供灵活性，让你能根据具体需求调整数据加载过程，例如处理不平衡数据集或使用可变大小的输入。”

### 使用采样器控制样本选择

`DataLoader`使用`sampler`对象来决定从`Dataset`中抽取索引的顺序。默认情况下，如果`shuffle=True`，它使用`RandomSampler`；如果`shuffle=False`，则使用`SequentialSampler`。但是，你可以通过`sampler`参数 (parameter)显式传入自己的采样器实例（请注意：如果你提供了`sampler`，则必须将`shuffle`设为`False`，因为随机排列是由采样器本身定义的）。

PyTorch在`torch.utils.data`中提供了几种内置采样器：

- `SequentialSampler`：按顺序采样元素，总是以相同的顺序。
- `RandomSampler`：随机采样元素。如果`replacement=True`，则进行有放回采样。
- `SubsetRandomSampler`：从给定索引列表中随机采样元素。它适用于创建验证集划分，而无需修改原始数据集。
- `WeightedRandomSampler`：根据给定概率（权重 (weight)）从`[0,..,len(weights)-1]`中采样元素。这对于处理不平衡数据集特别有用，例如你想对少数类进行过采样或对多数类进行欠采样。

**示例：对不平衡数据使用`WeightedRandomSampler`**

设想一个分类数据集，其中类别“0”有900个样本，类别“1”有100个样本。简单的随机采样会导致批次严重偏向类别“0”。我们可以使用`WeightedRandomSampler`来提高类别“1”样本被选中的概率。

```python
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# 假设“dataset”是你的torch.utils.data.Dataset实例
# 假设“targets”是一个列表或张量，包含每个样本的类别标签
# e.g., targets = [0, 0, 1, 0, ..., 1, 0]

# 为每个样本计算权重
class_counts = torch.bincount(torch.tensor(targets)) # 各类别的计数：例如 [900, 100]
num_samples = len(targets) # 总样本数：1000

# 每个样本的权重是 1 / (其所属类别的样本数)
sample_weights = torch.tensor([1.0 / class_counts[t] for t in targets])

# 创建采样器
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=num_samples, replacement=True)

# 使用自定义采样器创建DataLoader
# 注意：使用采样器时，shuffle必须为False
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

# 现在，从这个dataloader中抽取的批次将随着时间推移，在类别表示上更加平衡。
# for batch_features, batch_labels in dataloader:
#     # 训练步骤...
#     pass
```

你也可以创建完全自定义的采样策略，通过继承`torch.utils.data.Sampler`并实现`__iter__`和`__len__`方法。

### 使用`collate_fn`自定义批次创建

一旦采样器为一个批次提供了索引列表，`DataLoader`会使用`dataset[index]`从`Dataset`中获取对应的样本。然后，它需要将这些单独的样本组装成一个批次。这个组装过程由`collate_fn`参数 (parameter)处理。

默认的`collate_fn`在许多标准情况下都能很好地工作。它会尝试：

- 将NumPy数组和Python数字转换为PyTorch张量。
- 保留数据结构（例如，如果你的`Dataset.__getitem__`返回一个字典，则整理后的批次将是一个字典，其中每个值是对应项目的批次）。
- 沿新维度（批次维度）堆叠张量。

但是，如果你的样本具有不同的大小（例如，不同长度的序列）或包含它不知道如何堆叠的数据类型，默认的`collate_fn`可能会失败或产生不理想的结果。

在这种情况下，你可以为`DataLoader`的`collate_fn`参数提供一个自定义函数。这个函数接收一个样本列表（其中每个样本是`Dataset.__getitem__`的输出），并负责以所需格式返回整理后的批次。

**示例：填充可变长度序列**

一个常见情况是涉及长度不同的序列（例如NLP中的句子）。默认的`collate`函数不能直接将它们堆叠成一个张量。自定义的`collate_fn`可以将每个批次中的序列填充到该批次中的最大长度。

```python
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# 返回可变长度张量的示例Dataset
class VariableSequenceDataset(Dataset):
    def __init__(self, data):
        # data是一个张量列表，例如 [torch.randn(5), torch.randn(8), ...]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 为简单起见，假设每个项目也有一个标签（例如其长度）
        sequence = self.data[idx]
        label = len(sequence)
        return sequence, label

# 自定义collate函数
def pad_collate(batch):
    # batch是一个元组列表：[(序列1, 标签1), (序列2, 标签2), ...]
    # 按序列长度对批次元素进行排序（可选，但通常为了RNN效率而进行）
    # batch.sort(key=lambda x: len(x[0]), reverse=True) # 对于填充不是严格必需的

    # 分离序列和标签
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # 将序列填充到批次中最长序列的长度
    # `batch_first=True` 使输出形状变为 (batch_size, 最大序列长度, 特征)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # 堆叠标签（假设它们是简单的标量）
    labels = torch.tensor(labels)

    return padded_sequences, labels

# 创建数据集和dataloader
sequences = [torch.randn(torch.randint(5, 15, (1,)).item()) for _ in range(100)]
dataset = VariableSequenceDataset(sequences)

dataloader = DataLoader(dataset, batch_size=4, collate_fn=pad_collate)

# 遍历dataloader
# for padded_batch, label_batch in dataloader:
#     # padded_batch 形状：如果序列是一维的，则为 (4, 该批次中的最大长度, 1)
#     # label_batch 形状：(4,)
#     # 模型处理...
#     pass
```

这个自定义`collate_fn`使用`torch.nn.utils.rnn.pad_sequence`来处理填充，确保批次中的所有序列长度相同，使它们适合RNN等模型处理。

### 其他DataLoader自定义参数 (parameter)

除了`sampler`和`collate_fn`，其他参数也提供性能和行为调整：

- `num_workers` (整数，可选)：指定用于数据加载的子进程数量。将其设置为正整数可启用多进程数据加载，这可以显著加快数据获取速度，尤其当数据加载涉及磁盘I/O或CPU上的复杂预处理时。一个常见的起始设置是将其设为可用CPU核心的数量。默认值：`0`（数据加载在主进程中进行）。
- `pin_memory` (布尔值，可选)：如果为`True`，`DataLoader`在返回张量之前会将其复制到CUDA固定内存中。固定内存可以加快从CPU到GPU的数据传输。这仅在你使用GPU进行训练时才有效。默认值：`False`。
- `drop_last` (布尔值，可选)：如果为`True`，当数据集大小不能被批次大小整除时，将丢弃最后一个不完整的批次。如果为`False`（默认值），则最后一个批次可能小于`batch_size`。

通过理解和使用采样器、自定义`collate`函数以及其他`DataLoader`参数，你可以对数据管道获得精确的控制，从而能高效处理各种数据类型和结构，解决数据集不平衡问题，并优化数据加载性能以加快模型训练。

## 参考资料

- [\`torch.utils.data.DataLoader\`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  官方文档，介绍了DataLoader类、其参数以及与数据集和采样器的集成。
- [\`torch.utils.data.Sampler\`](https://pytorch.org/docs/stable/data.html#samplers) — PyTorch Authors (2025)
  Publisher: PyTorch Foundation
  官方文档，详细介绍了各种内置采样器并提供了创建自定义采样策略的指导。
- [\`torch.nn.utils.rnn.pad_sequence\`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) — PyTorch Authors (2024)
  序列填充的官方文档，展示了自定义批处理函数的特定实用工具。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本提供PyTorch实用建议的书籍，涵盖数据加载、数据集设置和自定义技术。
