---
course: "planning-optimizing-ai-infrastructure"
chapter: "strategies-for-performance-optimization"
lesson: "optimizing-data-loading-pipelines"
sourceId: 7030
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-5-strategies-for-performance-optimization/optimizing-data-loading-pipelines"
title: "优化数据加载和预处理管道"
description: "构建高效的数据加载和预处理管道，以确保GPU永不等待数据。"
order: 5
plots: []
sourceHash: "e6c5e1311ffd662c14467747f68ede0cb44ee6e2bdc0fd99dbe2fc019152019d"
sourceCorrections: []
---

强大的GPU其运行速度受限于其可获取的数据速度。许多机器学习 (machine learning)工作流程中，训练循环的瓶颈通常不是GPU的计算能力，而是数据加载和预处理管道。当GPU处理完一个批次并等待下一个批次时，它就会处于空闲状态。这种“GPU饥饿”浪费昂贵资源并延长训练时间。优化数据管道不只是微小调整；它是实现高吞吐量 (throughput)训练的必要步骤。

主要问题在于数据加载和模型训练常常被视为顺序步骤。CPU准备一个批次，将其交给GPU，然后才开始准备下一个批次。高效管道将此转变为并行、类似流水线的流程，其中CPU总是在准备下一个批次，而GPU则忙于处理当前批次。

> 在简单管道中，CPU和GPU轮流工作，导致空闲时间。在优化管道中，CPU对下一个批次进行预处理与GPU对当前批次进行训练并行进行，从而最大化资源使用效率。

### 识别常见管道瓶颈

在解决问题之前，您需要知晓问题所在。最常见的瓶颈包括：

- **I/O 限制:** 进程因从存储读取数据而变慢。在处理数千个小文件（如单独的JPEG）或从慢速网络存储读取时，这种情况很常见。打开、读取和关闭每个文件的开销会累积。
- **CPU 限制:** 应用于数据的转换计算量很大。复杂的图像增强、文本分词 (tokenization)或在CPU上执行的其他繁重预处理任务可能比GPU训练一个批次所需的时间更长。
- **效率低下的转换:** 使用纯Python循环或非向量 (vector)化操作进行数据处理比使用NumPy、TensorFlow或PyTorch等库中的优化函数慢得多。

### 核心优化技术

现代深度学习 (deep learning)框架提供内置工具来构建高性能数据管道。目标是尽可能并行处理、预取和缓存数据。

#### 使用多工作器并行加载数据

加快数据管道速度最直接的方法是使用多个进程并行加载和预处理数据。这在克服I/O和CPU瓶颈方面尤其有效。

在PyTorch中，`DataLoader`对象具有`num_workers`参数 (parameter)。将`num_workers`设置为大于0的值会启动相应数量的独立Python进程在后台加载数据。

```python
# PyTorch：使用多工作器并行加载数据
from torch.utils.data import DataLoader

# 在Linux上，一个好的起点是将 num_workers 设置为 CPU 核心数。
# 在 Windows 上要小心，因为进程创建开销更大。
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=8,
    pin_memory=True # 加快 CPU 到 GPU 的数据传输
)
```

在TensorFlow中，`tf.data` API通过`dataset.map()`函数中的`num_parallel_calls`参数提供类似功能。这会将您的预处理函数应用于多个CPU核心。

```python
# TensorFlow：对 map 转换使用并行调用
import tensorflow as tf

# 让 TensorFlow 动态调整并行级别以获得最佳性能。
AUTOTUNE = tf.data.AUTOTUNE

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_and_preprocess_function, num_parallel_calls=AUTOTUNE)
```

#### 预取数据以重叠计算

预取将数据生产（CPU）与数据消耗（GPU）解耦。它创建了一个预处理批次的小缓冲区，这些批次已准备好发送到GPU。当GPU训练批次`N`时，CPU已在准备批次`N+1`。这种简单方法几乎可以完全消除GPU饥饿，只要数据管道能够跟上。

在TensorFlow中，这通过将`prefetch()`作为`tf.data`管道的最后一步来实现。

```python
# TensorFlow：在管道末尾添加预取
dataset = dataset.batch(128)
dataset = dataset.prefetch(buffer_size=AUTOTUNE) # 预取一些批次

# 现在，当模型请求一个批次时，它很可能已经可用。
for batch in dataset:
    # 训练步骤...
    pass
```

在PyTorch中，`num_workers > 0`的`DataLoader`会自动执行一种形式的预取。`pin_memory=True`参数通过将数据放置在特殊内存区域中进一步优化此过程，从而实现更快、异步的数据传输到GPU。

#### 在内存中缓存数据集

如果您的整个数据集小到足以放入机器的RAM中，您可以在第一个epoch后对其进行缓存。在第一个epoch期间，数据从磁盘加载并处理。结果随后存储在内存中。对于所有后续epoch，训练循环直接从这个快速的内存缓存中读取，完全绕过磁盘I/O和初始预处理步骤。

这对于大小不超过几GB的数据集非常有效。

```python
# TensorFlow：在内存中缓存数据集
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(parse_and_preprocess_function, num_parallel_calls=AUTOTUNE)

# 重要提示：为实现高效训练，请在混洗和批处理*之前*进行缓存。
# 这会缓存单个、已预处理的项。
dataset = dataset.cache()

# 现在应用每个 epoch 应该不同的操作。
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(128)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
```

> **警告：** 注意您可用的RAM。尝试缓存大于可用内存的数据集将导致系统显著变慢或崩溃。仅在适当情况下使用此方法。

#### 使用高效数据格式

您在磁盘上存储数据的方式很重要。读取数百万个小文件会产生显著的文件系统开销。从数量较少的大型连续二进制文件中读取效率更高。

- **TFRecord (TensorFlow):** 这是TensorFlow的标准格式。它存储一系列二进制记录，允许您高效地从磁盘流式传输大型数据集，而无需将整个文件加载到内存中。
- **WebDataset (PyTorch):** PyTorch的流行替代方案，它将数据集存储在简单的tar归档文件中。它提供高效的流式传输和混洗功能，非常适合分布式训练。
- **HDF5 / Parquet:** 这些是面向列的格式，非常适合表格数据，并允许非常快速地读取特定数据列。

将数据集从单个文件迁移到TFRecord等格式可以大幅减少数据加载时间，尤其是在基于网络的存储系统上。

### 一个完整的优化管道

通过结合这些技术，您可以构建高性能数据管道。TensorFlow中推荐的做法是按特定顺序链接这些操作，以最大化其效益。

以下是`tf.data`输入管道的模板：

```python
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def build_efficient_pipeline(filenames, batch_size):
    # 1. 从高性能文件格式（如 TFRecord）创建数据集。
    dataset = tf.data.TFRecordDataset(filenames)

    # 2. 对于适合内存的小型数据集，使用 .cache()。
    #    这应该在管道的早期进行。
    dataset = dataset.cache()

    # 3. 混洗数据。大缓冲区大小对随机性很重要。
    dataset = dataset.shuffle(buffer_size=10_000, reshuffle_each_iteration=True)

    # 4. 并行应用预处理。
    dataset = dataset.map(parse_and_preprocess_function, num_parallel_calls=AUTOTUNE)

    # 5. 对数据进行批处理。
    dataset = dataset.batch(batch_size)

    # 6. 预取以重叠 CPU/GPU 工作。这应该是最后一步。
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset

# 用法：
train_pipeline = build_efficient_pipeline(train_files, batch_size=256)

# 模型现在将以最小延迟获得数据。
model.fit(train_pipeline, epochs=10)
```

通过系统地分析和优化模型接收数据的方式，您确保昂贵的计算资源始终得到充分利用，这直接转化为更快的实验和更高效的模型训练。

## 参考资料

- [Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data) — TensorFlow Developers (2024)
  `tf.data` API的官方指南，涵盖并行化、预取、缓存以及TFRecord等高效数据格式。
- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — PyTorch Developers (2025)
  PyTorch `DataLoader`的官方文档，解释了使用`num_workers`和`pin_memory`选项进行并行数据加载。
- [Designing Machine Learning Systems: New Ways of Working with AI](https://www.oreilly.com/library/view/designing-machine-learning/9781492080352/) — Daniel Sarfati, Noah Shamma, and Shreyas Rade (2022)
  Publisher: O'Reilly Media
  一本关于机器学习系统设计和优化的书籍，内容涵盖数据管道、基础设施和性能考量。
