---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "batching-shuffling"
sourceId: 1769
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/batching-shuffling"
title: "批处理与混洗"
description: "配置批处理和混洗以实现有效的 tf.data 模型训练。"
order: 5
plots: []
sourceHash: "73a81656f527a30c3f94c9117e2fb71625e4b717a1e82e727cffa000ba1b846a"
sourceCorrections: []
---

数据集对象通常由张量或文件等来源创建，并可使用 `map()` 进行转换。为模型训练准备数据，混洗和批处理是主要步骤。这些操作对提高训练效率和模型性能非常重要。

### 混洗数据以提升泛化能力

随机梯度下降 (gradient descent)（SGD）及其变体基于这样的假设：数据点（或小批量）是从底层数据分布中独立同分布采样的。如果你的数据集有固有顺序（例如，按类别或收集时间排序），直接将其馈送给模型可能导致收敛不良和泛化能力下降。模型可能纯粹基于此顺序学习到虚假模式，或者梯度更新在批次之间存在偏差。

混洗数据有助于打破这些关联，并确保每个批次更能代表整体数据集分布。`tf.data` API 为此提供了 `shuffle()` 转换。

```python
import tensorflow as tf
import numpy as np

# 例子：一个有固有顺序的数据集 (0, 1, 2, ..., 9)
dataset = tf.data.Dataset.range(10)
print("原始数据集:", list(dataset.as_numpy_iterator()))

# 混洗数据集
# buffer_size 理想情况下应 >= 数据集大小以实现完美混洗，
# 但对于大型数据集，较小的缓冲区可提供折衷方案。
shuffled_dataset = dataset.shuffle(buffer_size=10, seed=42) # 使用种子以保证可复现性
print("混洗后的数据集 (运行 1):", list(shuffled_dataset.as_numpy_iterator()))

# 再次运行会得到不同的顺序 (如果种子未重置)
shuffled_dataset = dataset.shuffle(buffer_size=10)
print("混洗后的数据集 (运行 2):", list(shuffled_dataset.as_numpy_iterator()))

# 使用更小的缓冲区大小
small_buffer_shuffle = dataset.shuffle(buffer_size=3, seed=42)
print("混洗后的数据集 (buffer=3):", list(small_buffer_shuffle.as_numpy_iterator()))
```

**理解 `buffer_size`**

`shuffle()` 转换通过维护一个固定大小的缓冲区来工作。它用数据集中前 `buffer_size` 个元素填充此缓冲区。然后，每当请求一个元素时，它会从缓冲区中随机选择一个元素输出，并将选中的元素替换为输入数据集中的下一个元素。

- **更大的 `buffer_size`**：提供更好、更均匀的混洗，当缓冲区大小接近数据集大小时，混洗效果接近完美。但它需要更多内存，并可能增加启动时间，因为缓冲区需要初始填充。
- **更小的 `buffer_size`**：使用更少内存，启动更快，但混洗随机性较差。元素只能在加载到缓冲区后才能被选中。
- **建议**：为了完美混洗，将 `buffer_size` 设置为数据集中的元素数量（`tf.data.experimental.cardinality(dataset)` 有时可以帮助确定这一点，尽管对于复杂数据流，它可能返回 `tf.data.UNKNOWN_CARDINALITY`）。如果数据集太大无法放入内存，请选择一个明显大于批次大小的 `buffer_size`（例如，1000、10000 或更多），以平衡随机性和资源使用。一个一般的经验法则是使其足够大以容纳几分钟的数据量。

在批处理*之前*进行混洗很重要，以确保每个批次内的元素来自原始数据集序列的不同部分。

### 批处理数据以实现高效处理

一次训练一个示例（批次大小为 1）在计算上效率低下，特别是在 GPU 或 TPU 等硬件加速器上，它们在大型张量的并行计算中表现出色。此外，来自单个示例的梯度估计可能噪声很大。

将数据点分组为小批量解决了这些问题。处理批次允许进行向量 (vector)化操作，大幅加快计算速度。它还提供对梯度更稳定的估计，从而在训练期间实现更平稳的收敛。`tf.data` API 使用 `batch()` 转换。

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
print("原始数据集:", list(dataset.as_numpy_iterator()))

# 以批次大小为 4 对数据集进行批处理
batched_dataset = dataset.batch(4)
print("\n批处理后的数据集 (drop_remainder=False):")
for batch in batched_dataset.as_numpy_iterator():
    print(batch, "形状:", batch.shape)

# 对数据集进行批处理，丢弃最后一个较小的批次
batched_dataset_dropped = dataset.batch(4, drop_remainder=True)
print("\n批处理后的数据集 (drop_remainder=True):")
for batch in batched_dataset_dropped.as_numpy_iterator():
    print(batch, "形状:", batch.shape)
```

**理解 `batch()` 参数 (parameter)**

- `batch_size`：一个整数，指定要组合成单个批次的连续元素数量。`batch()` 转换的每个输出元素将是一个张量（或张量结构），其中第一维是批次大小（如果 `drop_remainder=False`，则最后一个批次可能更小）。
- `drop_remainder` (可选，默认为 `False`)：一个布尔值。如果为 `True`，并且数据集中元素总数不能被 `batch_size` 整除，则最后一个批次（其大小将小于 `batch_size`）会被丢弃。如果你的模型架构或后续数据流步骤需要固定批次维度的输入，这有时是必要的。然而，启用它意味着你将丢失少量训练数据。在训练期间通常保持 `False` 以使用所有数据，但在评估期间如果需要一致性，或在使用可能需要固定形状的 TPU 等硬件时，可能会设置为 `True`。

### 标准顺序：先混洗后批处理

对于典型训练场景，标准和推荐的做法是首先混洗单个元素，*然后*将它们分组为批次。

```python
import tensorflow as tf

BUFFER_SIZE = 100
BATCH_SIZE = 32
# 假设已定义 AUTOTUNE 以实现最佳预取
AUTOTUNE = tf.data.AUTOTUNE

# 使用 load_and_preprocess 函数的例子
# dataset = load_dataset(...)
# dataset = dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)

# 更具体的例子:
dataset = tf.data.Dataset.range(1000) # 模拟一个更大的数据集

# 先进行混洗，然后进行批处理
final_dataset = dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE)

# 可选地添加预取以提升性能
final_dataset = final_dataset.prefetch(buffer_count=AUTOTUNE)

print(f"数据集规范: {final_dataset.element_spec}")

# 迭代几个批次
print("\n前几个批次（先混洗后批处理）:")
for i, batch in enumerate(final_dataset.take(3)): # 取前 3 个批次
    print(f"批次 {i+1} 形状: {tf.shape(batch).numpy()}")
    # print(batch.numpy()) # 取消注释以查看内容
```

为什么 `shuffle().batch()` 是正确的顺序？

- `shuffle()` 作用于单个元素。通过先进行混洗，可以确保在分组*之前*从缓冲区中随机抽取元素。这会使得批次包含的元素混合，这些元素在原始数据集序列中可能相距很远。
- 如果你使用 `batch().shuffle()`，你会首先创建固定批次（例如，元素 0-31，然后是 32-63 等），*然后*再混洗这些*批次*的顺序。每个批次的内部组成将保持不变（0-31 总是会在一起）。这会显著降低随机性，并且通常偏离了 SGD 所做的假设。

下图说明了包含 map、shuffle 和 batch 的典型数据流：

> 数据从源头流出，进行逐元素处理，使用缓冲区进行混洗，分组为批次，并可能进行预取，然后由模型使用。

通过正确结合 `shuffle()` 和 `batch()`，你可以创建一个输入数据流，向模型馈送随机的数据批次，这有助于更有效的训练和更好的模型泛化能力，同时利用批处理的计算效率。请记住根据你的数据集大小和模型要求，考虑混洗时的 `buffer_size` 以及批处理时 `drop_remainder=True` 的潜在需求。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  关于构建高效数据输入管道的指南，涵盖了洗牌和分批策略，以优化模型训练。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  解释了使用小批量数据的理论基础以及随机化数据在梯度优化算法（如SGD）中的重要性。
