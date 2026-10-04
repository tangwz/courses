---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "why-tf-data"
sourceId: 1757
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/why-tf-data"
title: "为什么选择 tf.data？"
description: "了解使用 tf.data 处理大型数据集和提升性能的优势。"
order: 1
plots: []
sourceHash: "51cc34671fe4344fb8c11824d624554ca71fc0d302fa02bbe222951b7629e3cf"
sourceCorrections: []
---

高效地将数据输入到模型是构建高性能机器学习 (machine learning)系统的重要方面。当数据集小到可以完全放入内存时，使用 NumPy 等库或直接输入 Python 列表可能看起来足够了。然而，随着数据集的增长，或者当预处理步骤变得计算密集时，这种简单的方法很快就会导致瓶颈。您昂贵的硬件（如 GPU 或 TPU）最终可能会等待数据，显著减慢整个训练过程。这正是 `tf.data` 旨在解决的问题。

`tf.data` API 提供工具来构建灵活高效的输入流水线。您可以把输入流水线想象成您数据的一条装配线：它获取原始数据，应用必要的转换（如解析、打乱、批处理、数据增强），并在训练或推理 (inference)需要时及时将数据交付给模型。

那么，为什么选择 `tf.data` 而不是更简单的方法呢？有几个很好的理由：

### 性能优化

训练深度学习 (deep learning)模型通常涉及多次迭代大型数据集。这些迭代期间数据加载和预处理的效率直接影响训练时间。`tf.data` 包含多项性能优化：

1. **流水线化：** `tf.data` 允许预处理步骤（`map`、`filter` 等）与模型在加速器（GPU/TPU）上的训练步骤并发运行。当加速器忙于计算当前批次的梯度时，CPU 可以准备下一个批次。这种重叠最大程度地减少了加速器的空闲时间。`dataset.prefetch(tf.data.AUTOTUNE)` 转换在此处很基本。它将数据生成的时间与数据使用的时间解耦，允许流水线在后台获取或转换数据。
2. **优化执行：** `tf.data` 流水线中的操作以高度优化的 C++ 实现，并且可以在 Python 解释器的全局解释器锁 (GIL) 之外执行。这允许真正的并行性，特别是对于 I/O 密集型操作（如读取文件）或使用 TensorFlow 操作定义的 CPU 密集型预处理任务。您通常可以通过在 `map` 等方法中使用 `num_parallel_calls` 参数 (parameter)来并行化数据转换步骤，从而进一步提升速度。
3. **提升资源利用率：** 通过高效管理数据流和重叠计算，`tf.data` 有助于确保您的 CPU 和加速器保持繁忙，从而加快训练速度并提升硬件利用率。

下图对比了朴素的顺序方法与 `tf.data` 流水线实现的重叠执行。

> 顺序数据处理与使用 `tf.data` 流水线进行重叠处理的对比。每个彩色框代表对一批数据（加载、预处理、训练）的操作，随时间步推进。在流水线方法中，当前批次用于训练时，下一个批次的加载和预处理会同时发生。

### 处理大型数据集

现代机器学习 (machine learning)通常涉及的数据集太大，无法完全放入单台机器的内存中。`tf.data` 在设计时就考虑到了这个限制。它擅长处理存储在磁盘或分布式文件系统上的数据。

- **流式处理：** 从文件源（如 `tf.data.TFRecordDataset` 或 `tf.data.TextLineDataset`）创建的数据集会增量读取数据。任何时候只有数据集的必要部分加载到内存中，让您能够处理 TB 级别的数据集而不会耗尽 RAM。
- **标准格式：** 它与 TFRecord 等优化文件格式良好集成，这些格式旨在高效存储和检索 TensorFlow 中的结构化数据。

### 灵活性和可组合性

`tf.data` API 采用基于可组合转换的函数式编程风格。您从一个源数据集（例如，来自文件、张量或生成器）开始，然后链式连接各种转换，例如：

- `map()`: 对每个元素应用一个函数。
- `filter()`: 根据一个谓词移除元素。
- `batch()`: 将元素分组为批次。
- `shuffle()`: 随机打乱元素。
- `repeat()`: 将数据集重复多个周期。
- `prefetch()`: 重叠预处理和模型执行。

这种可组合的特性使得构建精确匹配您的数据加载和预处理需求的复杂输入流水线变得简单。与需要复杂状态管理的手动迭代循环相比，生成的代码通常更清晰、更易维护。

### 与 TensorFlow 生态系统集成

`tf.data.Dataset` 对象直接与 Keras 等高级 API 集成。您可以将 `Dataset` 对象直接传递给 `model.fit()`、`model.evaluate()` 和 `model.predict()`。Keras 会自动处理数据集的迭代，使得从内存中的 NumPy 数组到高效 `tf.data` 流水线的转换过程顺畅。

总结来说，虽然简单的数据加载可能适用于小型项目，但当您处理大型数据集并要求训练循环具有更高性能时，`tf.data` 变得必不可少。它侧重于性能、可伸缩性、灵活性和集成性，使其成为 TensorFlow 中处理数据输入的标准且推荐的方式。在接下来的章节中，我们将介绍如何使用这个强大的 API 来创建和转换数据集。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  一份官方指南，解释如何使用 `tf.data` 构建高效的TensorFlow输入管道以优化性能，涵盖预取和并行处理等概念。
- [tf.data.Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) — TensorFlow Authors (2024)
  官方 `tf.data.Dataset` 类API文档，详细说明了其方法、转换和用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 391-430
  这本权威书籍的第13章，标题为“使用Data API加载和预处理数据”，对 `tf.data` 进行了实用介绍（第三版）。
