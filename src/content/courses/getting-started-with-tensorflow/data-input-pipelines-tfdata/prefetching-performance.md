---
course: "getting-started-with-tensorflow"
chapter: "data-input-pipelines-tfdata"
lesson: "prefetching-performance"
sourceId: 1772
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-5-data-input-pipelines-tfdata/prefetching-performance"
title: "为提高性能而预取"
description: "理解并应用预取以使数据预处理和模型执行重叠。"
order: 6
plots: []
sourceHash: "ea9cf040821cc2beacd539956d9290437ad4037f97f99608d20c902749ed6a30"
sourceCorrections: []
---

在训练模型时，尤其是使用 GPU 或 TPU 等硬件加速器时，加速器可能会在等待 CPU 准备下一批数据时处于闲置状态。反之，CPU 也可能在等待加速器处理完当前批次时等待。这种“停-等”模式会引入延迟并导致硬件资源利用不足，从而减慢整体训练过程。

考虑一个未经优化的简化训练循环时间线：

> 未预取的执行时间线。请注意 CPU 和 GPU 上的闲置期，因为它们相互等待。

`tf.data` API 为此问题提供了一个简单解决方案：`prefetch()` 变换。预取使预处理和模型执行步骤重叠。当模型在 GPU 上使用批次 `N` 执行训练步骤（前向和后向传播）时，输入管道会使用 CPU 读取并预处理批次 `N+1` 的数据。

这通过在管道末尾添加 `dataset.prefetch(buffer_size)` 来实现。`buffer_size` 参数 (parameter)指定将预取的最大元素数量（或批次，如果应用于 `batch()` 之后）。

```python
import tensorflow as tf

# 假设 'dataset' 是初始加载后的 tf.data.Dataset 对象
# 应用如 map、shuffle、batch 等变换
dataset = dataset.map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(batch_size=32)

# 在末尾添加预取
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# 现在数据集已准备好传递给 model.fit()
# model.fit(dataset, epochs=10)
```

### 选择 `buffer_size`

手动确定最佳 `buffer_size` 可能很复杂。它取决于预处理与模型执行所需的时间、可用内存和系统配置等因素。缓冲区过小可能无法完全隐藏数据准备延迟，而缓冲区过大可能会消耗过多内存。

幸运的是，TensorFlow 提供了 `tf.data.AUTOTUNE`（如上例所示）。当您设置 `buffer_size=tf.data.AUTOTUNE` 时，`tf.data` 运行时会动态调整该值，试图使用保持加速器处于工作状态所需的最小缓冲区大小，同时兼顾可用内存。在大多数情况下，使用 `tf.data.AUTOTUNE` 是推荐的方法。

### `prefetch()` 的放置位置

为了达到最佳效果，`prefetch()` 通常应是添加到数据集管道的**最后**一个变换。这确保所有前面的操作（加载、映射、混洗、批处理）都异步执行并与模型的训练步骤重叠。

使用预取后，执行时间线看起来更有效率：

> 使用预取的执行时间线。当 GPU 忙于训练当前批次时，CPU 准备下一个批次，这显著减少了闲置时间。

通过将 `.prefetch(tf.data.AUTOTUNE)` 作为输入管道的最后一步，您使 TensorFlow 能够自动管理数据准备和模型计算之间的重叠，通常会以极小的代码改动带来训练吞吐量 (throughput)的显著提升。这是一种简单而高效的方法，用于优化数据管道的性能。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2023)
  Journal: TensorFlow official documentation
  这是优化`tf.data`管道的官方主要资源，详细介绍了`prefetch()`和`AUTOTUNE`。
- [tf.data: A Flexible and Performant Data Input Pipeline for TensorFlow](https://www.usenix.org/conference/sysml18/workshop-program/presentation/holtz) — Ethan Holtz, Anjali Sridhar, Martin Wicke, Emily Newman, Anirudh Raman, Jeremy Hyrkas, Sameer Singh (2018)
  Journal: 2nd Conference on Systems for Machine Learning (SysML); Publisher: USENIX; DOI: [10.5555/3295222.3295229](https://doi.org/10.5555/3295222.3295229)
  介绍了`tf.data` API，解释了其构建高效灵活数据管道的设计原则，包括`prefetch`等操作的动机。
- [Performance Guide](https://www.tensorflow.org/guide/performance) — TensorFlow Authors (2023)
  Journal: TensorFlow official documentation
  本指南概述了TensorFlow中的各种性能优化技术，其中`tf.data`输入管道优化（包括预取）是关键组成部分。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本全面介绍TensorFlow的实用指南，涵盖了使用`tf.data`构建和优化数据输入管道，提供了具体的性能实践和建议。
