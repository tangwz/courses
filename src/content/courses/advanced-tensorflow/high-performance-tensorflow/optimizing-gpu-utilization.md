---
course: "advanced-tensorflow"
chapter: "high-performance-tensorflow"
lesson: "optimizing-gpu-utilization"
sourceId: 2822
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow/optimizing-gpu-utilization"
title: "优化 GPU 利用率"
description: "在 TensorFlow 训练和推理期间，确保高效使用 GPU 资源的策略。"
order: 2
plots: ["plots/2822-0.json"]
sourceHash: "0c5a7a892ca5408e3ce7001eb388711274b35914c5de65418805a478ff2f87c2"
sourceCorrections: []
---

通过性能分析工具识别潜在性能问题时，图形处理器 (GPU) 的使用情况常常是一个需要关注的常见方面。GPU 擅长同时执行大量并行计算，这使得它们非常适合深度学习 (deep learning)中的矩阵乘法和卷积运算。然而，仅仅拥有一个强大的 GPU 并不能保证最佳性能。如果 GPU 闲置等待数据或指令，潜在的加速效果就会丢失。为了最大化其计算吞吐量 (throughput)，将着重介绍一些策略来确保您的 GPU 持续繁忙。

## 了解 GPU 工作负载和瓶颈

GPU 通过大规模并行来达到高性能。它们包含数千个核心，旨在同时在不同数据点上执行相同的操作（内核）。这与 CPU 形成对比，CPU 通常具有数量较少但更强大的核心，这些核心经过优化以处理顺序任务和复杂的控制流。

为了有效使用 GPU，您需要向其提供大量可并行处理的工作块。GPU 利用率低下的常见原因有：

1. **数据准备瓶颈：** CPU 加载、预处理和传输数据到 GPU 的速度不够快。GPU 完成当前批次的处理后，会闲置等待下一个批次。
2. **小批次大小：** 处理非常小的数据批次可能无法提供足够的并行工作量来充分使用 GPU 的核心。启动计算内核的开销可能占据实际计算时间的大部分。
3. **频繁数据传输：** CPU 内存和 GPU 内存之间频繁的数据复制速度较慢，可能导致计算停滞。
4. **顺序操作或小内核：** 无法有效并行化或涉及许多小型、独立 GPU 内核启动的操作会限制吞吐量 (throughput)。

## 监控 GPU 活动

持续监控对于诊断利用率问题非常重要。

### TensorBoard 分析器

如前所述，TensorBoard 分析器提供了模型执行期间 GPU 活动的详细信息。请密切关注：

- **GPU 利用率：** 在时间线视图中查找 GPU 利用率百分比显著下降的时段。
- **内核执行时间：** 查看内核是否高效运行，或者是否存在许多小型、短持续时间的内核。
- **输入管道分析：** 将低 GPU 利用率与 `tf.data` 管道中可能出现的停滞相关联（稍后将详细介绍）。

### 外部工具：nvidia-smi

对于实时监控，NVIDIA 系统管理界面 (`nvidia-smi`) 命令行工具非常有帮助。在观察模式下运行它可以提供持续更新：

```bash
watch -n 1 nvidia-smi
```

需要观察的重要指标包括：

- **GPU-Util：** 过去一秒钟内，一个或多个内核在 GPU 上执行的时间百分比。在密集训练阶段，目标是持续达到高值（理想情况下 >80-90%）。
- **内存使用：** 占用了多少 GPU 内存。这有助于判断您是否可以增加批次大小。
- **功耗/温度：** 可以表明 GPU 是否得到了有效使用。

低 GPU-Util 通常表明存在 CPU 瓶颈、数据加载效率低下或模型/操作结构不佳等问题。

## 最大化 GPU 吞吐量 (throughput)的策略

根据监控结果，有几种技术可以帮助提高 GPU 利用率：

### 1. 优化批次大小

这通常是影响最大的调整。更大的批次可以为每次迭代提供更多并行工作，从而降低内核启动开销的相对影响，并可能提高利用率。

- **实验：** 逐渐增加批次大小，直到接近 GPU 内存的限制。使用 `nvidia-smi` 监控内存使用情况，并使用 TensorBoard 监控训练速度和 GPU 利用率。
- **内存限制：** 请注意，过大的批次可能导致内存不足 (OOM) 错误。梯度累积等技术（通过顺序小步模拟更大的批次）有时会有帮助，但可能需要自定义训练循环。



![GPU 利用率与批次大小的关系示意图](plots/2822-0.json)



> 增加批次大小通常会提高 GPU 利用率，但会受到硬件容量和内存的限制。

### 2. 高效输入管道 (`tf.data`)

慢速的输入管道是导致 GPU 闲置的常见原因。请确保您的 `tf.data` 管道经过优化：

- **预取：** 将 `dataset.prefetch(tf.data.AUTOTUNE)` 用作管道的最后一步。这使得 CPU 可以在 GPU 处理当前批次时准备下一个批次，从而实现数据准备和计算的重叠。
- **并行处理：** 在 `dataset.map()` 操作中，对于图像增强或数据解析等转换，使用 `num_parallel_calls=tf.data.AUTOTUNE`。这会使用多个 CPU 核心进行数据预处理。
- **缓存：** 如果您的数据集能够放入内存并且预处理开销较大，请使用 `dataset.cache()` 在初始 epoch 后存储结果。

这些 `tf.data` 优化在“`tf.data` 管道的性能考虑”部分中会更详细地介绍。

### 3. 最小化 CPU-GPU 数据传输

主机 (CPU) 和设备 (GPU) 之间的数据传输是瓶颈。

- **将数据保留在 GPU 上：** 尽可能直接对驻留在 GPU 内存中的张量执行操作。在模型或训练循环中，避免不必要地将数据传回 CPU。
- **谨慎使用 `tf.device`：** 虽然您可以使用 `tf.device('/GPU:0')` 明确指定操作的设备，但 TensorFlow 通常能很好地处理设备放置。主要在需要覆盖默认行为或明确管理多个 GPU 时使用它。

### 4. 运用 TensorFlow 特性

- **`tf.function`：** 正如第 1 章中所讨论的，使用 `tf.function` 会将 Python 代码编译成 TensorFlow 图。这种图执行通常快得多，Python 开销更少，并允许框架级别的优化，例如更好地调度 GPU 操作。
- **XLA 编译（预览）：** 加速线性代数 (XLA) 可以通过将多个操作合并为更少、更高效的 GPU 内核来进一步优化性能。这减少了启动开销，并可以大大加快计算速度。启用 XLA（本章稍后介绍）通常会提高兼容模型的利用率。
- **混合精度训练（预览）：** 使用 $float16$ 精度（下一节会介绍）可以减少内存使用，允许使用更大的批次大小，并且可以在兼容硬件（如 NVIDIA Tensor Core）上加速计算，直接影响吞吐量 (throughput)。

### 5. 异步执行模型

请了解 TensorFlow 的执行引擎会尝试异步运行操作。当您在 Python 中调用 GPU 操作时，控制权通常会立即返回，而操作在 GPU 上在后台执行。有效使用 `tf.data.prefetch` 可以补充这一点，它确保在 GPU 需要时数据已准备就绪，从而促进 CPU 预处理和 GPU 计算之间的重叠。

## 结论

优化 GPU 利用率是一个迭代过程，包括监控、找出瓶颈和应用有针对性的解决方案。通过仔细管理批次大小、确保 `tf.data` 数据管道高效、最小化数据传输以及运用 `tf.function` 等 TensorFlow 特性，您可以大幅提升训练和推理 (inference)任务的吞吐量 (throughput)。关于混合精度训练和 XLA 编译的后续章节将介绍更多有力的技术，以从您的硬件加速器中获得最大性能。

## 参考资料

- [Better performance with the tf.data API](https://www.tensorflow.org/guide/data_performance) — TensorFlow Authors (2024)
  本指南介绍了构建高效输入管道的技术，包括预取、缓存和并行转换，这些对于保持 GPU 高效运行至关重要。
- [Mixed precision training](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Authors (2024)
  此官方指南详细说明了如何在 TensorFlow 中实现混合精度训练，利用 float16 减少内存使用并加速兼容 GPU 上的计算，从而允许更大的批次大小并提高吞吐量。
- [NVIDIA System Management Interface (nvidia-smi)](https://docs.nvidia.com/deploy/nvidia-smi/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  用于实时监控 GPU 利用率、内存使用情况及其他重要统计数据的命令行工具的官方文档，对于诊断性能问题至关重要。
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  本指南提供了在 NVIDIA GPU 上进行深度学习训练和推理的优化实践和技术，涵盖数据传输、内核启动和硬件特定功能，以最大化计算吞吐量。
