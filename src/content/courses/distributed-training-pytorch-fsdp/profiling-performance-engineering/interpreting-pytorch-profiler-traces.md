---
course: "distributed-training-pytorch-fsdp"
chapter: "profiling-performance-engineering"
lesson: "interpreting-pytorch-profiler-traces"
sourceId: 7907
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-6-profiling-performance-engineering/interpreting-pytorch-profiler-traces"
title: "解读 PyTorch Profiler 追踪"
description: "阅读并理解追踪堆栈，以找出 GPU 空闲时间和内核启动开销。"
order: 1
plots: ["plots/7907-0.json"]
sourceHash: "86299be8361497f0193d513a9c836093c11497af2d761f0068a90dfdd59145d2"
sourceCorrections: []
---

`nvidia-smi` 报告的高使用率数字常会使工程师误认为他们的训练管线很高效。一个通用的使用率指标只表示 CUDA 内核驻留在设备上；它不能区分高吞吐量 (throughput)的矩阵乘法和内存受限的操作，也无法说明设备正在等待来自 CPU 或网络的数据。为了有效优化 FSDP 训练，必须绕过高层级指标，直接通过追踪文件分析执行时间线。

### GPU 追踪的组成

在 Perfetto 或 Chrome Trace Viewer 中查看时，标准的 PyTorch 分析器追踪将执行数据组织成水平轨迹。对于 NVIDIA GPU 上的分布式训练，有三类特定的轨迹提供了性能调试所需的信号。

**CPU Python 线程**轨迹显示了高层级的应用逻辑，包括 `nn.Module` 调用和数据加载器循环。在此之下是 **CPU CUDA 运行时**轨迹，它记录了对 CUDA 驱动程序进行的 API 调用（例如 `cudaLaunchKernel`、`cudaMemcpyAsync`）。最后，**GPU 流**轨迹可视化了内核在硬件上的实际执行情况。

在优化的 FSDP 设置中，这些轨迹之间的关系是严格分层但异步的。CPU 将内核分派到启动队列，GPU 消耗它们。性能下降通常表现为这种生产者-消费者关系的破裂。

> 分布式环境中主机分派与设备执行流之间的交互。

### 找出 CPU 限制的执行

大规模训练中常见的病症是“GPU 饥饿”，即 GPU 完成工作队列的速度快于 CPU 补充的速度。在分析器追踪中，这表现为 GPU 流轨迹上块之间的空白或间隙。

如果观察到每次内核执行之间存在小的、明显的间隙，则系统存在启动开销问题。PyTorch 的即时执行会为每次内核启动带来固定成本（大约 3-10 微秒）。当 FSDP 封装一个包含许多小层的模型（例如隐藏维度相对于批大小较低的 Transformer）时，启动开销与计算时间的比率会增加。

这里要关注的指标是 GPU 时间线的相对密度。在一个健康的追踪中，内核块紧密排列，呈现为实心的颜色条。明显的空白表示 CPU 在图序列化方面遇到困难。这通常通过启用 `torch.compile` 来融合逐点操作得到解决，从而减少每步所需的内核启动总数。

### 通信与计算重叠

FSDP 很大程度上依赖于通信（NCCL 操作）与计算的重叠。在前向传播期间，当 GPU 计算第 $N$ 层时，系统应同时 `AllGather` 第 $N+1$ 层的参数 (parameter)。

为了验证此行为，请在 GPU 轨迹中找到 **NCCL** 内核。这些内核通常被标记 (token)为 `ncclKernel`、`ncclAllGatherRingLL` 或根据后端版本而定的类似变体。应该看到这些通信内核在与计算内核（例如 `sgemm`、`elementwise_kernel`）并行的一个单独流上执行。

如果 NCCL 内核和计算内核按序执行，即一个流空闲而另一个流活跃，则表示重叠配置失败。这种按序执行是由于不当的 CUDA 流优先级设置或 NCCL 进程组配置中的限制造成的。

这种重叠的效率是可以量化 (quantization)的。设 $T_{compute}$ 为层的前向/反向传播 (backpropagation)所需时间，而 $T_{comm}$ 为收集或分散其参数所需时间。暴露的通信开销 $T_{exposed}$ 为：

$T_{exposed} = \max(0, T_{comm} - T_{compute})$

当 $T_{exposed}$ 趋近于零时，会发生理想的扩展。如果分析器显示 $T_{comm}$ 明显超过 $T_{compute}$，则网络带宽是瓶颈，简单地增加更多 GPU 可能无法线性提升训练速度。

### 分析反向传播 (backpropagation)结构

FSDP 中的反向传播很复杂，因为它涉及两种不同的通信原语：`AllGather`（用于获取完整参数 (parameter)以进行梯度计算）和 `ReduceScatter`（用于同步梯度并立即对其进行分片）。

反向传播的追踪应显示跨流的“之字形”模式。当第 $L$ 层的反向计算完成后：

1. 第 $L$ 层梯度的 `ReduceScatter` 操作在通信流上开始。
2. 第 $L-1$ 层参数的 `AllGather` 操作开始（预取）。
3. 计算流执行第 $L-1$ 层的反向传播。



![理想的 FSDP 反向传播重叠](plots/7907-0.json)



> 描述理想流并发的甘特图。注意相邻层的通信操作如何与当前层的计算同时执行。

在次优的追踪中，你将观察到 CPU 轨迹上的“等待”内核（通常表示为 `cudaStreamSynchronize` 或 `cudaEventSynchronize`）持续较长时间。这表明 Python 代码在等待 GPU 完成通信事件后才能分派下一个计算内核，这违背了异步执行的目的。

### 内存分配器碎片化

除了计算和通信，分析器还显露了内存管理开销。频繁调用 `cudaMalloc` 和 `cudaFree` 代价高昂。PyTorch 使用缓存分配器来缓解此问题，但在 FSDP 中，对完整参数 (parameter)分片的持续分配和释放可能导致碎片化。

在分析器中，查看内存时间线轨迹。当层被实例化和丢弃时，FSDP 中出现锯齿模式是预期且正常的。但是，如果在训练步骤中间（不仅仅是前几次迭代）看到 `cudaMalloc` 调用激增，这表明缓存分配器正在频繁地重新分配。分配器被迫寻找空闲块或拆分现有块，这会阻塞 CPU 分派线程。这通常通过设置 `PYTORCH_CUDA_ALLOC_CONF` 环境变量来调整 `max_split_size_mb` 得到纠正，从而强制分配器保留更大的连续块以供周期性参数扩展使用。
