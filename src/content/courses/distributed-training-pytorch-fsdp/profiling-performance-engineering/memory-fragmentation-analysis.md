---
course: "distributed-training-pytorch-fsdp"
chapter: "profiling-performance-engineering"
lesson: "memory-fragmentation-analysis"
sourceId: 7909
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-6-profiling-performance-engineering/memory-fragmentation-analysis"
title: "显存碎片分析"
description: "识别并解决长时间运行训练任务中的显存分配器碎片问题。"
order: 3
plots: ["plots/7909-0.json"]
sourceHash: "298e6c50856c43ed5bbe7ba349ed1e1c5111f2efe994e07800c823d54307b9d1"
sourceCorrections: []
---

显存 (VRAM)不足（OOM）错误常被认为是容量问题。人们会立刻认为模型参数 (parameter)、梯度和优化器状态超出可用显存。然而，在高性能分布式训练中，即使GPU报告有数GB可用显存时，仍会出现一类主要的OOM错误。这种现象便是显存碎片。

使用FSDP训练时，系统会大量地分配和释放显存。当通过 `AllGather` 聚合分片进行计算并随后释放时，PyTorch缓存分配器管理着一个极具动态性的内存堆。经过数千次迭代，显存空间可能变得不连续，就像一块瑞士奶酪。当分配器试图为一个大型张量（例如包含数十亿参数的完整梯度集）预留一个连续块时，尽管总可用显存足够，它仍可能找不到一个足够大的连续空间。

### 缓存分配器机制

为了了解碎片问题，我们必须查看 `c10::cuda::CUDACachingAllocator`。PyTorch通过维护自己的GPU显存 (VRAM)缓存，避免了原生 `cudaMalloc` 和 `cudaFree` 调用带来的高延迟。当一个张量被释放时，显存不会返回给操作系统；相反，它在PyTorch缓存中被标记 (token)为可用。

如果随后的分配请求小于一个可用的缓存块，分配器可能会将一个大块分成两个小块：一个用于即时请求，另一个作为“碎片”保留。随着时间推移，这些碎片会累积。FSDP尤其容易出现这种情况，因为它处理的是大小可变的分配：用于存储的小分片，以及用于计算的大型、完全具体化的层。

> 显存碎片堆的可视化呈现，其中总可用显存超过请求大小，但由于缺少连续空间，分配仍会失败。

### 诊断碎片问题

标准的 `nvidia-smi` 工具不足以进行这种分析，因为它只报告PyTorch进程预留显存 (VRAM)的最高水位，而非内部碎片状态。要诊断此问题，你需要直接查询分配器。

确定问题的主要衡量指标是 `reserved_memory` 和 `allocated_memory` 之间的比率。如果 `reserved_memory` 接近GPU容量，但 `allocated_memory` 明显较低（例如，60-70%），那么碎片问题很可能是原因。

我们可以使用碎片比率公式来量化 (quantization)这一点：

$\text{碎片比率} = 1 - \frac{\text{已分配字节}}{\text{已预留字节}}$

然而，一种更准确的诊断方法涉及捕获显存快照。PyTorch提供了 `torch.cuda.memory._dump_snapshot("snapshot.pickle")`。这会生成分配器状态的序列化转储，通过可视化查看可以确切了解显存中的“空洞”所在位置。

在FSDP正常运行时，预期会看到显存使用量呈现锯齿形模式。在前向传播期间，当层被聚合时，显存会飙升；当它们被释放时，显存会下降。如果“谷底”（最小显存使用量）随时间逐渐升高，或者峰值预留显存增长而已分配显存没有相应增长，则表明分配器未能重组已拆分的块。



![显存概览: 正常运行与碎片化FSDP](plots/7909-0.json)



> 已分配显存（蓝色）与已预留显存（红色）之间的差异表明碎片化加剧，而理想配置（绿色）则使已预留显存紧密贴合分配需求。

### 调整分配器

一旦确定了碎片问题，最有效的解决方法是调整环境变量 `PYTORCH_CUDA_ALLOC_CONF`。此变量控制缓存分配器的行为。

首要参数 (parameter)是 `max_split_size_mb`。默认情况下，分配器会拆分任何块以满足请求。通过设置 `max_split_size_mb`，你可以阻止分配器拆分大于此阈值的块。如果有一个请求需要小块显存 (VRAM)，而唯一可用的块都大于 `max_split_size_mb`，分配器将保持这些大块完整，而是向CUDA驱动请求新的分配。

这种策略为FSDP所需的大规模all-gather操作保留了大的连续块。

对于隐藏层尺寸在4096到8192范围内的模型，起始配置通常如下所示：

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

这告诉PyTorch：“如果你有一个大于512MB的块，不要将其拆分以满足小型的可变请求。” 这确保了当FSDP需要具体化一个大层时，很可能有一个512MB以上的连续块可用。

### 处理垃圾回收

另一个核心参数 (parameter)是 `garbage_collection_threshold`。在某些特殊情况下，在复杂图的反向传播 (backpropagation)过程中，分配器可能不会足够积极地触发垃圾回收。设置阈值比率有助于确保由临时缓冲区产生的碎片在关键分配高峰前得到清理。

例如，设置 `garbage_collection_threshold:0.8` 会在显存 (VRAM)压力达到80%时触发更积极的清理。虽然这会为显存管理增加轻微的CPU开销，但它通常能避免在一个epoch中间出现的致命OOM峰值。

分析显存碎片问题将OOM错误从一个硬性停滞变为一个优化问题。通过使分配器的拆分逻辑与你特定FSDP配置的分配模式对齐 (alignment)，你可以在不改变模型架构的情况下，回收数GB的有效显存。
