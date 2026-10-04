---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-6-profiling-performance-engineering"
sourceId: 1377
chapter: "profiling-performance-engineering"
title: "性能分析与调优"
order: 6
description: "调试和优化FSDP性能。使用PyTorch Profiler分析CUDA流、内存占用和通信重叠。"
hasQuiz: false
---

成功启动分布式训练运行验证了正确性，但不能保证效率。当扩展到数十或数百个GPU时，同步或内存管理中微小的低效会累积，导致吞吐量显著下降。在此环境下，性能优化不再是简单的代码调整，而是侧重于分析CPU、GPU核和网络结构之间的系统层面协作情况。

本节介绍分布式设置的诊断过程。我们首先使用PyTorch Profiler生成并解读追踪数据，以显示执行时间线。您将学习如何区分计算密集型操作和通信密集型操作，找出GPU空闲的时间段。

本文包含通信重叠的特定分析技术。在理想的FSDP配置中，收集分片参数所需的通信与前一层的计算同时发生。我们将查看如何验证这一行为，并调整反向预取设置，以缩短显露的通信时间。

内存管理同样需要仔细检查。即使总参数量在显存中装得下，内存碎片化也可能触发昂贵的分配器重试或内存不足错误。我们将介绍检查缓存分配器状态的方法来处理这些问题。最后，本章转向高级指标，将模型浮点运算利用率（MFU）定义为衡量训练效率的主要指标。您将学习如何计算硬件的理论峰值性能，并用标准利用率公式测量您的配置与该极限的接近程度：

$$ \text{MFU} = \frac{\text{Achieved TFLOPS}}{\text{Peak Device TFLOPS}} $$
