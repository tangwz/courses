---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization"
sourceId: 1374
chapter: "mixed-precision-memory-optimization"
title: "混合精度与内存优化"
order: 3
description: "实现BFloat16混合精度、激活检查点和CPU卸载，以优化FSDP训练期间的内存使用。"
hasQuiz: false
---

即使有高效的分片策略，训练大型语言模型所需的内存也常常超出高端GPU的容量。虽然参数分片可以分散模型权重，但激活值和梯度所需的瞬态内存会随批次大小和序列长度线性增长。本章主要讲解如何减少这种占用，以最大化吞吐量并实现训练更大规模的模型结构。

我们首先分析混合精度训练。从FP32切换到BFloat16可以将内存使用量减少一半，并使用专用张量核心以加速矩阵乘法。你将学习配置FSDP中的`MixedPrecision`策略，以保持数值稳定性，同时加速模型收敛。

接下来，我们介绍激活检查点。这种技术不是为反向传播存储所有中间激活值，而是仅存储一部分并按需重新计算其余部分。这以计算周期换取内存，通常根据配置将激活内存开销从$O(N)$减少到$O(\sqrt{N})$，使你能够在现有硬件上训练更深层的模型。我们还将说明CPU卸载，这是一种当GPU内存不足时，使用系统RAM来存储优化器状态和参数的策略。到本章结束时，你将能够结合这些技术，优化太字节级别规模训练的内存-计算效率比。
