---
course: "advanced-pytorch"
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism"
sourceId: 637
chapter: "distributed-training-parallelism"
title: "分布式训练与并行"
order: 5
description: "学习分布式训练的理念，并在PyTorch中实现数据并行 (DDP)、模型并行、流水线并行和FSDP。"
hasQuiz: false
---

现代深度学习模型经常超出单个GPU的内存容量，并且在大量数据集上训练可能需要不切实际的时间。本章侧重于PyTorch中的分布式训练和并行技术，以应对这些挑战。

我们将研究跨多个GPU和节点扩展训练的方法。主要内容包括：

*   与训练相关的分布式计算基本思想。
*   数据并行，使用`DistributedDataParallel` (DDP)。
*   处理极大型模型的方法，例如张量模型并行和流水线并行。
*   内存高效的完全分片数据并行 (FSDP)。
*   使用不同后端配置分布式环境。
*   直接使用PyTorch的底层通信原语 (`torch.distributed`)。

在本章结束时，您将明白如何应用各种并行处理策略，以更高效地训练更大的模型。
