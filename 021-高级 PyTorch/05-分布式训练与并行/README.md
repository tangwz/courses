# 第 5 章：分布式训练与并行

来源：[原章节](https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism)

[返回课程目录](../README.md)

现代深度学习模型经常超出单个GPU的内存容量，并且在大量数据集上训练可能需要不切实际的时间。本章侧重于PyTorch中的分布式训练和并行技术，以应对这些挑战。

我们将研究跨多个GPU和节点扩展训练的方法。主要内容包括：

*   与训练相关的分布式计算基本思想。
*   数据并行，使用`DistributedDataParallel` (DDP)。
*   处理极大型模型的方法，例如张量模型并行和流水线并行。
*   内存高效的完全分片数据并行 (FSDP)。
*   使用不同后端配置分布式环境。
*   直接使用PyTorch的底层通信原语 (`torch.distributed`)。

在本章结束时，您将明白如何应用各种并行处理策略，以更高效地训练更大的模型。

## 小节

- 1. [分布式计算基本原理](01-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%A1%E7%AE%97%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 2. [使用 DistributedDataParallel (DDP) 进行数据并行](02-%E4%BD%BF%E7%94%A8%20DistributedDataParallel%20%28DDP%29%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.md)
- 3. [张量模型并行](03-%E5%BC%A0%E9%87%8F%E6%A8%A1%E5%9E%8B%E5%B9%B6%E8%A1%8C.md)
- 4. [流水线并行实现](04-%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C%E5%AE%9E%E7%8E%B0.md)
- 5. [全分片数据并行（FSDP）](05-%E5%85%A8%E5%88%86%E7%89%87%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%EF%BC%88FSDP%EF%BC%89.md)
- 6. [使用 torch.distributed 通信原语](06-%E4%BD%BF%E7%94%A8%20torch.distributed%20%E9%80%9A%E4%BF%A1%E5%8E%9F%E8%AF%AD.md)
- 7. [设置分布式环境](07-%E8%AE%BE%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83.md)
- 8. [实践操作：设置DDP训练脚本](08-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E8%AE%BE%E7%BD%AEDDP%E8%AE%AD%E7%BB%83%E8%84%9A%E6%9C%AC.md)
