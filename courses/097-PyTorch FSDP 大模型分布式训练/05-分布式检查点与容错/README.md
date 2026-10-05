# 第 5 章：分布式检查点与容错

来源：[原章节](https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-5-distributed-checkpointing-fault-tolerance)

[返回课程目录](../README.md)

大型语言模型的训练通常持续数周，因此硬件可靠性是一个主要考量。当使用数十或数百个GPU进行训练时，将模型的完整状态字典汇集到单个进程进行序列化的传统做法效率不高，并且由于主机内存限制常常无法实现。如果模型大小$M$超出协调器节点的CPU内存，简单的`torch.save()`调用将引发内存不足（OOM）错误。

本章侧重于专为分片架构设计的持久化方案。我们考察当参数在集群中被分区时如何管理状态字典。你将学会实现PyTorch的分布式检查点（DCP）API，它支持并行I/O操作，每个进程只保存其本地的模型分片。

$$ \text{Total I/O Bandwidth} \approx N \times \text{Per-Rank Bandwidth} $$

通过使用所有可用的存储控制器，我们减少了在输入/输出阻塞状态中花费的时间。此外，我们通过TorchElastic处理容错问题。你将配置训练脚本，使其能够检测工作进程故障、重新建立进程组，并无需手动干预地从最后一个有效的快照自动恢复。

## 小节

- 1. [分片状态字典与完整状态字典](01-%E5%88%86%E7%89%87%E7%8A%B6%E6%80%81%E5%AD%97%E5%85%B8%E4%B8%8E%E5%AE%8C%E6%95%B4%E7%8A%B6%E6%80%81%E5%AD%97%E5%85%B8.md)
- 2. [PyTorch 分布式检查点 API](02-PyTorch%20%E5%88%86%E5%B8%83%E5%BC%8F%E6%A3%80%E6%9F%A5%E7%82%B9%20API.md)
- 3. [弹性训练集成](03-%E5%BC%B9%E6%80%A7%E8%AE%AD%E7%BB%83%E9%9B%86%E6%88%90.md)
- 4. [实践：实现可恢复训练](04-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%8F%AF%E6%81%A2%E5%A4%8D%E8%AE%AD%E7%BB%83.md)
