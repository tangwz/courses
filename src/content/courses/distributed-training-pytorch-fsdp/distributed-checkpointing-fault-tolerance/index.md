---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-5-distributed-checkpointing-fault-tolerance"
sourceId: 1376
chapter: "distributed-checkpointing-fault-tolerance"
title: "分布式检查点与容错"
order: 5
description: "为FSDP实现可靠的分布式检查点。了解状态字典序列化和弹性训练恢复。"
hasQuiz: false
---

大型语言模型的训练通常持续数周，因此硬件可靠性是一个主要考量。当使用数十或数百个GPU进行训练时，将模型的完整状态字典汇集到单个进程进行序列化的传统做法效率不高，并且由于主机内存限制常常无法实现。如果模型大小$M$超出协调器节点的CPU内存，简单的`torch.save()`调用将引发内存不足（OOM）错误。

本章侧重于专为分片架构设计的持久化方案。我们考察当参数在集群中被分区时如何管理状态字典。你将学会实现PyTorch的分布式检查点（DCP）API，它支持并行I/O操作，每个进程只保存其本地的模型分片。

$$ \text{Total I/O Bandwidth} \approx N \times \text{Per-Rank Bandwidth} $$

通过使用所有可用的存储控制器，我们减少了在输入/输出阻塞状态中花费的时间。此外，我们通过TorchElastic处理容错问题。你将配置训练脚本，使其能够检测工作进程故障、重新建立进程组，并无需手动干预地从最后一个有效的快照自动恢复。
