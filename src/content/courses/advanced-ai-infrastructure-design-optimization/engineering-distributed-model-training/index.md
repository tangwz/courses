---
course: "advanced-ai-infrastructure-design-optimization"
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training"
sourceId: 1222
chapter: "engineering-distributed-model-training"
title: "第二章：分布式模型训练的工程化"
order: 2
description: "实现先进的分布式训练策略，包括数据并行和模型并行，使用生产级框架，如DeepSpeed。"
hasQuiz: false
---

当今模型的规模，尤其是那些参数量达到数十亿（$N > 10^9$）的模型，常常超出单个GPU的内存容量。训练这些模型需要将工作负载分配到加速器集群上。本章提供构建高效分布式训练系统的工程原理和实用方法。

我们将从审视主要的分布式策略开始。您将学习数据并行机制，其中模型被复制，数据被分片，以及梯度同步中涉及的通信模式。对于单个设备无法容纳的模型，我们将介绍模型并行和流水线并行，这涉及将模型的层或操作划分到多个加速器上。

接着，重点将转向使用生产级框架进行实现。我们将使用Horovod，因为它对数据并行采取直接方法，然后转向微软的DeepSpeed，以实现先进的内存优化技术，例如零冗余优化器（ZeRO）。

最后，我们将讨论大规模训练的运行实际情况。您将学习如何通过高效的检查点机制来设计容错能力，这是长时间运行任务的必要组成部分。本章以实践实验室环节结束，在此环节中，您将配置并运行一个使用PyTorch的完全分片数据并行（FSDP）的Transformer模型分布式训练任务。
