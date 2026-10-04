---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-2-model-wrapping-initialization-policies"
sourceId: 1373
chapter: "model-wrapping-initialization-policies"
title: "模型包装与初始化策略"
order: 2
description: "掌握PyTorch FSDP中的自动包装策略、Transformer专用包装和高效参数初始化。"
hasQuiz: false
---

FSDP的有效实现不仅是简单地实例化包装类。分片算法的性能很大程度上受到模型层如何分组的影响，此过程称为包装。如果模型被视为一个整体单元，FSDP在前向传播期间必须同时收集所有参数，这将抵消分片带来的主要内存优势。为了优化执行，模型必须被分割成更小的单元，以便系统能够动态地收集和释放参数。

本章考察规定模型如何划分的策略。你将配置为Transformer架构设计的自动包装策略，确保编码器或解码器层等特定模块被单独处理。这种细粒度方法使优化器能够将峰值内存使用保持在与单个分片大小加上当前工作单元成比例的水平，而不是整个模型大小$M$。对于不符合标准模式的架构，我们将研究如何使用lambda函数构建自定义包装策略，以精确控制计算图的切割位置。

我们还将处理初始化瓶颈问题。在分片之前将大型模型加载到CPU RAM中，常常导致主机内存不足错误。你将学会使用PyTorch `meta` 设备来初始化模型结构，而无需立即为权重分配存储空间。通过将延迟初始化与`reset_parameters()`方法结合，你可以实例化总参数数量超出可用系统内存的模型。最后，本文涉及处理共享参数的技术方案，以在分布式进程中保持同步而无需重复存储。
