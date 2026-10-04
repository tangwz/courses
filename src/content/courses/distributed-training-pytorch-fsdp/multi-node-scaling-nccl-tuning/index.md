---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-4-multi-node-scaling-nccl-tuning"
sourceId: 1375
chapter: "multi-node-scaling-nccl-tuning"
title: "多节点扩展与NCCL调优"
order: 4
description: "在多节点集群中扩展FSDP。优化NCCL参数，处理网络拓扑，并管理进程组。"
hasQuiz: false
---

将训练任务扩展到多台机器时，会遇到与网络延迟和带宽相关的独特难题。虽然节点内通信常采用NVLink等高速互连方式，但多节点设置则依赖于网络结构，这可能成为主要的性能瓶颈。本章将着重介绍如何在这些分布式集群环境中配置PyTorch FSDP，以使其高效运行。

你将学习如何初始化多节点进程组，并配置底层的NCCL后端，以保证稳定性和速度。我们将详细说明驱动FSDP运行的集体通信原语，特别是`AllGather`和`ReduceScatter`，并分析它们如何在集群中同步分片。本文还会涉及反向预取技术，这些技术旨在通过将通信与计算重叠来隐藏通信延迟。

最后，我们将实现混合分片数据并行（HSDP）。这种方法结合了节点内的完全分片与跨节点的数据复制，提供了一种平衡内存节省与网络通信成本的结构化方法。到本章结束时，你将能够配置一个集群环境，即使节点数量增加，也能保持高吞吐量。
