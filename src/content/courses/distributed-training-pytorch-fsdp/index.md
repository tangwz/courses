---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp"
sourceId: 246
title: "PyTorch FSDP 大模型分布式训练"
description: "使用PyTorch的全分片数据并行（FSDP）实现并优化TB级模型的分布式训练。"
category: "Large Language Models"
level: 4
duration: 30
order: 97
prerequisites: "PyTorch高级应用，分布式基础知识"
color: "red"
chapterCount: 6
lessonCount: 28
outcomes: [{"topic": "FSDP架构", "description": "利用ZeRO阶段对参数、梯度和优化器状态进行分区，设计扩展方案。"}, {"topic": "内存优化", "description": "实现激活检查点和CPU卸载，以最大化每GPU的吞吐量。"}, {"topic": "多节点网络", "description": "配置并调整NCCL通信，以实现高效的跨节点扩展。"}, {"topic": "性能分析", "description": "分析通信与计算重叠，并解决内存碎片问题。"}]
hasProject: false
---

扩展深度学习 (deep learning)模型超出单GPU限制，需要精密的并行化方案。本课程介绍PyTorch中的FSDP（全分片数据并行）技术，它对训练大型语言模型（LLM）及其他参数 (parameter)量巨大的模型结构非常实用。内容涉及DDP的局限性，并原生实现了Zero冗余优化器（ZeRO）算法。所讲内容包括分片策略、BFloat16混合精度训练、激活检查点和CPU卸载。课程还涉及多节点集群配置、使用NCCL分析网络瓶颈，以及管理分布式状态字典以实现容错。重点在于TB级模型的性能调优和内存效率。
