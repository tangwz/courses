---
course: "advanced-vector-search-llms"
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-4-scaling-vector-search-production"
sourceId: 785
chapter: "scaling-vector-search-production"
title: "面向生产系统的向量搜索扩容"
order: 4
description: "为大规模生产LLM应用设计和管理可扩展的分布式向量搜索系统。"
hasQuiz: false
---

构建高效的向量搜索索引并对本地查询进行优化是基础步骤。然而，将这些系统迁移到处理真实流量和海量数据集时，会带来独特的运维和架构挑战。当向量数量 $N$ 增长到数十亿规模，且查询吞吐量 ($QPS$) 需要保持高水平时，前面讨论的单节点方法将变得不足。

本章将着重介绍在生产规模下构建和管理能可靠、高效运行的向量搜索系统所需的策略和架构模式。你将了解到：

*   针对向量搜索设计分布式系统架构。
*   实现分片技术，以在多台机器上对大型索引进行分区。
*   通过复制确保高可用性和容错性。
*   使用负载均衡有效分配查询负载。
*   监控向量搜索系统的主要性能指标。
*   管理实时环境中的索引更新和维护。
*   考虑大型部署的成本影响和优化策略。

到本章结束时，你将理解构建能够支持严苛、大规模LLM应用的向量搜索方案所需的工程原理。
