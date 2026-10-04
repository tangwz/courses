---
course: "mixture-of-experts"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts/chapter-4-scaling-moe-distributed-training"
sourceId: 900
chapter: "scaling-moe-distributed-training"
title: "扩展 MoE 模型：分布式训练"
order: 4
description: "实施专为大规模专家混合模型定制的高级分布式训练策略。"
hasQuiz: false
---

基于之前讨论的架构设计和训练机制，本章侧重于扩展专家混合模型以应对大规模应用的实际需要。MoE 模型庞大的体量和独有的稀疏激活模式，与密集型架构相比，带来特殊困难，常常使得标准数据并行化方法无法单独胜任。

在此，您将学习专门为 MoE 模型调整的分布式训练策略。我们将考察 *专家并行*，这是一种将 MoE 层中的各个专家分散到不同处理单元上的方法。您将学习如何将其与现有数据并行和流水线并行策略结合，以有效管理计算负载和内存占用。

本章重要篇幅讨论 MoE 训练中固有的通信开销，尤其是将令牌表示 ($x$) 路由到其指定专家 ($E_j$) 所需的 *All-to-All* 通信步骤，这些专家可能位于不同设备上。从原理上讲，这涉及根据门控决策 ($g(x)$) 对令牌进行映射：
$$
x_{\text{on device } i} \xrightarrow{g(x)} E_{j (\text{on device } k)}
$$
我们将讨论优化这些通信模式的方法，例如计算-通信重叠。此外，我们还将提及为方便实施分布式 MoE 训练而开发的软件库和框架。实践练习将包括为 MoE 模型配置分布式设置。
