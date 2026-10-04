---
course: "mixture-of-experts-advanced-implementation"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-2-advanced-routing-mechanisms"
sourceId: 1233
chapter: "advanced-routing-mechanisms"
title: "进阶路由机制"
order: 2
description: "学习实现和分析进阶路由机制，包括噪声 Top-k 门控、基于哈希的路由和 Switch Transformers。"
hasQuiz: false
---

专家混合模型（MoE）的性能很大程度上取决于其路由机制。该部分负责将每个输入 token 引导至一小部分专家。此路由判定的质量直接影响模型的性能、训练稳定性和计算效率。尽管标准的 Top-k 路由器是一个可行的起点，但它可能导致负载不均衡等问题，例如有些专家持续被过度选用，而另一些则未得到充分使用。

本章介绍一系列进阶路由机制，旨在解决基础门控的局限性。我们将考量不同路由算法之间的权衡，包括它们的计算成本、对负载均衡的影响以及对专家分工的影响。

在本章中，你将学习实现和分析几种重要的路由策略：

*   **噪声 Top-k 门控：** 一种向门控 logits $h(x)$ 引入噪声的技术，以在训练期间改善负载分配。
*   **Switch Transformers：** 一种通过将每个 token 发送到仅一个专家 ($k=1$) 来简化路由的架构，从而减少通信开销。
*   **基于哈希的路由：** 一种使用哈希函数进行 token 分配的确定性方法，消除了对学习型门控网络的需要。
*   **Soft MoE：** 一种完全可微的方法，计算所有专家的加权平均值，创建“软”分配而非硬性离散选择。

我们还将介绍分析路由器决策的方法，以理解专业化是如何形成的。本章最后是实践部分，你将实现这些不同的路由器，以对它们的运作机制和性能特征获得实际理解。
