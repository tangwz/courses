---
course: "federated-learning"
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-5-communication-efficiency-optimization"
sourceId: 743
chapter: "communication-efficiency-optimization"
title: "通信效率与系统优化"
order: 5
description: "通过应用先进的梯度压缩、稀疏化、量化和异步方法来优化联邦学习系统。"
hasQuiz: false
---

联邦学习常需协调数百或数千台设备，每台设备都可能训练大型模型。在每次通信回合中，客户端与中心服务器之间发送完整的模型更新或梯度，会消耗大量带宽和时间，经常成为主要的性能瓶颈。从客户端到服务器的上行通信，在许多边缘设备上尤其受限。

本章侧重于减轻这种通信开销的技术。我们将研究减少传输数据大小的方法，例如：
*   **梯度压缩：** 包括量化（例如，用更少比特表示梯度，如 $g_{quantized} = Q(g)$）和稀疏化（只发送最重要的梯度值）等技术。
*   **误差补偿：** 旨在抵消压缩方法可能引入的精度损失的策略。
*   **异步方法：** 允许客户端和服务器无需严格同步即可运行的协议，在处理慢速或间歇性连接时可能提升系统吞吐量。

我们将分析与这些技术相关的权衡，考虑它们对通信成本、计算开销和模型收敛速度的影响。您还将学习如何在实践中应用其中一些方法。
