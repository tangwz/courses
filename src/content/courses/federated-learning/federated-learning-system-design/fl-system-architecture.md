---
course: "federated-learning"
chapter: "federated-learning-system-design"
lesson: "fl-system-architecture"
sourceId: 3608
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-6-federated-learning-system-design/fl-system-architecture"
title: "联邦学习系统的架构"
description: "概述客户端-服务器联邦学习架构中的主要组成部分和交互方式。"
order: 1
plots: []
sourceHash: "43e57a854ff06e1080a520f88467dfbfe72115be0098339a1885d68a8852fc67"
sourceCorrections: []
---

在前几章奠定的算法基础上，我们现在将重心转向联邦学习系统的结构设计。理解这种架构对有效实施、部署和管理联邦学习非常重要。尽管存在不同形式，大多数联邦学习系统都遵循一种常见的架构模式，通常包括一个中央协调服务器和多个分布式客户端。

### 核心组件

一个标准的联邦学习系统包含几个不同的组件，每个都有其特定的作用和职责：

1. **客户端（或工作节点）：** 这些实体持有用于训练的本地私有数据。客户端可以是单个移动设备或传感器（跨设备联邦学习），也可以是整个组织或数据孤岛（跨数据中心联邦学习）。它们的主要职责包括：

   - 安全存储和管理本地数据集。
   - 从服务器接收当前的全局模型参数 (parameter)和训练指令。
   - 在本地数据上执行一次或多次训练模型的迭代。
   - 计算模型更新（例如，梯度或参数差异）。
   - 可能对其更新应用隐私保护技术（如差分隐私噪声添加或裁剪）或通信优化方法（如压缩）。
   - 将处理后的更新发送回服务器。
2. **服务器（或协调器/聚合器）：** 这个中心实体协调整个联邦学习过程。它不直接访问原始客户端数据。它的主要功能是：

   - 初始化全局模型参数。
   - 在每个训练轮次中选择一部分可用客户端参与（客户端选择）。
   - 向选定的客户端广播当前的全局模型和训练配置（例如，学习率、本地迭代次数）。
   - 接收来自参与客户端的模型更新。
   - 可能实施安全聚合协议（使用SMC或HE，如第3章所述），在不查看单个贡献的情况下组合更新。
   - 聚合接收到的更新（使用第2章中FedAvg、FedProx、SCAFFOLD等算法），生成一个改进的全局模型。
   - 根据聚合结果更新全局模型参数。
   - 评估全局模型的性能（通常使用保留测试集或通过协调分布式评估）。
   - 重复此过程，直至达到预设的通信轮次或满足收敛标准。
3. **模型：** 这是正在协同训练的机器学习 (machine learning)模型。它可以是适合任务的任何类型模型，例如线性模型、支持向量 (vector)机，或常见的深度神经网络 (neural network)。模型架构通常集中定义并由所有客户端共享。个性化技术（第4章）可能涉及在本地调整该模型的部分。
4. **通信协议：** 这规定了客户端和服务器如何交互和交换信息。它包括：

   - 用于数据传输的网络协议（例如gRPC、基于HTTPS的REST API）。
   - 模型和更新的序列化格式（例如Protocol Buffers、NumPy数组）。
   - 处理客户端可用性、掉线和潜在网络故障的机制。
   - 用于身份验证和加密通信通道的安全措施（在`fl-system-security-considerations`中进一步讨论）。

### 典型交互流程（同步联邦学习）

最常见的交互模式，特别是在跨数据中心环境或模拟中，遵循同步的、基于轮次的方法：

1. **初始化：** 服务器定义初始全局模型 $w_0$。
2. **客户端选择：** 在第 $t$ 轮中，服务器选择客户端子集 $S_t$。
3. **广播：** 服务器将当前全局模型 $w_t$ 发送给 $S_t$ 中的所有客户端。
4. **本地训练：** 每个选定的客户端 $k \in S_t$ 在其本地数据 $D_k$ 上训练模型 $w_t$ 若干迭代次数 $E$，得到本地模型更新 $\Delta w_k^{t+1}$（或完整的本地模型 $w_k^{t+1}$）。此步骤通常涉及最小化本地损失函数 (loss function) $L_k(w)$。
5. **更新传输：** 每个客户端 $k$ 将其计算出的更新 $\Delta w_k^{t+1}$（可能在应用隐私或压缩技术后）发送回服务器。
6. **聚合：** 服务器从足够数量的客户端收集更新。然后，它使用所选算法聚合这些更新（例如，FedAvg 的加权平均：$\Delta w^{t+1} = \sum_{k \in S_t'} \frac{n_k}{N_t} \Delta w_k^{t+1}$，其中 $S_t'$ 是成功返回更新的客户端集合，$n_k = |D_k|$，且 $N_t = \sum_{k \in S_t'} n_k$）。
7. **全局模型更新：** 服务器更新全局模型：$w_{t+1} = w_t + \eta \Delta w^{t+1}$（其中 $\eta$ 是服务器学习率，通常设为 1）。
8. **迭代：** 过程从步骤 2 开始，重复进行下一轮（$t+1$），直到终止。

以下图表说明了这种典型的客户端-服务器架构和交互流程：

> 一种典型的联邦学习客户端-服务器架构。服务器协调训练轮次，广播模型，接收客户端更新，并对其进行聚合以改进全局模型。客户端在私有数据上执行本地训练。

这种客户端-服务器模型提供清晰的职责分离并简化编排。然而，中央服务器可能成为瓶颈或单点故障。异步协议（第5章讨论）和点对点联邦学习等其他架构也存在，但它们引入了不同的复杂性。对于多数运用复杂聚合或隐私技术的高级实现，这种服务器协调的架构仍然是最常见的模式。在本章后续部分讨论特定框架实现和部署考量之前，理解其组件和流程非常必要。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html) — Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 54; Pages: 1273-1282
  介绍了联邦平均（FedAvg），这是一种基础算法，概述了本节中描述的同步客户端-服务器交互和聚合过程。
- [Federated Learning: A Comprehensive Guide to Distributed Machine Learning](https://link.springer.com/book/10.1007/978-981-19-4820-9) — Yang Li, Cong Lei (2022)
  Publisher: Springer; DOI: [10.1007/978-981-19-4820-9](https://doi.org/10.1007/978-981-19-4820-9)
  一本系统概述联邦学习的著作，包含专门讨论系统架构、组件和交互模式的章节，适合高级课程使用。
