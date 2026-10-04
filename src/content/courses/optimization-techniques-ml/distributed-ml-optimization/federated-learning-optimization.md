---
course: "optimization-techniques-ml"
chapter: "distributed-ml-optimization"
lesson: "federated-learning-optimization"
sourceId: 1295
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-5-distributed-ml-optimization/federated-learning-optimization"
title: "联邦学习优化原理"
description: "介绍联邦学习特有的优化难题和算法（例如，联邦平均算法）。"
order: 6
plots: []
sourceHash: "c6f4ca16858a2d8da900b3de4ce5126758f2c7f49fb4b4e61b97c90303a7fe79"
sourceCorrections: []
---

联邦学习（FL）与以数据中心为核心的分布式学习模式相比，带来了一组独特的优化难题。尽管目标仍是使用分布式数据训练一个共享模型，但联邦学习的标志性特点是数据永久性地保留在边缘设备（如手机或物联网设备）上，这些设备常被称为客户端。这一限制从根本上改变了优化过程，原因包括：隐私考量、大规模分布的客户端、潜在不可靠的网络连接以及客户端数据集之间显著的统计异构性。

与数据可能被分区但位于受控、高带宽环境中的参数 (parameter)服务器或All-Reduce架构不同，联邦学习基于原始数据*不能*离开客户端设备的假设运行。这使得计算必须移至数据所在地，从而产生需要平衡客户端本地计算与到中央服务器通信的优化算法。

### 联邦优化中的核心挑战

在联邦环境中优化模型需要应对几个独特的障碍，这些障碍在传统分布式训练中通常不那么突出：

1. **统计异构性（非独立同分布数据）**：客户端数据集通常根据个人使用模式生成，导致网络中数据分布*不*独立同分布（非IID）。一个用户的数据可能包含猫的图片，另一个用户的数据可能包含狗的图片。在这类偏斜数据上独立训练可能导致本地模型（$w_k$）与最优全局模型（$w^*$）大幅偏离，这种现象常被称为“客户端漂移”。这种异构性违背了许多分布式SGD标准收敛分析所作的假设，并使聚合过程复杂化。
2. **大规模分布和慢速客户端**：联邦学习网络可能涉及数百万客户端，远超典型的集群规模。在任何给定时间，这些客户端中只有一小部分可能可用或适合训练（例如，连接Wi-Fi、充电中、空闲）。此外，客户端硬件（计算能力、内存）和网络条件差异巨大，导致“慢速客户端”需要更长时间来计算或通信其更新，可能减慢同步训练轮次的速度。
3. **通信瓶颈**：通信，特别是从客户端到服务器的上行链路，与数据中心网络相比，通常受限于带宽不足和更高延迟。这使得在传统分布式SGD中常见的频繁通信轮次变得极其昂贵。优化算法必须设计成执行大量的本地计算以分摊通信成本。
4. **隐私顾虑**：由于数据保留在本地，所通信的信息（通常是模型更新或梯度）不得无意中透露客户端私有数据的敏感信息。尽管核心优化算法本身可能不是一种隐私技术，但其设计必须与差分隐私或安全聚合等隐私增强技术兼容，这些技术通常会向交换的消息添加噪声或增加复杂性。

### 联邦平均（FedAvg）：核心算法

解决这些挑战的最基础算法是联邦平均（FedAvg）。它通过在通信轮次之间在客户端上进行更广泛的本地计算，将标准分布式SGD适应到联邦设置中。

FedAvg过程通常在第 $t$ 轮按以下步骤进行：

1. **客户端选择**：中央服务器选择可用客户端中的一小部分 $C$（例如，以 $k$ 索引的 $K$ 个客户端）。
2. **广播**：服务器将当前全局模型状态 $w_t$ 传输给选定的客户端。
3. **本地计算**：每个选定的客户端 $k$ 使用 $w_t$ 初始化其本地模型，并使用其本地数据集 $D_k$ 执行多次本地SGD步骤（或其变体）。这包括运行 $E$ 个本地训练周期。设 $w_{t+1}^k$ 为这些本地更新后客户端 $k$ 上的结果本地模型。
   
   $$
   w_{t+1}^k \leftarrow \text{本地SGD}(w_t, D_k, E)
   $$
   
4. **通信**：每个客户端 $k$ 将其更新后的本地模型 $w_{t+1}^k$（有时是*变化量* $\Delta_k = w_{t+1}^k - w_t$）发送回服务器。
5. **聚合**：服务器聚合来自 $K$ 个客户端的接收到的更新，以形成新的全局模型 $w_{t+1}$。一种常见的聚合策略是根据每个客户端的数据量 $n_k = |D_k|$ 进行加权平均：
   
   $$
   w_{t+1} = \sum_{k=1}^K \frac{n_k}{n} w_{t+1}^k \quad \text{此处 } n = \sum_{k=1}^K n_k
   $$
   

> 联邦平均（FedAvg）算法单轮概览。

FedAvg的主要思想是，执行多次本地更新（$E > 1$）能使每次通信轮次取得更大进展，与传统大批量同步SGD中每一步梯度更新后都进行通信相比，大幅减少了所需的总通信量。然而，过度增加 $E$ 会加剧客户端漂移，尤其是在高度非IID数据的情况下，因为本地模型在平均之前可能会大幅偏离。调整 $E$、客户端抽样比例 $C$、本地学习率以及服务器端优化器（如果有）都是重要的实际考量。

### 处理异构性和通信成本

尽管FedAvg提供了一个基线方法，但许多研究致力于提升其稳定性和效率：

- **处理统计异构性**：非IID数据导致的客户端漂移仍然是主要的性能障碍。像`FedProx`这样的算法在本地客户端目标函数中引入了一个近端项。该项惩罚本地模型 $w_k$ 相对于当前全局模型 $w_t$ 的大幅偏离：

  
  $$
  \min_{w_k} F_k(w_k) + \frac{\mu}{2} \| w_k - w_t \|^2
  $$
  

  这里，$F_k(w_k)$ 是客户端 $k$ 数据 $D_k$ 上的本地损失，$\mu \ge 0$ 控制正则化 (regularization)的强度。这鼓励本地更新保持更接近全局模型，从而减轻漂移。其他方法如`SCAFFOLD`使用控制变量来纠正聚合过程中的客户端漂移。
- **通信效率**：通过本地更新减少轮次，通用分布式设置中的技术被适应于联邦学习。这包括：

  - **模型/梯度压缩**：使用量化 (quantization)（降低权重 (weight)/更新的精度）或稀疏化（仅发送最重要的更新）等技术来减小从客户端发送的消息大小。
  - **联邦Dropout**：在本地训练期间或发送更新之前选择性地应用Dropout。
- **个性化**：认识到由于数据异构性，单个全局模型可能不适合所有客户端，个性化技术旨在调整全局模型或为每个客户端训练个性化模型。这可能涉及在本地微调 (fine-tuning)全局模型、学习个性化层或在联邦框架内使用元学习方法。

因此，联邦学习中的优化是分布式优化原理、统计异构性、系统限制、通信效率和隐私要求之间的一种复杂关联。像FedAvg这样的算法提供了一个起点，但持续进行的研究不断发展更复杂的方法来应对这种日益重要的学习方式所带来的独特挑战。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a.html) — Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 54; Pages: 1273-1282
  引入联邦平均（FedAvg）的奠基性论文，阐述了其旨在解决联邦环境中通信瓶颈和统计异质性的设计。
- [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](https://proceedings.mlr.press/v119/karimireddy20a.html) — Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, Ananda Theertha Suresh (2020)
  Journal: Proceedings of the 37th International Conference on Machine Learning; Publisher: PMLR; Volume: 119; Pages: 5132-5143
  介绍了SCAFFOLD算法，该算法使用控制变量来纠正由统计异质性引起的客户端漂移，提供了改进的收敛保证和实际性能。
