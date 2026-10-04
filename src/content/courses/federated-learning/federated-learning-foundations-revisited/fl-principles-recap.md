---
course: "federated-learning"
chapter: "federated-learning-foundations-revisited"
lesson: "fl-principles-recap"
sourceId: 3506
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-1-federated-learning-foundations-revisited/fl-principles-recap"
title: "联邦学习原则回顾"
description: "简要回顾联邦学习的基本理念和工作流程。"
order: 1
plots: []
sourceHash: "7c584463ab20a4d84580f3766e58320b25f37263a2efd36260c6a71c7df77dbe"
sourceCorrections: []
---

联邦学习（FL）代表了传统集中式机器学习 (machine learning)的一个巨大转变。它不将大量用户数据收集到中央存储库中进行模型训练，而是允许在分布式设备上或在独立的组织内部直接协作构建模型，同时将原始数据保留在本地。这种方式自然地提升了数据隐私性，因为敏感信息从未离开过客户端的掌控。支持这种分布式学习方法的核心原则在此回顾。

## 联邦学习工作流程

联邦学习最常用且最根本的方法是联邦平均（FedAvg）。它遵循一个由中央服务器协调的迭代过程，通常涉及以下步骤：

1. **初始化：** 服务器开始时有一个初始全局模型，通常是随机初始化或在公共数据集上预训练 (pre-training)过的模型。
2. **客户端选择：** 在每个通信轮次 $t$ 中，服务器选择可用客户端（例如，移动设备、医院）的一个子集来参与训练。选择策略可能不同，通常涉及随机抽样。
3. **模型分发：** 服务器将当前的全局模型参数 (parameter) $w_t$ 传输给选定的客户端。
4. **本地训练：** 每个选定的客户端 $k$ 使用其本地数据 $D_k$ 更新收到的模型。这通常涉及在其本地目标函数 $F_k(w)$ 上运行多步梯度下降 (gradient descent)（或其变体），该函数从其私有数据导出。设 $w_{t+1}^k$ 为客户端 $k$ 上更新后的本地模型参数。
5. **模型更新传输：** 客户端将其计算出的更新发送回服务器。这可以是完整的更新模型参数 $w_{t+1}^k$，或者更常见的是差值 $\Delta_k = w_{t+1}^k - w_t$，或者梯度 $\nabla F_k(w_t)$。这种传输可能是潜在的通信瓶颈和隐私风险区域。
6. **聚合：** 服务器聚合来自参与客户端的更新，以计算新的全局模型 $w_{t+1}$。在联邦平均中，这通常是根据每个客户端用于训练的数据量进行的加权平均：
   
   $$
   w_{t+1} = \sum_{k \in S_t} \frac{n_k}{n} w_{t+1}^k
   $$
   
   其中 $S_t$ 是第 $t$ 轮中选定客户端的集合，$n_k = |D_k|$ 是客户端 $k$ 上的数据点数量，而 $n = \sum_{k \in S_t} n_k$ 是所有选定客户端的数据点总数。或者，如果发送更新 $\Delta_k$：
   
   $$
   w_{t+1} = w_t + \sum_{k \in S_t} \frac{n_k}{n} \Delta_k
   $$
   
7. **迭代：** 该过程从步骤2开始重复，直到达到预设的通信轮次或满足收敛条件。

这种循环过程使得全局模型能够从分布式数据集中包含的集体知识中学习，而无需将数据集中化。

> 一张图表，说明了涉及服务器和代表性客户端的标准同步联邦学习周期。

## 实体：服务器与客户端

联邦学习生态系统主要包含两种类型的实体：

- **客户端：** 这些是持有本地数据的设备或组织（例如，智能手机、笔记本电脑、医院、银行）。它们拥有计算资源，可以根据服务器的指令执行本地模型训练。它们的参与可能是间歇性的，其资源（CPU、网络带宽、数据量）可能差异很大，从而导致系统异构性。
- **服务器：** 这个中心实体协调学习过程。它初始化模型、选择客户端、分发模型、聚合更新并维护全局状态。虽然服务器不访问原始客户端数据，但其作用对协调和收敛非常重要。服务器本身可能是潜在的单点故障或攻击目标。

## 优化目标

正如章节引言中提到的，首要目标通常是最小化全局目标函数 $F(w)$，它代表了所有 $N$ 个客户端的聚合性能：


$$
\min_{w} F(w) \quad \text{其中} \quad F(w) = \sum_{k=1}^N p_k F_k(w)
$$


这里，$F_k(w) = \mathcal{L}(w; D_k)$ 是客户端 $k$ 在其本地数据集 $D_k$ 上计算的本地损失函数 (loss function)，而 $p_k$ 是分配给客户端 $k$ 的权重 (weight)，通常为 $p_k = n_k / \sum_{j=1}^N n_j$，其中 $n_k = |D_k|$。这种表述表明，我们的目标是获得一个在所有客户端数据分布上平均表现良好的模型。

## 与其他学习模式的对比

区分联邦学习与其他方法很重要：

- **集中式学习：** 要求将所有数据汇集到一个位置，由于隐私、通信或法规限制，联邦学习明确避免了这一点。
- **经典分布式学习：** 通常假设数据分布在集群中的节点上（例如，使用参数 (parameter)服务器），但通常是在单个可信域内，节点能力更同质，且数据分区通常是IID（独立同分布）的。联邦学习特别针对具有更高异构性、不可信环境（可能）和非IID数据的场景。

"对联邦学习基本原则和标准联邦平均工作流程的这份回顾，为后续内容做了铺垫。尽管直接明了，但这个基础模型在几个简化假设下运行。联邦环境带来了数据和系统异构性、隐私漏洞、通信成本以及潜在对抗行为方面的重要挑战，这些挑战推动了我们将在本课程中研究的高级技术。"

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) — H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics; Publisher: PMLR; Volume: 54; Pages: 1273-1282
  介绍了联邦学习和联邦平均（FedAvg）算法，为该领域确立了基本方法。
- [Advances and Open Problems in Federated Learning](https://arxiv.org/pdf/1912.04977.pdf) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Martin Bober, Marc Buděšínský, Aaron Chow, Francisco Corella-Sáez, Heather Davis, K. Addison Debs, Mason R. Greenberg, Clement H. C. Guo, Ram Herlands, Paul K. Ireland, Maryann Y. Kamoun, Felix X. J. Kerschbaumer, Sanmi Koyejo, Tong Li, Sanja L. Fidler, Vladan Markovic, Dara Mirza, Alireza Narimani, Daniel Ogawa, José Augusto Pinto, Nicolas E. Ryffel, Sara Scalia, Lawrence I. Smith, Koray S. Tugan, William Wei, Galen Andrew, Sergey Arbuzov, Hugo Bagpipe, Blakeley H. B. Bauman, Zachary Charles, Geoffrey M. Hinton, Thomas P. M. Furlan, Jan Krcmar, Alon Z. G. Ravid, Andreas Terzis, Andrew C. T. Yao (2021)
  Journal: Foundations and Trends® in Machine Learning; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  对联邦学习的全面综述，讨论了其原理、挑战（例如异构性、隐私）和未来的研究方向。
- [Federated Learning: Challenges, Methods, and Future Directions](https://ieeexplore.ieee.org/document/9044944) — Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith (2020)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 37; Pages: 50-60; DOI: [10.1109/MSP.2020.2975572](https://doi.org/10.1109/MSP.2020.2975572)
  提供了联邦学习的动机、核心工作流程、系统和数据异构挑战以及各种方法的结构化概述。
- [Federated Learning: An Overview of Concepts and Applications](https://arxiv.org/abs/2103.01227) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Martin Bober, Marc Buděšínský, Aaron Chow, Francisco Corella-Sáez, Heather Davis, K. Addison Debs, Mason R. Greenberg, Clement H. C. Guo, Ram Herlands, Paul K. Ireland, Maryann Y. Kamoun, Felix X. J. Kerschbaumer, Sanmi Koyejo, Tong Li, Sanja L. Fidler, Vladan Markovic, Dara Mirza, Alireza Narimani, Daniel Ogawa, José Augusto Pinto, Nicolas E. Ryffel, Sara Scalia, Lawrence I. Smith, Koray S. Tugan, William Wei, Galen Andrew, Sergey Arbuzov, Hugo Bagpipe, Blakeley H. B. Bauman, Zachary Charles, Geoffrey M. Hinton, Thomas P. M. Furlan, Jan Krcmar, Alon Z. G. Ravid, Andreas Terzis, Andrew C. T. Yao (2021)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S01095ED1V01Y202104MLT082](https://doi.org/10.2200/S01095ED1V01Y202104MLT082)
  由该领域众多创建者合著的权威书籍，对联邦学习的原理和应用提供了深入且系统的解释。
