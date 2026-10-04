---
course: "federated-learning"
chapter: "communication-efficiency-optimization"
lesson: "asynchronous-fl-optimizations"
sourceId: 3605
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-5-communication-efficiency-optimization/asynchronous-fl-optimizations"
title: "异步联邦学习优化"
description: "研究应对慢速客户端并提升异步联邦学习环境效率的方法。"
order: 6
plots: []
sourceHash: "7aff4be42b1b93c8d0d75e57066ae521ed6a25d55d98b8424cea835de60d0c82"
sourceCorrections: []
---

梯度压缩技术直接减小消息大小，而异步联邦学习则从时间安排和协作的角度解决通信瓶颈。标准的同步方法，通常被称为联邦平均（FedAvg），要求中央服务器等待选定的一批客户端完成更新并上传，然后才能执行聚合并开始下一轮。这种同步执行过程可能导致效率明显降低，尤其是在异构环境中。

设想这样一种场景：部分客户端拥有快速网络连接和强大硬件，而另一些则网络连接较慢或计算能力有限。在同步配置下，较快的客户端虽然能迅速完成本地训练，但之后却处于空闲状态，等待批次中最慢的客户端（即“慢速客户端”）完成并上传其更新。服务器也因此停滞。这种空闲时间表示资源被浪费，并大大减缓了整体训练流程。

异步联邦学习协议消除了这种严格的同步要求。客户端在本地进行训练，并在准备就绪时将更新发送给服务器。同样，服务器在收到更新后立即进行聚合，无需等待特定的客户端群组或固定的截止时间。

### 异步聚合工作方式

在典型的异步联邦学习系统中：

1. **客户端操作：** 客户端从服务器获取当前全局模型。它执行若干步或若干轮的本地训练。完成后，它将其计算出的更新（例如，梯度或模型权重 (weight)差值）发送回服务器。客户端可能会立即获取服务器上可用的*最新*全局模型，并开始下一个本地训练周期，无需等待其他客户端。
2. **服务器操作：** 服务器维护全局模型。当它收到来自客户端的更新时，会立即将该更新整合到全局模型中，通常采用修改后的聚合规则。它不等待预设数量的客户端。更新后的全局模型可供其他客户端下载。

这种持续的流程避免了同步方法中固有的空闲时间，有助于提升系统吞吐量 (throughput)，尤其是在客户端速度差异较大时。

> 同步与异步时间线的比较。在同步联邦学习中，服务器等待两个客户端（包括慢速客户端 2）全部完成后才继续。在异步联邦学习中，服务器立即处理客户端 1 的更新，使客户端 1 能够更早开始其下一个周期，而客户端 2 的更新则稍后到达。

### 陈旧性问题

异步操作虽然提升了系统使用效率，但也引入了一个重要问题：**陈旧性**。由于客户端独立运行且服务器持续更新模型，客户端的更新通常是基于较旧的全局模型版本计算得出的。客户端下载模型时的版本与服务器应用其更新时的版本之间的模型差异，被称为“陈旧性”（$\tau$）。

使用$\tau$个版本前的模型计算出的更新，可能对当前较新的全局模型$w_{global}$而言并非最优。高度陈旧性可能导致：

- **次优更新：** 基于旧模型计算的梯度，相对于当前模型状态，方向可能不正确。
- **收敛问题：** 如果陈旧性未能得到妥善处理，训练过程可能会出现震荡、收敛速度变慢，甚至发散。
- **偏差：** 较快客户端的更新可能被更频繁地整合到模型中，这可能使模型偏向这些客户端的数据分布。

### 陈旧性处理策略

已有若干优化方法，旨在减轻异步联邦学习中陈旧性的负面影响：

1. **陈旧性感知聚合函数：** 服务器可以根据传入更新$\Delta w_i$的陈旧性$\tau_i$来调整其对全局模型$w_{global}$的贡献，而不是简单地对其进行平均或相加。一种常用方法是降低较旧更新的权重 (weight)：

   
   $$
   w_{global}^{(t+1)} = w_{global}^{(t)} + \eta \cdot \alpha(\tau_i) \cdot \Delta w_i
   $$
   

   这里，$\eta$是服务器端的学习率或缩放因子，$\alpha(\tau_i)$是陈旧性适应函数。此函数通常随着陈旧性$\tau_i$的增加而减小（例如，对于某个常数$\beta > 0$，$\alpha(\tau_i) = 1 / (1 + \beta \tau_i)$，或采用多项式衰减）。这使得较新更新具有更高的重要性。
2. **自适应学习率：** 服务器端聚合和客户端本地训练都可以考虑使用自适应学习率，其可以考虑陈旧性或其他系统动态。
3. **有界陈旧性：** 一些协议对最大允许陈旧性（$\tau_{max}$）设定了上限。服务器可能会丢弃过于陈旧的更新，或者如果当前模型远比客户端所持有的模型新，客户端可能会短暂等待。这形成了半异步系统，旨在平衡效率和稳定性。
4. **服务器端梯度校正：** 更精巧的技术可能涉及服务器尝试估计，如果梯度是在当前模型上计算的，它会是什么样子，但这也会增加复杂性。

### 实现考量

实施异步联邦学习需要仔细考虑服务器和客户端的逻辑：

- **服务器状态管理：** 服务器需要高效处理并发到达的更新，应用聚合规则（可能感知陈旧性），并向请求客户端提供最新模型版本。这通常比同步系统需要更复杂的并发控制。
- **客户端逻辑：** 客户端需要具备获取模型、训练、发送更新的逻辑，并可根据通信成功情况或服务器可用性来管理其本地训练节奏。
- **收敛监控：** 由于进展不均匀以及陈旧性可能引发的震荡，监控收敛情况会更具挑战。评估指标可能需要通过更长的时间窗口进行平均。

### 权衡与应用场景

异步联邦学习为同步训练提供了一种有吸引力的替代方案，尤其适用于具有以下特点的环境：

- **高系统异构性：** 客户端计算能力、网络带宽或可用性方面存在显著差异。
- **大规模：** 在数千乃至数百万设备间协调严格同步具有挑战性。
- **对潜在收敛减慢的容忍：** 如果最大化系统吞吐量 (throughput)和资源的使用效率比在每个通信轮次中实现最快收敛时间更受重视。

然而，这些优势是以潜在的陈旧性导致的收敛问题和增加的实现复杂性为代价的。同步、异步或半异步协议的选择，以及梯度压缩等方法的使用，很大程度上取决于具体的应用限制、网络状况、设备能力和期望的模型性能。分析这些权衡对于设计高效实用的联邦学习系统非常重要。

## 参考资料

- [Advances and Open Problems in Federated Learning](https://doi.org/10.1561/2200000083) — Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, Rafael G. L. D’Oliveira, Hubert Eichner, Salim El Rouayheb, David Evans, Josh Gardner, Zachary Garrett, Adrià Gascón, Badih Ghazi, Phillip B. Gibbons, Marco Gruteser, Zaid Harchaoui, Chaoyang He, Lie He, Zhouyuan Huo, Ben Hutchinson, Justin Hsu, Martin Jaggi, Tara Javidi, Gauri Joshi, Mikhail Khodak, Jakub Konecný, Aleksandra Korolova, Farinaz Koushanfar, Sanmi Koyejo, Tancrède Lepoint, Yang Liu, Prateek Mittal, Mehryar Mohri, Richard Nock, Ayfer Özgür, Rasmus Pagh, Hang Qi, Daniel Ramage, Ramesh Raskar, Mariana Raykova, Dawn Song, Weikang Song, Sebastian U. Stich, Ziteng Sun, Ananda Theertha Suresh, Florian Tramèr, Praneeth Vepakomma, Jianyu Wang, Li Xiong, Zheng Xu, Qiang Yang, Felix X. Yu, Han Yu, Sen Zhao (2021)
  Journal: Foundations and Trends® in Machine Learning; Publisher: now publishers; Volume: 14; Pages: 1-210; DOI: [10.1561/2200000083](https://doi.org/10.1561/2200000083)
  对联邦学习的广泛综述，其中包含关于异步联邦学习、其优势、挑战和各种优化技术的讨论。
