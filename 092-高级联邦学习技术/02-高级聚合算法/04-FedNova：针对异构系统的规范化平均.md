# FedNova：针对异构系统的规范化平均

来源：[原文](https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/fednova-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

"虽然联邦平均（FedAvg）为聚合客户端更新提供了一个简单的基准方法，但其在具有*系统异构性*的联邦网络中性能可能会明显下降。客户端通常拥有不同的计算能力（CPU、内存）、网络带宽和功耗限制。这种异构性导致每个客户端在给定通信轮次中能够执行的本地计算量不同。有些客户端可能完成很多本地训练周期（$E$），而其他客户端只能完成少量。"

标准的FedAvg聚合客户端模型（或更新）时，通常根据数据点数量（$n_k$）进行加权。然而，它没有明确考虑为完成这些更新所执行的不同本地工作量（$\tau_k$，即客户端$k$执行的本地梯度步数或更新次数）。当$\tau_k$在不同客户端之间差异很大时，FedAvg可能会出现以下问题：

1. **客户端漂移不匹配：** 执行更多本地步骤的客户端的更新可能表示模型已偏离初始全局模型$w^t$更远。简单地平均这些更新可能导致不稳定或收敛缓慢，因为聚合更新可能被进行了大量（可能导致偏离）本地优化的客户端所主导。
2. **过时更新（隐式）：** 在服务器等待达到法定数量的同步设置中，速度更快的客户端提前完成。在异步设置中，更新在不同时间到达，反映了不同的工作量。FedAvg本身不会补偿更新的“过时”或计算努力的差异。
3. **梯度近似效率低：** 聚合的目标通常是近似真实的全局梯度$\nabla F(w) = \sum p_k \nabla F_k(w)$。如果客户端执行不同数量的本地步骤（$\tau_k$），FedAvg中使用的本地更新的简单加权平均$\Delta_k^t = w_k^{t+1} - w^t$可能不是最有效的近似，特别是当涉及到本地学习率$\eta_l$时（因为本地更新$\Delta_k^t \approx - \eta_l \tau_k \nabla F_k(w^t)$）。

### FedNova方法：规范化本地更新

联邦规范化平均（FedNova）通过在聚合前对客户端更新进行规范化，直接处理系统异构性。核心思想是调整每个客户端的贡献，以抵消不同本地计算量的影响，旨在获得一个能更好地反映客户端平均*梯度方向*的更新，而与每个客户端在本地执行了多少步无关。

FedNova不是平均最终本地模型或原始更新$\Delta_k^t$，而是平均按已执行的本地工作量规范化后的更新。令$\tau_k$为客户端$k$在轮次$t$中执行的本地步骤数（例如，SGD更新次数）。客户端$k$提交的本地更新是$\Delta_k^t = w_k^{t+1} - w^t$。

FedNova按如下方式计算全局模型更新：


$$
w^{t+1} = w^t + \sum_{k \in S_t} p_k \left( \frac{\Delta_k^t}{\tau_k} \right)
$$


这里，$S_t$是参与轮次$t$的客户端集合，$p_k$是客户端$k$的聚合权重 (weight)（通常$p_k = n_k / \sum_{j \in S_t} n_j$，但也可以采用其他加权方式）。

将此与通过本地更新来理解的FedAvg中隐式更新方向进行比较：


$$
w^{t+1}_{\text{FedAvg}} = w^t + \sum_{k \in S_t} p_k \Delta_k^t
$$


实际上，FedNova旨在平均*平均本地步骤更新*（$\Delta_k^t / \tau_k$），并按$p_k$加权。这避免了执行许多本地步骤（$\tau_k \gg \tau_j$）的客户端仅仅因为计算量大而过度影响聚合更新的幅度和方向。它假设当$\tau_k$不同时，$\Delta_k^t / \tau_k$提供了比原始更新$\Delta_k^t$更好的、与全局目标相关的本地梯度方向估计。这对于常见的本地求解器（如SGD）通常表现良好。

### 可视化规范化效果

考虑两个拥有相同数据量（$p_1 = p_2 = 0.5$）但计算能力不同的客户端。客户端1执行$\tau_1 = 2$本地步骤，而客户端2执行$\tau_2 = 10$本地步骤。它们的原始更新是$\Delta_1^t$和$\Delta_2^t$。

> FedAvg直接平均了可能差异很大的更新$\Delta_1^t$和$\Delta_2^t$。如果客户端2的更新$\Delta_2^t$由于$\tau_2=10$而大得多，它可能占据主导地位。FedNova在平均之前，首先根据步数（$\tau_k$）进行规范化，可能导致一个不同的聚合方向，该方向能更好地反映平均单步进展。

### 优点与收敛性

FedNova的主要优点是其对系统异构性的鲁棒性。通过基于本地工作量规范化更新：

- **提升收敛速度：** 理论分析表明，在系统异构性显著的情况下，FedNova可以比FedAvg获得更快的收敛速度，因为它校正了由不同$\tau_k$引起的各种贡献。
- **公平性：** 它避免了计算能力强的客户端或滞后者（如果允许，他们可能会运行更长时间）仅仅因为本地计算量大而对全局模型更新产生不当影响。
- **稳定性：** 当$\tau_k$值在客户端或轮次之间大幅波动时，规范化可以使训练过程比FedAvg更稳定。

### 实际考量

实施FedNova需要对标准联邦学习协议进行少量修改：

- **报告本地工作量：** 客户端需要与模型更新（$\Delta_k^t$）一起报告执行的本地工作量（$\tau_k$）。这通常是一个标量值（例如，执行的本地梯度步数），增加的通信开销很小。
- **服务器端计算：** 服务器在执行加权聚合步骤之前执行规范化（$\Delta_k^t / \tau_k$）。

值得注意的是，FedNova主要处理*系统*异构性。它不直接解决*统计*异构性（非独立同分布数据）带来的问题，尽管它可以与为该目的设计的算法（如FedProx或SCAFFOLD）结合使用。FedNova的有效性依赖于以下假设：规范化更新$\Delta_k^t / \tau_k$是一个有意义的量，表示平均步进方向。这对于常见的本地求解器（如SGD）通常表现良好。

总之，FedNova提供了一种理论上合理且实践中有效的方法，用于缓解联邦网络中客户端计算资源和本地步数差异带来的不利影响。通过在聚合前规范化更新，它促进了更公平的贡献，并在异构系统环境中，与FedAvg相比，可以实现更快、更稳定的收敛。当客户端能力差异很大时，它是构建更高性能联邦学习系统的宝贵工具。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) — H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: PMLR; Volume: 54; Pages: 1319-1327; DOI: [10.48550/arXiv.1602.05629](https://doi.org/10.48550/arXiv.1602.05629)
  这篇奠基性论文介绍了联邦平均（FedAvg），这是一种用于通信高效联邦学习的基准算法，也是解决异构性方法的起点。
- [Federated Learning: Challenges, Methods, and Future Directions](https://ieeexplore.ieee.org/document/9427302) — Qiang Yang, Yuanshun Yao, Tian Li, Mao Yang, S. Yu Philip, Jian Li, Cong Shen, Lixin Fan (2021)
  Journal: IEEE Transactions on Big Data; Publisher: IEEE; Volume: 7; Pages: 1109-1120; DOI: [10.1109/TBDATA.2021.3075727](https://doi.org/10.1109/TBDATA.2021.3075727)
  这是一篇关于联邦学习的综合性综述，涵盖其定义、架构、包括系统异构性在内的各种挑战，并讨论了该领域的当前方法和未来研究方向。
- [When Does Federated Averaging Not Benefit From More Local Steps?](https://proceedings.mlr.press/v130/wang21a.html) — Jianyu Wang and Gauri Joshi (2021)
  Journal: Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS 2021); Publisher: PMLR; Volume: 130; Pages: 1367-1375; DOI: [10.1609/aistats.v24i1.18978](https://doi.org/10.1609/aistats.v24i1.18978)
  该论文从理论上分析了局部步数对联邦平均收敛性的影响，阐明了FedNova旨在解决的潜在问题，以在异构环境中提高性能。

---

[上一节](03-SCAFFOLD-%20%E8%81%94%E9%82%A6%E4%BC%98%E5%8C%96%E4%B8%AD%E7%9A%84%E6%96%B9%E5%B7%AE%E9%99%8D%E4%BD%8E.md) · [下一节](05-%E6%8B%9C%E5%8D%A0%E5%BA%AD%E5%AE%B9%E9%94%99%E8%81%9A%E5%90%88%E6%96%B9%E6%B3%95.md)
