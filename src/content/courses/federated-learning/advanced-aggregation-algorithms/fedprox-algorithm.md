---
course: "federated-learning"
chapter: "advanced-aggregation-algorithms"
lesson: "fedprox-algorithm"
sourceId: 3527
sourceUrl: "https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/fedprox-algorithm"
title: "FedProx：处理统计异质性"
description: "了解FedProx算法及其近端项如何帮助减轻因非独立同分布数据导致的客户端漂移。"
order: 2
plots: ["plots/3527-0.json"]
sourceHash: "39e2ae08ace293506962aaac4682891d53d6a4770a04f43c5493e4e1d0931a7d"
sourceCorrections: []
---

当客户端数据分布存在统计异质性（非独立同分布，Non-IID）时，标准FedAvg算法可能表现不佳。当客户端在聚合之前，对各自不同的数据集执行多次局部更新时，它们的局部模型可能会彼此之间以及与最优全局解明显偏离。这种现象常被称为“客户端漂移”。FedAvg简单的平均步骤并未明确抵消这种漂移，可能导致收敛速度变慢、震荡，甚至在严重情况下发散。

FedProx（带近端项的联邦优化）是一种专门为应对此难题而设计的算法。它对每个客户端所解决的局部优化问题进行修改，旨在限制每个局部模型在局部训练期间偏离当前全局模型的距离。

### 近端项

FedProx 的主要思想是在标准局部目标函数中加入一个近端项。客户端 $k$ 不再仅仅根据其局部数据 $D_k$ 最小化其局部损失 $F_k(w)$，而是在通信轮次 $t$ 的局部训练期间解决一个修改后的目标函数：


$$
\min_{w_k} L_{prox}(w_k; w^t) = F_k(w_k) + \frac{\mu}{2} ||w_k - w^t||^2
$$


其中：

- $w_k$ 表示由客户端 $k$ 在本地优化的模型参数 (parameter)。
- $F_k(w_k)$ 是客户端 $k$ 的标准局部损失函数 (loss function)（例如，其局部数据上的交叉熵损失）。
- $w^t$ 是在轮次 $t$ 开始时从服务器接收的全局模型。
- $||\cdot||^2$ 表示平方欧几里得范数（L2范数）。
- $\mu \ge 0$ 是一个控制近端项强度的超参数 (hyperparameter)。

### 工作原理：限制局部更新

项 $\frac{\mu}{2} ||w_k - w^t||^2$ 充当正则化 (regularization)惩罚项。它惩罚客户端更新的模型 $w_k$ 与该轮次开始时的初始全局模型 $w^t$ 之间的大幅偏离。

直观地理解：尽管客户端试图根据其特定数据最小化局部损失 $F_k(w_k)$，但近端项将解 $w_k$ 拉回到起始点 $w^t$。这会阻止局部模型过度拟合局部数据分布，并避免其过度偏离由 $w^t$ 代表的全局共识。

超参数 (parameter) (hyperparameter) $\mu$ 平衡着这种权衡：

- 如果 $\mu = 0$，FedProx 精确地退化为 FedAvg。局部优化只考虑局部损失 $F_k(w_k)$。
- 如果 $\mu > 0$，局部更新受到限制。更大的 $\mu$ 对偏离 $w^t$ 施加更强的惩罚，导致局部模型 $w_k$ 更接近初始全局模型。这增强了对抗异质性的稳定性，但如果 $\mu$ 过大，可能会略微减缓对局部数据的拟合过程。

为 $\mu$ 找到一个合适的值通常需要经验性调优，类似于机器学习 (machine learning)中的其他正则化超参数。

### FedProx 算法流程

FedProx 的整体流程与 FedAvg 相似，主要区别发生在客户端更新步骤：

1. **服务器初始化：** 初始化全局模型 $w^0$。
2. **通信轮次 (t = 0, 1, 2, ...):**
   - **服务器广播：** 服务器将当前全局模型 $w^t$ 发送给选定的客户端子集 $S_t$。
   - **客户端局部计算：** 每个选定的客户端 $k \in S_t$：
     - 接收 $w^t$。
     - 执行一定数量的步骤（例如，使用SGD进行 $E$ 个训练周期）以近似求解修改后的局部目标函数：$\min_{w_k} F_k(w_k) + \frac{\mu}{2} ||w_k - w^t||^2$。设得到的局部模型为 $w_k^{t+1}$。
     - 将更新后的模型 $w_k^{t+1}$ 发送回服务器。*注意：FedProx 不要求客户端精确地解决局部问题。*
   - **服务器聚合：** 服务器聚合接收到的局部模型，以形成新的全局模型 $w^{t+1}$。通常，使用基于局部数据集大小 $n_k$ 的加权平均，类似于FedAvg：
     
     $$
     w^{t+1} = \sum_{k \in S_t} \frac{n_k}{n} w_k^{t+1} \quad \text{其中 } n = \sum_{k \in S_t} n_k
     $$
     

### 优点与注意事项

**优势：**

- **非独立同分布数据上的稳定性提升：** 通过减轻客户端漂移，FedProx 在面对统计异质性时，通常会带来比 FedAvg 更稳定可靠的收敛。
- **理论收敛保证：** 即使在异构数据分布和客户端之间局部计算量不同的情况下，FedProx 也附带收敛保证。
- **对系统异质性的容忍度：** 尽管主要针对统计异质性，但近端项也使算法对不同客户端在本地执行的计算量变化（例如，不同数量的局部训练周期 $E_k$）更具容忍度。执行较少更新的客户端自然会保持更接近 $w^t$，符合近端目标。

**注意事项：**

- **超参数 (parameter) (hyperparameter)调优：** FedProx 的性能取决于 $\mu$ 的选择，需要针对特定问题和数据集进行调优。
- **实现：** 实现 FedProx 需要修改客户端的优化循环以包含近端项。这在大多数深度学习 (deep learning)框架中通常很简单，只需添加一个相对于该轮次*初始*模型 $w^t$ 的 L2 正则化 (regularization)项即可。

下面的图表展示了在非独立同分布设置下，FedProx 如何可能带来比 FedAvg 更稳定的收敛。



![收敛性：FedProx 与 FedAvg 在非独立同分布数据上的比较](plots/3527-0.json)



> FedAvg（红线）在异构数据上可能表现出更多的震荡行为或更慢的收敛，而 FedProx（蓝线）通过合适的 $\mu$ 可以减轻客户端漂移，从而实现更平滑且可能更高的最终准确率。

FedProx 通过明确纳入一种处理统计异质性的机制，代表了对 FedAvg 的显著改进。虽然它并非所有异质性挑战的完整解决方案，但其相对简单和有效性使其成为联邦学习实践者工作中的一个有用工具。在接下来的章节中，我们将审查 SCAFFOLD 和 FedNova 等其他高级聚合技术，它们采用不同的机制来应对异质性并改善收敛性。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) — H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: PMLR; Volume: 54; Pages: 1273-1282
  这篇基础论文介绍了联邦平均（FedAvg）算法，它是许多后续联邦学习算法的基准，其局限性正是 FedProx 旨在解决的问题。
