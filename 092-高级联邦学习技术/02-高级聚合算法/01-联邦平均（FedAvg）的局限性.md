# 联邦平均（FedAvg）的局限性

来源：[原文](https://apxml.com/zh/courses/federated-learning/chapter-2-advanced-aggregation-algorithms/fedavg-limitations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

联邦平均（FedAvg）在联邦学习中提供了一个简单直观的基准，但其有效性在实际场景中常受挑战。其局限性表明了更复杂的聚合方法的必要性。FedAvg的核心思想是客户端执行本地随机梯度下降 (gradient descent)（SGD）更新，中心服务器对生成的模型参数 (parameter)进行平均。然而，这种简单性却隐藏着一些在实际部署中常无法成立的假设。

让我们审视其主要不足：

### 统计异构性：非独立同分布数据问题

FedAvg最主要的局限性或许源于统计异构性，即各客户端的数据分布*并非*独立同分布（Non-IID）。在许多实际情形中（如移动键盘预测不同用户的下一个词，或医院根据其独特的病人群体训练模型），客户端数据本质上是个性化且多样化的。

考虑联邦学习旨在最小化的全局目标函数：


$$
F(w) = \sum_{k=1}^N p_k F_k(w)
$$


其中 $N$ 是客户端数量，$p_k$ 是客户端 $k$ 的权重 (weight)（通常与其数据集大小成比例），$F_k(w)$ 是客户端 $k$ 基于其本地数据 $D_k$ 的本地目标函数。

当数据为非独立同分布时，每个客户端 $k$ 的本地最优解 $w_k^* = \arg\min_w F_k(w)$ 可能显著不同，并且也与全局最优解 $w^* = \arg\min_w F(w)$ 不同。

在FedAvg的本地训练中，每个客户端的模型 $w_k$ 趋向于其*本地*最优解 $w_k^*$。如果客户端执行多次本地梯度下降 (gradient descent)步骤（在FedAvg中为减少通信而常用），它们的模型可能彼此之间以及与当前全局模型 $w^t$ 显著偏离。这种现象常被称为**客户端漂移**。

当服务器平均这些漂移的本地模型时（$w^{t+1} = \sum_{k \in S_t} p_k w_k^{t+1}$，其中 $S_t$ 是第 $t$ 轮选择的客户端集合），形成的全局模型 $w^{t+1}$ 可能不是一个好的折衷方案。它可能会在所有客户端上表现不佳，甚至发散。简单平均隐式假定本地最优解的平均值接近全局最优解，这在显著的非独立同分布条件下会失效。

> 由于非独立同分布数据，客户端模型向本地最优解（彩色圆圈）训练，这些解可能远离全局最优解（星形）。平均这些漂移的模型可能会导致次优的全局模型（橙色圆圈）。

这种因非独立同分布数据导致的客户端漂移会导致：

- **收敛速度慢：** 全局模型可能振荡或需要多轮才能收敛。
- **最终准确性降低：** 收敛后的全局模型性能可能显著差于在集中式数据上训练的模型。
- **潜在发散：** 在极端情况下，训练过程可能完全无法收敛。

### 系统异构性：应对多样化客户端

联邦网络常由能力差异很大的客户端组成：

- **硬件：** 不同的CPU速度、内存容量以及可能的加速器（GPU、TPU）可用性。
- **网络：** 不稳定或缓慢的网络连接（尤其是在移动设备上）。
- **可用性：** 客户端可能不可预测地加入或退出训练过程。

标准的同步FedAvg，其中服务器在执行聚合之前等待来自选定客户端群体更新，对系统异构性特别敏感。整体进度由每轮中最慢的客户端决定（即“掉队者”问题）。这会严重减缓训练速度，尤其是在具有多样化设备的大规模部署中。

尽管存在异步联邦学习变体，但在这些设置中简单平均也面临挑战。在未适当处理的情况下，对基于不同版本全局模型（陈旧性）计算出的模型或更新进行平均，会降低收敛性和稳定性。此外，如果客户端在发送更新之前执行不同数量的本地工作（例如，由于计算限制导致本地训练轮次可变），朴素的平均可能会不当地权衡它们的贡献。

### 对恶意更新的敏感性（拜占庭故障）

FedAvg在平均步骤中平等对待所有客户端更新。它假设客户端是诚实的并正确遵循协议。然而，一个恶意客户端（常被称为拜占庭客户端）可能会故意发送损坏或中毒的更新，旨在降低全局模型性能或植入后门。

由于FedAvg只是简单平均参数 (parameter)：


$$
w^{t+1} = \frac{1}{|S_t|} \sum_{k \in S_t} w_k^{t+1} \quad \text{(非加权版本)}
$$


单个客户端发送具有极大数值的参数值，可能完全主导平均值，有效地劫持全局模型。FedAvg缺乏对此类攻击的固有鲁棒性。即使少量恶意参与者也能严重损害整个训练过程。

### 通信瓶颈和收敛速度

尽管常被视为优点，FedAvg的通信效率仍带来挑战。重复传输完整模型更新（对于深度学习 (deep learning)模型可能包含数百万参数 (parameter)）成本高昂，尤其是在受限网络（如移动连接）上。FedAvg旨在减少通信轮次*数量*，而非发送每次梯度更新，但每次通信载荷的*大小*仍然很大。

此外，FedAvg的收敛速度可能比集中式训练慢很多，尤其是在客户端更新方差大（由于采样或非独立同分布数据）时，或当本地步数 ($E$) 未经仔细调整时。平均过程本身是一种近似，客户端采样和本地更新引入的方差会阻碍向全局最小值的平稳收敛。

这些局限性表明，尽管FedAvg奠定了基础，但它在复杂的联邦学习环境中常显不足。本章后续部分将介绍先进的聚合算法，这些算法专门设计用于应对这些挑战，从而实现更高效、更准确的联邦学习系统。FedProx 解决非独立同分布数据导致的客户端漂移问题，SCAFFOLD 处理方差和漂移问题，FedNova 处理系统异构性问题，拜占庭鲁棒方法提供对恶意行为者的抵御能力。

## 参考资料

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf) — H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas, Felix Hanzely (2017)
  Journal: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: Proceedings of Machine Learning Research; Volume: 54; Pages: 1273-1282; DOI: [10.55982/aistats.2017.96](https://doi.org/10.55982/aistats.2017.96)
  介绍了联邦平均算法，该算法是讨论其局限性的基础。
- [Federated Optimization in Heterogeneous Networks](http://proceedings.mlr.press/v119/li20b/li20b.pdf) — Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, Virginia Smith (2020)
  Journal: Proceedings of Machine Learning Research; Volume: 119; Pages: 3464-3474
  分析了统计异质性（非IID数据）和客户端漂移对FedAvg收敛性和性能的影响。

---

[上一节](../01-%E8%81%94%E9%82%A6%E5%AD%A6%E4%B9%A0%E6%A0%B8%E5%BF%83%E5%8E%9F%E7%90%86%E9%87%8D%E6%B8%A9/06-%E8%AF%84%E4%BC%B0%E8%81%94%E9%82%A6%E5%AD%A6%E4%B9%A0%E7%B3%BB%E7%BB%9F.md) · [下一节](02-FedProx%EF%BC%9A%E5%A4%84%E7%90%86%E7%BB%9F%E8%AE%A1%E5%BC%82%E8%B4%A8%E6%80%A7.md)
