---
course: "optimization-techniques-ml"
chapter: "distributed-ml-optimization"
lesson: "synchronous-vs-asynchronous-updates"
sourceId: 1292
sourceUrl: "https://apxml.com/zh/courses/optimization-techniques-ml/chapter-5-distributed-ml-optimization/synchronous-vs-asynchronous-updates"
title: "同步更新与异步更新"
description: "比较分布式设置中同步和异步更新策略的权衡利弊。"
order: 3
plots: []
sourceHash: "aead6c05347c39046df9e9c291f8ea241bbbf232f2f3ed07113b3a3e92ef7b89"
sourceCorrections: []
---

在分布式环境中训练模型，当多个工作节点处理不同数据子集时，一个基本问题浮现：各个工作节点计算的更新应如何合并以更新共享模型参数 (parameter)？两种主要策略是同步更新和异步更新，每种都在计算效率、通信开销和收敛行为方面呈现出各自的权衡。

### 同步更新：保持一致性但需等待

分布式模型训练涉及到多个工作节点处理不同的数据子集。如何将个体工作节点计算的更新结合起来，以更新共享模型参数 (parameter)，是一个根本问题。两种主要策略是同步更新和异步更新，每种策略在计算效率、通信开销和收敛行为之间呈现出不同的权衡。

1. **广播：** 一个中心实体，通常是参数服务器或指定的工作节点，将当前模型参数 ($w_t$) 广播给所有 $N$ 个工作节点。
2. **计算梯度：** 每个工作节点 $i$ 使用其本地数据批次和*完全相同*的参数版本 $w_t$ 计算梯度 $\nabla f_i(w_t)$。
3. **通信与聚合：** 工作节点将其计算的梯度发送回中心实体。重要的是，系统会等待所有工作节点报告其梯度。
4. **更新：** 中心实体聚合梯度（通常通过求平均值：$\frac{1}{N} \sum_{i=1}^N \nabla f_i(w_t)$），并将更新应用于模型参数：$w_{t+1} = w_t - \eta (\frac{1}{N} \sum_{i=1}^N \nabla f_i(w_t))$。
5. **重复：** 流程使用更新后的参数 $w_{t+1}$ 从步骤1重复。

> 同步 SGD 的工作流程。工作节点基于相同的模型版本 ($w_t$) 计算梯度，并在聚合梯度更新中心模型之前在同步屏障处等待。

**同步更新的优点：**

- **算法简单性：** 其行为与在单机上执行的标准小批量 SGD 密切相似，只是有效批量大小更大（所有工作节点的小批量之和）。
- **收敛性分析：** 理论收敛性质通常直接从顺序 SGD 分析中推导，使其更容易推断收敛速度和超参数 (hyperparameter)调优。
- **一致性：** 单个更新步骤中的所有梯度计算都使用相同的模型状态，确保更新基于最新的聚合信息。
- **可复现性：** 假设相同的初始状态、数据分区和随机种子，同步训练运行是确定性且可复现的。

**同步更新的缺点：**

- **慢节点问题：** 整个系统的速度由每次迭代中最慢的工作节点决定。如果一个工作节点因硬件差异、网络问题或遇到特别复杂的数据批次而变慢，所有其他工作节点都必须空闲等待其完成。这会显著降低整体吞吐量 (throughput)和硬件利用率。
- **同步开销：** 屏障同步本身会引入延迟。同时协调多个工作节点可能导致通信瓶颈，特别是在工作节点数量增加时。

### 异步更新：通过独立性最大化吞吐量 (throughput)

异步更新方案，例如异步随机梯度下降 (gradient descent)（Async-SGD），通过移除同步屏障来优先考虑工作节点利用率和系统吞吐量。工作节点更独立地运行：

1. **拉取参数 (parameter)：** 工作节点 $i$ 从参数服务器请求最新可用参数，例如 $w_{\tau_i}$。$\tau_i$ 表示工作节点 $i$ 接收到的参数版本或时间戳。
2. **计算梯度：** 工作节点使用其本地数据和接收到的参数 $w_{\tau_i}$ 计算梯度 $\nabla f_i(w_{\tau_i})$。
3. **推送更新：** 工作节点将其计算的更新（例如 $-\eta \nabla f_i(w_{\tau_i})$）发送回参数服务器。
4. **应用更新（服务器端）：** 参数服务器接收到来自工作节点 $i$ 的更新，并*立即*将其应用于当前参数版本 $w_{current}$：$w_{new} = w_{current} - \eta \nabla f_i(w_{\tau_i})$。请注意，$w_{current}$ 可能已经包含了在工作节点 $i$ 拉取 $w_{\tau_i}$ 之后到达的其他工作节点的更新。
5. **重复：** 工作节点立即通过拉取最新参数（这些参数可能已经反映了它自己的上一次更新和来自其他节点的更新）来开始下一个循环。

> 异步 SGD 的工作流程。工作节点独立地拉取参数、计算梯度并推送更新，无需等待。参数服务器立即应用更新，可能基于陈旧的参数 ($w_\tau$)。

**异步更新的优点：**

- **更高的吞吐量：** 工作节点之间互不等待。较快的工作节点可以在相同的时间内执行更多更新，可能导致更快的初期进展。
- **提高硬件利用率：** 消除了因在同步屏障处等待而导致的空闲时间，特别有利于异构环境。
- **慢节点容忍：** 慢的工作节点不会阻碍快的工作节点，缓解了同步方法中固有的慢节点问题。

**异步更新的缺点：**

- **梯度陈旧性：** 这是主要挑战。当工作节点 $i$ 基于 $w_{\tau_i}$ 的更新在服务器（时间 $t$）应用时，服务器的参数可能已经由其他工作节点更新了多次。因此，梯度 $\nabla f_i(w_{\tau_i})$ 相对于其所应用的参数 $w_t$ 是“陈旧”的。陈旧性（通常衡量为 $t - \tau_i$）随工作节点数量增加和通信延迟增高而增加。
- **收敛问题：** 陈旧梯度会给优化过程引入噪声。更新可能基于过时信息，如果管理不当（例如，通过降低学习率），可能导致次优收敛路径、振荡，甚至发散。尽管在某些假设下（如有限陈旧性）收敛性通常可以被证明，但其*统计效率*（每单位计算/梯度评估的收敛）可能低于 Sync-SGD。
- **调试与可复现性：** 更新时序的非确定性使得调试和复现特定训练运行变得极其困难。
- **实现复杂性：** 管理对参数服务器的并发访问和更新需要谨慎实现，以确保正确性并避免竞争条件，尽管成熟的框架通常会处理这些。

### 异步更新中的陈旧性问题

异步 SGD 的主要问题在于，工作节点 $i$ 基于参数 (parameter) $w_{\tau_i}$（来自时间 $\tau_i$ 的参数）计算的梯度 $\nabla f_i(w_{\tau_i})$，被用于更新参数服务器在稍后时间 $t$ 当前持有的参数 $w_t$。差值 $w_t - w_{\tau_i}$ 表示其他工作节点在间隔 $(\tau_i, t]$ 期间所做的更改。

如果这个差值很大（陈旧性高），梯度 $\nabla f_i(w_{\tau_i})$ 可能不是当前点 $w_t$ 的梯度 $\nabla f_i(w_t)$ 的良好近似。这种差异会阻碍收敛。理论分析通常表明，如果学习率足够小且陈旧性有界，Async-SGD 可以收敛，但所需的学习率可能小于 Sync-SGD，可能在迭代次数方面减慢收敛，即使每次迭代的实际时间更快。

### 在同步更新和异步更新之间做出选择

该决定需要平衡系统效率（吞吐量 (throughput)、利用率）与统计效率（每次迭代的收敛质量和速度）。

| 特性 | 同步更新 (Sync-SGD) | 异步更新 (Async-SGD) |
| --- | --- | --- |
| **协调方式** | 步调一致；每次迭代的屏障同步 | 独立操作；无需等待 |
| **梯度使用** | 梯度基于*相同*的模型版本 ($w_t$) 计算 | 梯度基于可能*陈旧*的版本 ($w_\tau$) 计算 |
| **吞吐量** | 受最慢工作节点限制（慢节点效应） | 潜力更高；不被慢工作节点阻塞 |
| **系统利用率** | 工作节点在等待时可能空闲 | 工作节点通常保持忙碌 |
| **通信** | 突发性；需要屏障同步 | 更连续；无屏障开销 |
| **收敛性** | 分析更简单；通常更稳定 | 分析复杂；可能不稳定或整体更慢 |
| **陈旧性** | 迭代内无梯度陈旧性 | 梯度陈旧性是固有的 |
| **可复现性** | 更易复现（给定固定种子/数据） | 因非确定性时序而更困难 |
| **实现难度** | 屏障逻辑更简单 | 需要谨慎处理并发更新 |

**实际考量：**

- **网络带宽与延迟：** 高延迟网络会加剧 Async-SGD 中的陈旧性问题，并增加 Sync-SGD 中的等待时间。低延迟、高带宽网络有利于 Sync-SGD，或使 Async-SGD 的陈旧性问题不那么突出。
- **工作节点数量：** 在工作节点数量非常多的情况下，Sync-SGD 屏障变得开销大，并且 Async-SGD 的陈旧性增加。
- **硬件同构性：** 在同构集群中，Sync-SGD 的慢节点问题不那么严重。异构集群通常更能从 Async-SGD 对不同工作节点速度的容忍中获益。
- **模型与任务：** 某些模型/任务可能比其他模型/任务对陈旧梯度更敏感。

尽管 Async-SGD 最初承诺显著加速，但实践经验和研究表明，陈旧性的不利影响可以抵消吞吐量优势，特别是在最终模型准确性方面。Sync-SGD，尽管存在慢节点问题，通常提供更稳定和可预测的收敛，使其成为一个常见的默认选择，特别是在使用高效通信协议（如 All-Reduce，稍后讨论）时。混合方法，如陈旧同步并行（SSP），允许有限的陈旧量，试图找到一个中间点。最终，最优选择在很大程度上取决于具体的硬件环境、网络条件和机器学习 (machine learning)任务的特征。

## 参考资料

- [Delay-Tolerant Distributed Stochastic Gradient Descent](https://proceedings.neurips.cc/paper/2015/file/5a5e389e03cc09ad4d0ab02a64797036-Paper.pdf) — Xiangru Lian, Ce Zhang, Cho-Jui Hsieh, Kai-Wei Chang, Inderjit S. Dhillon (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 28); Publisher: Proceedings of Machine Learning Research; Volume: 48; Pages: 233-241
  这篇论文为异步随机梯度下降（SGD）提供了理论收敛分析，特别讨论了梯度陈旧性的影响并提出了容忍延迟的算法。
- [Stale Synchronous Parallel Distributed Machine Learning](https://proceedings.neurips.cc/paper_files/paper/2013/file/b7bb35b9c6ca2aee2df08cf09d7016c2-Paper.pdf) — Qirong Ho, James Cipar, Henggang Cui, Seunghak Lee, Jin Kyu Kim, Phillip B. Gibbons, Garth A Gibson, Greg Ganger, Eric P Xing (2013)
  Journal: Advances in Neural Information Processing Systems 26; Publisher: Neural Information Processing Systems (NIPS); Volume: 26; Pages: 1223-1231
  这篇论文介绍了陈旧同步并行（SSP）范式，这是一种混合方法，通过允许有限的梯度陈旧性来平衡同步和异步更新的优点。
- [Distributed Deep Learning: A Guide to the State-of-the-Art](https://dl.acm.org/doi/10.1145/3291040) — Tal Ben-Nun, Torsten Hoefler (2019)
  Journal: Communications of the ACM; Publisher: Association for Computing Machinery (ACM); Volume: 62; Pages: 59-69; DOI: [10.1145/3291040](https://doi.org/10.1145/3291040)
  这篇综述全面概述了分布式深度学习技术，包括对同步和异步训练策略、通信模式及其权衡的详细讨论。
