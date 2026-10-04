---
course: "advanced-reinforcement-learning"
chapter: "rl-implementation-optimization"
lesson: "distributed-rl"
sourceId: 3500
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-8-rl-implementation-optimization/distributed-rl"
title: "分布式强化学习方法"
description: "扩展强化学习训练的背后原理（例如，分布式数据收集，分布式训练）。"
order: 6
plots: []
sourceHash: "290d8fed0e75d5051ff77ab667f5bcd3dfd160c9274dfb063654b27e619ef19e"
sourceCorrections: []
---

训练复杂的强化学习 (reinforcement learning)智能体，特别是那些在大状态空间上使用深度神经网络 (neural network)或需要大量环境交互的智能体，常常会超出单机计算的能力。正如本章前面关于优化部分所讨论的，扩展学习过程变得很有必要。分布式强化学习提供了并行计算的技术，主要侧重于加速两个主要瓶颈：环境交互（数据收集）和模型训练（梯度计算）。借助多台机器或多个处理单元，我们可以大幅减少训练出高效策略所需的实际运行时间。

### 扩展数据收集借助并行执行器

"强化学习 (reinforcement learning)中最耗时的方面之一是与环境交互以收集经验。模拟器可能运行缓慢，且交互本身受时间限制。一种常见且有效的方法是使用多个“执行器”进程并行化这个数据收集过程。"

每个执行器运行一个环境实例，并使用当前策略的副本与其交互。这些执行器独立并并行运行，收集轨迹（状态、动作、奖励、下一状态的序列）。收集到的经验随后被汇总，通常发送给一个中心学习器或共享经验回放缓冲区。

> 典型的分布式数据收集配置。多个执行器并行与环境实例交互，将收集到的经验发送到中心经验回放缓冲区，该缓冲区将数据提供给学习器进行策略更新。更新后的策略会定期分发回执行器。

这种方法显著增加了单位时间收集到的经验量。IMPALA 和 SEED RL 等架构非常依赖这一原理。

### 扩展训练借助分布式计算

第二个主要瓶颈是训练代表策略和/或价值函数的神经网络 (neural network)。随着模型变大和数据集（经验回放缓冲区）增长，梯度计算和参数 (parameter)更新变得计算密集。这里可以应用分布式深度学习 (deep learning)的标准技术。

- **数据并行：** 这是最常见的方法。模型在多个处理单元（例如，GPU 或 TPU）上复制。数据集（或从经验回放缓冲区采样的批次）被分区，每个副本处理不同的分区以计算梯度。这些梯度随后通过参数服务器或更高效的全规约算法（例如 Horovod 或 NCCL 常用的环形全规约）进行汇总（例如，求平均），然后在所有副本上一致地更新模型参数。

> 分布式训练中的数据并行。一个数据批次被拆分到多个设备上，每个设备都持有一个模型副本。梯度在本地计算，然后进行汇总，之后再更新所有副本上的模型参数。

- **模型并行：** 对于无法放入单个加速器内存的超大型模型，模型本身可以在多个设备之间分区。每个设备只计算模型操作的一部分。这在标准强化学习 (reinforcement learning)应用中不太常见，但可能与超大型基于 Transformer 的策略或复杂模型相关。

### 分布式强化学习 (reinforcement learning)架构

有几种有影响力的架构结合了分布式数据收集和训练原理：

1. **异步优势Actor-Critic (A3C)：** 早期成功的分布式方法之一。A3C 使用多个工作器，每个工作器都有自己的环境实例和模型副本。工作器根据其交互在本地计算梯度，并异步更新一个中心全局模型。虽然它开创了先河，但异步更新可能导致工作器使用“过时”策略（未反映最新更新的策略），与同步方法相比，这可能导致不稳定或次优收敛。它通常不使用经验回放缓冲区。
2. **同步优势Actor-Critic (A2C)：** 一种同步变体，在这种变体中协调器等待所有执行器完成一段经验收集。梯度被计算（通常在执行器上或中心地），并在对中心模型应用单个更新步骤之前进行平均。这通常比 A3C 带来更稳定的训练和更好的硬件利用率（特别是 GPU），但可能受限于最慢的工作器。
3. **IMPALA (重要性加权Actor-Learner架构)：** 专为大规模并行设计。它显著分离了执行和学习。许多执行器运行策略的不同版本（落后于学习器），并将轨迹发送到中心学习器。学习器以大批次处理这些轨迹，使用一种称为 V-trace 的离策略校正技术来弥补策略滞后。这使得数据吞吐量 (throughput)极高。
4. **SEED RL (可扩展、高效的深度强化学习)：** 优化加速器（GPU/TPU）的使用。执行器只专注于环境步骤，并将观测值发送到中心学习器。运行在加速器上的学习器执行模型推断（为所有执行器计算动作）和训练更新（梯度计算）。这最大限度地减少了通信（只发送观测值和动作，而不是频繁发送完整的模型参数 (parameter)或梯度），并使强大的加速器得到高利用。

> A3C 和 SEED RL 架构的简化比较。A3C 使用工作器计算自己的梯度并进行异步更新。SEED RL 将模型推断和训练集中在加速器上，执行器主要处理环境交互。

### 实践考量

实施分布式强化学习 (reinforcement learning)系统会带来一些特有挑战：

- **通信：** 在网络上，可能在数百或数千个进程之间传输数据（经验、观测值、动作、梯度、参数 (parameter)）可能成为瓶颈。高效的序列化和网络带宽很重要。SEED RL 等架构通过中心化推断来最大限度地减少这一点。
- **同步：** 选择异步更新和同步更新涉及权衡。异步方法最初可能更简单，但可能因过时的梯度而导致不稳定。同步方法提供更高的稳定性，并且通常能更好地利用硬件，但需要协调机制（例如，屏障、全规约），并且可能因慢速工作器而变慢。
- **资源管理：** 协调大量 CPU 执行器和可能多个 GPU/TPU 学习器需要基础设施管理，通常使用 Kubernetes 等集群管理器或专用工具。
- **框架：** 像 Ray RLlib、Acme 或 Tianshou 这样的库提供了常见分布式模式（例如，并行环境、分布式经验回放缓冲区、具有分布式执行的 IMPALA 或 PPO 等特定算法）的高级抽象和实现，显著简化了开发。

分布式系统对于扩展强化学习的实现能力非常有帮助。通过仔细权衡不同并行策略之间的利弊，并借助合适的框架，您可以在比单机更具挑战性的问题上更高效地训练更复杂的智能体。

## 参考资料

- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (2016)
  Journal: ICML 2016; Pages: 1928-1937; DOI: [10.48550/arXiv.1602.01783](https://doi.org/10.48550/arXiv.1602.01783)
  介绍A3C，一种基础的异步分布式强化学习算法，对后续许多方法产生了影响。
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](http://proceedings.mlr.press/v80/espeholt18a.html) — Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 80; Pages: 1407-1416; DOI: [10.5555/3305890.3305963](https://doi.org/10.5555/3305890.3305963)
  提出了IMPALA架构，通过使用V-trace离策略校正，显著提升了分布式强化学习的可扩展性和吞吐量。
- [Distributed Reinforcement Learning: A Survey](https://www.sciencedirect.com/science/article/pii/S092523122030990X) — Xin Zhang, Bo Zhang, Hanjiang Hu, Chengdong Hu, Wei He (2020)
  Journal: Neurocomputing; Publisher: Elsevier; Volume: 418; Pages: 19-33; DOI: [10.1016/j.neucom.2020.08.006](https://doi.org/10.1016/j.neucom.2020.08.006)
  提供了分布式强化学习方法的全面综述，涵盖了不同的架构、挑战和未来发展方向。
