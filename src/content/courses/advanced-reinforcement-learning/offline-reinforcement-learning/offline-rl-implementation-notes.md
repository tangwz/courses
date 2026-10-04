---
course: "advanced-reinforcement-learning"
chapter: "offline-reinforcement-learning"
lesson: "offline-rl-implementation-notes"
sourceId: 3493
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-7-offline-reinforcement-learning/offline-rl-implementation-notes"
title: "离线强化学习实现方面的考量"
description: "实现和评估离线强化学习算法的实际考量。"
order: 11
plots: ["plots/3493-0.json"]
sourceHash: "525be9cfa8727bb0ad5f2451d2c65d6970a68ff11bccfa585dfd767fc5e3142f"
sourceCorrections: []
---

成功地在离线设置中训练强化学习 (reinforcement learning)智能体，不仅仅是选择BCQ或CQL这样的算法。由于无法与环境交互以获取纠正反馈或直接评估策略，实现时的具体事项以及如何处理静态数据集变得格外重要。核心难题仍然是减轻分布偏移的影响，即学习到的策略可能查询在提供数据中表现不佳或完全不存在的状态-动作对。这里是有效实现离线强化学习算法的实际考量。

### 数据质量与准备

任何离线强化学习 (reinforcement learning)工作的出发点都是数据集本身。与在线学习不同，你不能通过收集更多或不同的数据来弥补糟糕的数据集。

- **数据集覆盖范围**：数据集必须充分覆盖与潜在最优策略相关的状态-动作空间。如果数据集只包含次优行为策略$\pi_b$的轨迹，那么学习一个显著更好的策略可能是不可能的，特别是如果从未访问过高回报区域。分析数据集：是什么行为生成了它？是否包含成功的轨迹？在相似的状态下采取的动作有多么不同？
- **数据量**：需要有足够的数据来可靠地估计值或策略梯度并进行泛化。所需的数据量很大程度上取决于环境的复杂性和行为策略的覆盖范围。
- **预处理**：标准的深度学习 (deep learning)实践在此适用。根据*离线数据集*的统计数据，将状态特征标准化为零均值和单位方差。如果回报的尺度非常大或非常小，则对其进行标准化，但请注意，改变回报可能会影响Q值的大小。高效地组织数据，通常以元组形式$(s, a, r, s', d)$。这里，$s$代表状态，$a$是行为策略采取的动作，$r$是获得的回报，$s'$是下一个状态，$d$是一个表示回合结束的布尔值。

### 算法选择与超参数 (parameter) (hyperparameter)调整

选择和调整离线强化学习 (reinforcement learning)算法需要仔细思考，因为在线试错并非一个可行的选项。

- **算法适用性**：当数据集已知包含接近最优行为时，或者需要严格遵循数据分布时，策略限制方法（如BCQ）通常很有效。如果数据集具有良好的覆盖范围但包含次优动作，值正则化 (regularization)方法（如CQL）可能更具适应性，并可能学习到更好的策略，因为它们侧重于惩罚OOD动作，而不是明确模仿行为策略。
- **超参数敏感性**：离线强化学习算法对超参数非常敏感。例如：
  - 在BCQ中，扰动模型中的阈值$\tau$显著影响哪些动作被视为“在分布内”。
  - 在CQL中，正则化系数$\alpha$控制着标准贝尔曼误差最小化与OOD动作Q值惩罚之间的权衡。较小的$\alpha$可能不足以对抗分布偏移，而较大的$\alpha$可能会过度抑制Q值，导致悲观的、次优的策略。
- **评估的难题**：由于在开发过程中不能在真实环境中运行学习到的策略，可靠的离线策略评估（OPE）对于超参数调整和模型选择来说是绝对必要的。

### 离线策略评估（OPE）的实践

OPE方法仅使用由$\pi_b$收集的静态数据集来估计所学策略$\pi$的性能。

- **OPE的必要性**：使用OPE来比较不同的算法，选择最佳超参数 (parameter) (hyperparameter)，并在潜在部署之前估计所选策略的最终性能。
- **方法**：
  - **重要性采样（IS）**：根据目标策略$\pi$和行为策略$\pi_b$之间的似然比重加权回报。期望总回报$J(\pi)$的基本IS估计量是：
    
    $$
    J(\pi) \approx \frac{1}{N} \sum_{i=1}^{N} \left( \prod_{t=0}^{T_i-1} \frac{\pi(a_{i,t} | s_{i,t})}{\pi_b(a_{i,t} | s_{i,t})} \right) R_i
    $$
    
    式中，$N$表示轨迹的数量，$R_i$是轨迹$i$的总回报，乘积项是累积重要性比率。IS存在高方差问题，特别是对于长时程或当$\pi$和$\pi_b$显著不同时。
  - **加权重 (weight)要性采样（WIS）**：对基本IS的一种常见改进，通常可以减少方差。
  - **鲁棒方法**：将基于模型的估计与重要性采样相结合，以平衡偏差和方差。
  - **基于模型的评估**：从离线数据中学习环境模型$\hat{P}(s'|s, a), \hat{R}(s, a)$，并用它来模拟在所学策略$\pi$下的推演。准确性受限于所学模型的保真度。
- **OPE的实践考量**：实现和验证OPE方法很复杂。从WIS等更简单的方法开始。始终使用数据集的一个保留部分专门用于评估，以避免评估指标过拟合 (overfitting)。监控重要性权重的方差；极高的方差表明估计不可靠。需要知道或估计行为策略$\pi_b$。如果$\pi_b$未知，可能需要使用行为克隆等方法从数据中估计。

> 工作流程图示了在没有在线交互的情况下，对超参数调整和策略选择，离线策略评估的依赖性。

### 软件与架构

尽管核心原理有所不同，但底层工具通常与在线强化学习 (reinforcement learning)重叠。

- **网络**：标准的前馈网络（MLPs）用于基于向量 (vector)的状态，或卷积网络（CNNs）用于基于图像的状态，通常用于Q函数和策略。如果观测是部分可观测的，可能需要循环网络（LSTMs, GRUs）。网络容量应足以捕捉值函数或策略的复杂性，但要避免对静态数据集过拟合 (overfitting)。
- **库**：TensorFlow、PyTorch和JAX等库是构建基础。强化学习特定框架（例如RLlib、Tianshou、Acme）可能提供一些实用工具，但通常需要为离线设置进行调整（例如，自定义数据加载器，集成CQL等特定离线算法）。确保您选择的库在OPE或BCQ等算法需要时，可以方便地访问策略概率$\pi(a|s)$。
- **训练稳定性**：目标网络（用于基于Q学习的方法）、梯度裁剪和仔细的学习率调度等技术仍然很重要。像CQL这样的值正则化 (regularization)方法会引入与惩罚项相关的自身稳定性考量。

### 离线强化学习 (reinforcement learning)的调试

由于缺乏交互式反馈，调试具有挑战性。重点是分析数据和算法的内部状态：

- **监控Q值**：特别是对于基于值或正则化 (regularization)的方法。Q值是否在失控增长？在CQL中，绘制数据集中存在的动作的Q值与同一状态下的随机动作的Q值。您应该会看到随机（很可能是OOD）动作的Q值相对于数据集中动作的Q值被抑制。



![CQL Q值惩罚](plots/3493-0.json)



> 示例说明了在给定状态下，CQL如何为数据集支持不佳的动作（随机动作）分配较低的Q值，而为批次中存在的动作（数据动作）分配较高的Q值。

- **检查策略行为（定性）**：检查学习到的策略$\pi$在从数据集中采样得到的状态下*会*采取的动作。它们与$(s, a, \dots)$元组中实际存在的动作$a$有多么不同？策略限制方法应该产生接近数据集动作的结果。
- **验证OPE**：如果可能，使用多种OPE方法。它们是否产生大致一致的估计？分析重要性权重 (weight)统计（均值、方差、最大值）。高方差是一个危险信号。
- **数据集切片**：尝试在数据的子集上进行训练或评估（例如，仅高回报轨迹），以了解敏感性。

实现离线强化学习需要从在线学习中转变思维方式。成功取决于仔细的数据分析、离线评估以及精心调整专门用于处理从固定数据集学习挑战的算法。

## 参考资料

- [A Survey on Offline Reinforcement Learning: Taxonomy, Challenges, and Techniques](https://ieeexplore.ieee.org/document/9769363) — Kai Wang, Chenjie Zeng, Xiang Zhang, Ruiquan Li, Minqiang Yu, and Mingkui Tan (2022)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 45; Pages: 2487-2508; DOI: [10.1109/TPAMI.2022.3168266](https://doi.org/10.1109/TPAMI.2022.3168266)
  全面概述了离线强化学习，包括其基本挑战、算法、数据考量和评估方法。
- [Off-Policy Deep Reinforcement Learning without Exploration](https://proceedings.mlr.press/v97/fujimoto19a.html) — Scott Fujimoto, David Meger, Doina Precup (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning; Publisher: PMLR; Volume: 97; Pages: 2052-2062; DOI: [10.48550/arXiv.1812.02900](https://doi.org/10.48550/arXiv.1812.02900)
  介绍了批量受限Q学习（BCQ），一种离线强化学习的策略约束方法，通过将动作限制在数据集中充分表示的范围内来解决分布偏移问题。
- [A Survey of Offline Policy Evaluation in Reinforcement Learning](https://ieeexplore.ieee.org/document/10310214) — Shengchao Liu, Fangfang Gao, Zongkai Ding, Jincai Huang, and Bo Yang (2023)
  Journal: IEEE Transactions on Artificial Intelligence; Publisher: IEEE; Volume: 4; Pages: 1076-1087; DOI: [10.1109/TAI.2023.3323091](https://doi.org/10.1109/TAI.2023.3323091)
  专门针对离线策略评估（OPE）方法的调查，详细介绍了重要性采样和基于模型的各种技术，并讨论了它们的偏差-方差权衡。
