---
course: "advanced-reinforcement-learning"
chapter: "rl-implementation-optimization"
lesson: "rl-hyperparameter-tuning"
sourceId: 3496
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-8-rl-implementation-optimization/rl-hyperparameter-tuning"
title: "超参数调整策略"
description: "深度强化学习算法中超参数调整的技术和挑战。"
order: 2
plots: ["plots/3496-0.json"]
sourceHash: "3a6ddf2a445b6998cf2778221fafeabdec5774f6ab9eccdd3f2084827c05ecab"
sourceCorrections: []
---

尽管选取合适的网络架构提供了依据，但深度强化学习 (reinforcement learning)智能体要达到良好性能，通常取决于有效调整其众多超参数 (parameter) (hyperparameter)。与数据集静态的监督学习 (supervised learning)不同，强化学习涉及与环境的互动，这使得调整过程更复杂，计算要求更高。性能可能不平稳、敏感，寻找最佳配置需要有条理的方法。

### 深度强化学习 (reinforcement learning)中的超参数 (parameter) (hyperparameter)

不同的算法有各自的超参数集，但有几个是许多深度强化学习方法所共有的：

- **学习率($\alpha$)：** 控制更新网络权重 (weight)的步长。对于稳定性和收敛速度很重要。过高会导致训练发散；过低则训练缓慢或停滞。在Actor-Critic方法中，Actor网络和Critic网络通常需要不同的学习率。
- **折扣因子($\gamma$)：** 决定未来奖励相对于即时奖励的重要性（$0 \le \gamma \le 1$）。接近1的值强调长期奖励，适用于延迟满足的任务。接近0的值优先考虑即时奖励。如果奖励密集，将其设得过于接近1有时会导致不稳定或收敛缓慢。
- **网络架构：** 层数、每层神经元数量、激活函数 (activation function)、专用层的使用（例如，用于图像输入的卷积层，用于部分可观察性的循环层）。这些定义了函数逼近器的容量。
- **批量大小：** 每次梯度更新步中使用的样本数量。较大的批量能提供更稳定的梯度估计，但需要更多内存，有时可能导致更尖锐的局部最优，从而可能妨碍泛化。
- **回放缓冲区大小（离策略算法）：** 存储的过去经验的最大数量（例如在DQN、DDPG、SAC中）。较大的缓冲区能提供更多样的数据，但消耗更多内存，并且如果策略快速变化，可能会保留过时信息过久。
- **试探参数：**
  - **$\epsilon$-贪婪策略：** $\epsilon$的值及其衰减策略。确定试探（随机动作）和趋利（贪婪动作）之间的平衡。
  - **熵正则化 (regularization)系数（策略梯度/Actor-Critic）：** 鼓励策略随机性，促进试探。找到合适的平衡很重要；过高可能导致策略过于随机，过低可能导致过早收敛到次优策略。对于SAC等算法，此系数($\alpha$)有时可以自动调整。
- **更新频率/目标网络更新率($\tau$)：** 主网络更新的频率，或目标网络更新的频率/强度（例如通过Polyak平均）。影响稳定性和学习速度。

### 调整强化学习 (reinforcement learning)超参数 (parameter) (hyperparameter)的挑战

强化学习中的调整带来了独特的困难：

1. **高采样成本：** 训练一个强化学习智能体通常需要数百万次环境交互。运行多个超参数调整实验很快就会变得计算成本高昂。
2. **敏感性和不稳定性：** 性能会随着某些超参数（如学习率）的微小变化而显著改变。如果参数设置不当，许多算法容易发散。
3. **嘈杂的性能指标：** 环境和策略中固有的随机性导致奖励信号嘈杂。评估一组特定的超参数通常需要使用不同的随机种子运行多次，以获得可靠的性能估计。
4. **延迟效应：** 超参数选择的影响可能仅在训练过程后期才变得明显。
5. **相互依赖性：** 超参数通常以复杂的方式相互影响。孤立地优化一个参数可能无法获得最佳的整体配置。

### 系统调整策略

鉴于这些挑战，仅凭直觉进行手动调整通常是不够的。需要更有条理的方法：

#### 网格搜索

这涉及为每个超参数 (parameter) (hyperparameter)定义一组离散值，并评估所有可能的组合。尽管有条理，但它受到维度灾难的困扰。组合数量随着超参数的数量和每个参数测试值的数量呈指数增长。它通常在评估超参数空间中没有前景的区域时浪费计算资源。

#### 随机搜索

随机搜索不是使用固定的网格，而是从指定的分布（例如，均匀分布、对数均匀分布）中随机采样超参数。研究表明，在相同的计算预算下，它通常比网格搜索更高效，尤其是在只有少数超参数明显影响性能时。与侧重于特定轴的网格搜索相比，它不太可能忽略重要的相互作用。



![学习率和随机种子的影响](plots/3496-0.json)



> 示例显示了不同学习率和随机种子的性能变化。可靠的比较需要对多个种子进行平均。

#### 贝叶斯优化

这是一种基于模型的方法，用于寻找难以评估的目标函数（此处为智能体性能）的最小值或最大值。其工作原理是：

1. 基于先前评估过的点，建立超参数与性能之间关系的概率替代模型（通常是高斯过程）。
2. 使用采集函数（例如，预期改进、置信上限）来确定下一组要评估的超参数。采集函数平衡了对超参数空间不确定区域的试探性搜寻和对已知表现良好区域的优势发挥。

贝叶斯优化通常比网格搜索或随机搜索更样本高效，尤其是在评估成本高昂时。Optuna、Hyperopt和Ray Tune等工具提供了其实现。

#### 基于种群的训练（PBT）

PBT采取了不同的方法，即在训练过程中优化超参数。它维护着一个并行训练的智能体群体。定期地：

1. 表现不佳的智能体复制表现较好的智能体的模型权重 (weight)和超参数（趋利行为）。
2. 复制的超参数随后被随机扰动（试探行为）。

这使得PBT能够找到自适应的超参数策略，而非固定值，可能带来更好的最终性能和更快的收敛。它与分布式训练设置结合良好。

> 基于种群的训练（PBT）流程图。智能体定期进行评估，复制成功的配置，并扰动超参数。

### 实用建议

- **借鉴先验知识：** 从论文或可靠代码库（例如，Stable Baselines3 Zoo、RLlib）中提及的超参数 (parameter) (hyperparameter)范围和默认值开始。如有可能，请勿从零开始。
- **优先处理敏感性：** 识别已知敏感的超参数（通常是学习率、熵系数），并将初步调整精力集中于此。
- **使用对数尺度：** 对于像学习率或正则化 (regularization)系数这样可以跨越几个数量级的参数，应在对数尺度上采样它们（例如，$10^{-5}$到$10^{-2}$）。
- **运行多个种子：** 总是使用多个随机种子（至少3-5个，理想情况下更多）评估超参数设置，并报告性能的均值和标准差。这有助于区分真实的改进与随机运气。
- **自动化与跟踪：** 使用库（Ray Tune、Optuna、W&B Sweeps）自动化搜索过程，并使用工具（MLflow、Weights & Biases）细致地记录每个实验的超参数、代码版本和结果。这对于复现性和分析很重要。
- **早期停止：** 实现机制，根据中间性能指标及早停止没有前景的试验，以节省计算资源。
- **资源分配：** 在多个核心或机器上并行运行。云平台为大规模调整提供可扩展的资源。

深度强化学习 (reinforcement learning)中的超参数调整通常是一个迭代过程，涉及实验、分析和优化。尽管计算密集，但投入进行系统调整通常是发挥先进强化学习算法全部潜力的必要条件。

## 参考资料

- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) — James Bergstra and Yoshua Bengio (2012)
  Journal: Journal of Machine Learning Research; Volume: 13; Pages: 281-305
  介绍了随机搜索作为超参数优化的网格搜索的有效替代方法，表明当少数超参数重要时，其效果显著。
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://proceedings.neurips.cc/paper/2012/file/05311655a15b75a610cf007ac3aef859-Paper.pdf) — Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012)
  Journal: Advances in Neural Information Processing Systems (NIPS) 25; Publisher: NeurIPS Foundation; Volume: 25; Pages: 2951-2959
  为使用高斯过程调优机器学习算法提供了实用指南和演示，突出了其样本效率。
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu (2017)
  Journal: arXiv preprint arXiv:1711.09846; DOI: [10.48550/arXiv.1711.09846](https://doi.org/10.48550/arXiv.1711.09846)
  介绍了基于种群的训练（PBT），该方法在训练过程中联合优化超参数和神经网络权重，实现自适应的超参数调度。
