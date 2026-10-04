---
course: "advanced-reinforcement-learning"
chapter: "model-based-reinforcement-learning"
lesson: "learning-environment-models"
sourceId: 3443
sourceUrl: "https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-5-model-based-reinforcement-learning/learning-environment-models"
title: "学习环境动态模型"
description: "从经验中学习转移函数和奖励函数的方法，包括概率模型。"
order: 3
plots: []
sourceHash: "4541333fd4b3597cacf208bfdfe875bc7ccdde960ec00881fa8052c30e5ab3cb"
sourceCorrections: []
---

在基于模型的强化学习 (reinforcement learning)中，智能体主动建立自己对环境运作方式的理解。智能体不是仅仅依靠试错直接学习策略或价值函数，而是首先构建一个环境的*模型*。该模型通常包含两个主要部分：

1. **转移动态模型：** 给出当前状态 $s$ 和所执行的动作 $a$，此模型预测下一个状态 $s'$。形式上，它近似真实的环境转移概率分布 $P(s' | s, a)$。
2. **奖励函数模型：** 此模型预测在状态 $s$ 中执行动作 $a$ 并转移到状态 $s'$ 后获得的即时奖励 $r$。它近似真实的奖励函数，该函数可以表示为 $R(s, a)$ 或 $R(s, a, s')$。

学习环境动力学模型将强化学习问题（至少部分地）转换为监督学习 (supervised learning)问题。智能体通过与实际环境的交互收集经验元组 $(s_t, a_t, r_{t+1}, s_{t+1})$。这些收集到的数据用作学习转移模型和奖励模型的训练集。

### 学习转移模型

学习转移模型 $\hat{P}_{\theta}(s' | s, a)$ （$\theta$ 代表模型参数 (parameter)，通常是神经网络 (neural network)权重 (weight)）的复杂程度，很大程度上取决于环境是确定性的还是随机的，以及状态空间的性质。

#### 确定性转移

在状态 $s$ 中执行动作 $a$ 总是导致完全相同的下一个状态 $s'$ 的环境中，任务会大大简化。目标是学习一个函数 $\hat{s}' = f_{\theta}(s, a)$，它直接预测下一个状态。

- **方法：** 这是一个标准的回归问题。
- **函数逼近器：** 神经网络（如多层感知器，MLPs）常被使用。如果状态 $s$ 包含空间信息（如图像），可能会采用卷积神经网络 (CNN)（CNNs）。对于序列或历史依赖，循环神经网络 (RNN)（RNNs）或Transformer可能适用。
- **训练数据：** 从收集到的经验中获得的 $((s_t, a_t), s_{t+1})$ 对。
- **损失函数 (loss function)：** 一个常见的选择是预测的下一个状态 $\hat{s}'_{t+1} = f_{\theta}(s_t, a_t)$ 和实际观察到的下一个状态 $s_{t+1}$ 之间的均方误差（MSE）：
  
  $$
  L(\theta) = \frac{1}{N} \sum_{i=1}^{N} || f_{\theta}(s_i, a_i) - s_{i+1} ||^2
  $$
  
  其中 $N$ 是一个批次或数据集中数据点的数量。

#### 随机性转移

当环境是随机的，相同的 $(s, a)$ 对可以根据概率分布 $P(s' | s, a)$ 导致不同的下一个状态 $s'$。此时学习模型意味着近似整个分布，而不仅仅是单个结果。

- **方法：** 这需要学习条件概率分布 $\hat{P}_{\theta}(s' | s, a)$。
- **函数逼近器：** 同样，神经网络是标准工具，但它们的输出需要表示一个分布。常见的方法有：

  - **类别分布：** 如果状态空间是离散的（或离散化的），网络可以为每个可能的下一个状态 $s'$ 输出一个概率。损失函数通常是预测分布与观察到的 $s_{t+1}$ 的独热编码之间的交叉熵。
  - **高斯混合模型（GMMs）：** 对于连续状态空间，网络可以输出以 $(s, a)$ 为条件的高斯混合模型参数（均值 $\mu_k$、标准差 $\sigma_k$ 和混合权重 $\pi_k$）。模型预测下一个状态分布为 $\hat{P}_{\theta}(s' | s, a) = \sum_k \pi_k(s, a) \mathcal{N}(s' | \mu_k(s, a), \sigma_k(s, a))$。损失函数是观察到的 $s_{t+1}$ 在此预测分布下的负对数似然（NLL）：
    
    $$
    L(\theta) = - \sum_{i=1}^{N} \log \hat{P}_{\theta}(s_{i+1} | s_i, a_i)
    $$
    
  - **基于流的模型或其他生成模型：** 对于图像等极高维状态空间，可以训练更先进的生成模型（归一化 (normalization)流、变分自编码器、生成对抗网络 (GAN)）来建模 $P(s'|s,a)$，通常需要专门的架构和训练过程。
- **训练数据：** 从经验中获得的 $(s_t, a_t, s_{t+1})$ 元组。

下图说明了训练两个模型的监督学习 (supervised learning)设置：

> 从环境中收集的经验元组 $(s_t, a_t, r_{t+1}, s_{t+1})$ 用于通过监督学习训练独立的转移网络和奖励网络。这些网络根据当前状态和动作预测下一个状态（或其分布）和奖励。这些预测与实际结果进行比较，以计算损失信号，然后用这些信号来更新网络参数（$\theta$ 和 $\phi$）。

### 学习奖励模型

学习奖励模型 $\hat{R}_{\phi}(s, a, s')$ 或有时简化为 $\hat{R}_{\phi}(s, a)$，通常比学习转移动态更直接。奖励通常是标量值，这使得它成为一个标准的回归任务。

- **方法：** 预测预期奖励 $\hat{r} = g_{\phi}(s, a)$ 或 $\hat{r} = g_{\phi}(s, a, s')$。
- **函数逼近器：** 一个独立的神经网络 (neural network)（多层感知器通常足够）是常见的。
- **训练数据：** 从收集到的经验中获得的 $((s_t, a_t), r_{t+1})$ 或 $((s_t, a_t, s_{t+1}), r_{t+1})$ 对。
- **损失函数 (loss function)：** 均方误差常被使用：
  
  $$
  L(\phi) = \frac{1}{N} \sum_{i=1}^{N} ( g_{\phi}(s_i, a_i) - r_{i+1} )^2
  $$
  
  （如果奖励取决于下一个状态，则使用 $g_{\phi}(s_i, a_i, s_{i+1})$）。

### 模型学习中的考量

- **数据收集：** 数据 $(s, a, r, s')$ 的质量和覆盖范围显著影响学习模型的准确性。数据可以通过随机策略、当前学习到的策略或混合方式收集。智能体的数据获取策略直接影响可用于模型训练的数据。
- **模型偏差：** 学习到的模型 $\hat{P}_{\theta}, \hat{R}_{\phi}$ 几乎总是不完美的真实环境动态近似。模型中的误差在规划过程中会累积，如果智能体过于依赖有缺陷的模型，可能导致次优甚至有害的计划。这被称为模型偏差。
- **模型不确定性：** 智能体了解其模型预测的*信心程度*通常很有用。未被频繁访问的状态-动作空间区域的模型预测准确性可能会较低。使用贝叶斯神经网络 (neural network)或模型集成等方法可以帮助量化 (quantization)这种不确定性，然后可以将其纳入规划过程（例如，优先在已充分理解的状态-动作区域进行规划）。
- **计算成本：** 训练复杂的动态模型，特别是对于高维状态空间或随机环境，可能计算量很大。

一旦这些模型训练完毕，它们就能够进行规划。智能体可以使用学习到的 $\hat{P}_{\theta}$ 和 $\hat{R}_{\phi}$ 来模拟可能的轨迹，评估动作序列，并更新其策略或价值函数，而无需必然与真实环境进行进一步交互，正如我们将在后续讨论Dyna-Q和MCTS结合的章节中看到的那样。

## 参考资料

- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton, Andrew G. Barto (2018)
  Publisher: The MIT Press
  一本经典教材，提供了强化学习的基础知识，包括基于模型的强化学习原理和基本动态模型学习。
- [Model-Based Reinforcement Learning: A Survey](https://www.nowpublishers.com/article/Details/MAL-088) — Tim van de Moerland, Joost Pieterse, Marnix Suilen, Aske Plaat (2021)
  Journal: Foundations and Trends® in Machine Learning; Publisher: Now Publishers; Volume: 14; Pages: 1177–1233; DOI: [10.1561/2200000088](https://doi.org/10.1561/2200000088)
  对基于模型的强化学习的全面综述，涵盖了各种模型学习技术、规划方法以及模型不确定性和偏差等相关挑战。
- [Learning Latent Dynamics for Planning from Pixels](https://proceedings.mlr.press/v97/hafner19a.html) — Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, James Davidson (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning; Publisher: PMLR; Volume: 97; Pages: 2555-2565
  介绍了一种在随机环境中从高维观测（像素）学习潜在动态模型的方法，从而实现高效规划。
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://www.nature.com/articles/s41586-020-03051-4) — Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Gabriel Guli, Demis Hassabis, Thore Graepel, Timothy Lillicrap (2020)
  Journal: Nature; Publisher: Nature Portfolio; Volume: 588; Pages: 604–609; DOI: [10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4)
  介绍了MuZero，一个通过学习环境模型，在潜在空间中预测与规划相关的结果（价值、策略、奖励）的智能体，无需明确的游戏规则或状态表示知识。
