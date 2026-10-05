# 深度Q网络 (DQN) 算法

来源：[原文](https://apxml.com/zh/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/dqn-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

如第一章所述，当状态空间或动作空间变得非常大时，标准强化学习 (reinforcement learning)算法常常难以应对。为每个状态（或状态-动作对）显式表示值函数或策略在计算上变得不可行。函数逼近提供了一种解决办法，但简单的线性逼近器无法捕捉许多复杂问题中存在的复杂联系，例如从视频游戏中的原始像素数据学习，或从高维传感器输入进行机器人控制。

深度Q网络 (DQN) 结合了深度神经网络 (neural network)与Q学习算法。它不使用表格或线性函数，而是使用神经网络来逼近最优动作值函数 $Q^*(s, a)$。

### 使用神经网络 (neural network)逼近Q函数

其主要思想很简单：我们定义一个神经网络，通常称为Q网络，其参数 (parameter)为权重 (weight) $\theta$。该网络接收状态 $s$ 的表示作为输入，并输出一个Q值向量 (vector)，其中包含该状态下每个可能动作 $a$ 的Q值。我们将网络对给定状态 $s$ 和动作 $a$ 的输出表示为 $Q(s, a; \theta)$。

例如，在从屏幕像素学习玩Atari游戏时，常见的架构包括：

1. **输入：** 游戏屏幕的原始像素数据，通常经过预处理（例如，转换为灰度、下采样，并堆叠连续的几个时间步以捕捉运动）。
2. **卷积层：** 多个卷积层用于从输入图像堆栈中提取空间特征。它们的作用类似于计算机视觉任务中的特征提取器。
3. **全连接层：** 卷积层之后的一个或多个全连接层。
4. **输出层：** 一个带有线性激活函数 (activation function)的最终全连接层。输出神经元的数量等于智能体在环境中可以采取的有效动作的数量。每个输出神经元代表在输入状态下采取相应动作的估计Q值。

> DQN中用于处理基于图像输入的典型卷积神经网络 (CNN)架构。

### 训练Q网络：最小化贝尔曼误差

我们如何训练这个Q网络的权重 (weight) $\theta$？我们调整标准的Q学习更新规则。回顾最优动作值函数 $Q^*(s, a)$ 的贝尔曼方程：


$$
Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot|s,a)} [r + \gamma \max_{a'} Q^*(s', a')]
$$


Q学习迭代地逼近这个最优值，使用基于经验转移 $(s_t, a_t, r_t, s_{t+1})$ 的更新。在函数逼近的背景下，我们希望我们的网络 $Q(s, a; \theta)$ 满足这个方程。我们可以根据获得的奖励和下一个状态的估计值定义一个目标值 $y_t$：


$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)
$$


然后我们通过最小化目标 $y_t$ 与网络当前预测 $Q(s_t, a_t; \theta)$ 之间的差异来训练网络。一个常用的损失函数 (loss function)是均方误差 (MSE)，通常称为均方贝尔曼误差 (MSBE)：


$$
L(\theta) = \mathbb{E}_{(s_t, a_t, r_t, s_{t+1})} \left[ (y_t - Q(s_t, a_t; \theta))^2 \right]
$$


期望通常通过对从智能体经验中采样的迷你批次转移的平方误差求平均来近似。网络权重 $\theta$ 使用梯度下降 (gradient descent)方法（如RMSprop或Adam）进行更新以最小化此损失。

### 解决训练不稳定性

在早期尝试中，将此更新规则直接应用于非线性函数逼近器（如深度神经网络 (neural network)）被证明非常不稳定。这种不稳定性主要来自两个方面，与第一章中提到的“致命三要素”（函数逼近、自举和离策略学习）相关：

1. **相关样本：** 当智能体与环境交互时，观察到的转移序列 $(s_t, a_t, r_t, s_{t+1})$、$(s_{t+1}, a_{t+1}, r_{t+1}, s_{t+2})$ 等高度相关。按照经验顺序在连续样本上训练网络违反了大多数随机梯度下降 (gradient descent)优化方法所依赖的独立同分布（i.i.d.）数据假设。这可能导致学习效率低下和潜在的振荡。
2. **移动目标值：** 目标值 $y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$ 取决于当前的网络权重 (weight) $\theta$。由于权重 $\theta$ 在每一步都会更新以减少当前批次的损失，因此后续批次的目标值也会发生变化。这就像追逐一个移动的目标，可能导致学习过程振荡或发散。

### DQN的稳定性改进

最初的DQN论文提出了两种基本技术，以减轻这些不稳定性并使深度Q网络能够稳定训练：

1. **经验回放：** 不像直接对采集到的样本进行训练，转移 $(s_t, a_t, r_t, s_{t+1})$ 被存储在一个称为回放记忆或回放缓冲区 ($D$) 的大缓冲区中。在训练期间，会从该缓冲区中*随机*采样迷你批次的转移。这打破了批次内样本之间的时间相关性，从而带来更稳定和高效的学习。它还允许智能体多次重用过去的经验。(这将在“经验回放机制”部分详细说明)。
2. **目标网络：** 为了解决移动目标问题，DQN使用一个单独的、定期更新的目标网络，表示为 $\hat{Q}$，其参数 (parameter)为 $\theta^{-}$。这个目标网络与在线Q网络（参数为 $\theta$ 的 $Q$）具有相同的架构，但其权重 (weight)在一定步数内保持不变。目标值 $y_t$ 使用这个固定的目标网络计算：

   
   $$
   y_t = r_t + \gamma \max_{a'} \hat{Q}(s_{t+1}, a'; \theta^{-})
   $$
   

   然后使用梯度下降 (gradient descent)训练在线网络 $Q(s, a; \theta)$ 以最小化MSE损失：$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim D} [ (y_t - Q(s, a; \theta))^2 ]$。周期性地（例如，每 $C$ 个训练步），通过复制在线网络的权重来更新目标网络的权重：$\theta^{-} \leftarrow \theta$。这种固定目标网络的使用显著稳定了学习过程。(这将在“目标网络提高训练稳定性”部分详细说明)。

### DQN算法

综合这些思想，得到以下算法：

1. 初始化回放记忆 $D$ 到容量 $N$。
2. 用随机权重 (weight) $\theta$ 初始化在线Q网络 $Q$。
3. 用权重 $\theta^{-} = \theta$ 初始化目标Q网络 $\hat{Q}$。
4. 对于 `episode` = 1 到 `M`：
   a. 观察初始状态 $s_1$。
   b. 对于 `t` = 1 到 `T`（或直到回合结束）：
   i. 使用 $\epsilon$-贪婪策略选择动作 $a_t$，基于 $Q(s_t, \cdot; \theta)$（即，以概率 $\epsilon$ 选择随机动作，否则选择 $a_t = \arg\max_{a} Q(s_t, a; \theta)$）。
   ii. 在环境中执行动作 $a_t$。
   iii. 观察奖励 $r_t$ 和下一个状态 $s_{t+1}$。判断 $s_{t+1}$ 是否为终止状态。
   iv. 将转移 (s\_t, a\_t, r\_t, s\_{t+1}, \text{is\_terminal}) 存储在 $D$ 中。
   v. 从 $D$ 中随机采样一个包含 $K$ 个转移的迷你批次 $(s_j, a_j, r_j, s_{j+1}, \text{is\_terminal}_j)$。
   vi. 对于迷你批次中的每个转移 $j$，计算目标 $y_j$：
   如果 $\text{is\_terminal}_j$ 为真：
   $y_j = r_j$
   否则：
   $y_j = r_j + \gamma \max_{a'} \hat{Q}(s_{j+1}, a'; \theta^{-})$
   vii. 对在线网络参数 (parameter) $\theta$ 执行损失函数 (loss function)的梯度下降 (gradient descent)步骤：
   $L = \frac{1}{K} \sum_{j=1}^{K} (y_j - Q(s_j, a_j; \theta))^2$
   $\theta \leftarrow \theta - \alpha \nabla_{\theta} L$
   viii. 每 $C$ 步更新目标网络权重：$\theta^{-} \leftarrow \theta$。
   c. 可选地衰减 $\epsilon$。

> 深度Q网络 (DQN) 算法的示意图，描绘了智能体（包含在线和目标网络）、环境、回放缓冲区和训练更新机制之间的交互。

DQN的引入是一项里程碑式的成就，表明深度神经网络 (neural network)可以有效地用于强化学习 (reinforcement learning)任务的训练，使得智能体能够直接从像素等高维感知输入中学习复杂的策略。它奠定了许多后续深度强化学习进展的基础，本章后续部分将对一些进行考察。

## 参考资料

- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin A. Riedmiller (2015)
  Journal: Nature; Volume: 518; Pages: 529-533; DOI: [10.1038/nature14236](https://doi.org/10.1038/nature14236)
  提出了原始的深度Q网络算法，引入了经验回放和目标网络，以在Atari游戏等复杂环境中实现稳定学习。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press
  全面介绍了强化学习，包括Q学习、函数近似以及DQN等深度强化学习方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: The MIT Press
  一本关于深度学习的权威教科书，涵盖了神经网络架构（如卷积神经网络）和训练方法，这些对于理解DQN的“深度”方面至关重要。

---

[上一节](01-%E7%BA%BF%E6%80%A7%E5%87%BD%E6%95%B0%E8%BF%91%E4%BC%BC%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md) · [下一节](03-%E7%BB%8F%E9%AA%8C%E5%9B%9E%E6%94%BE%E6%9C%BA%E5%88%B6.md)
