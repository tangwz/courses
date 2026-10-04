---
course: "intro-to-reinforcement-learning"
chapter: "introduction-deep-q-networks-dqn"
lesson: "challenges-neural-networks-rl"
sourceId: 1659
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-7-introduction-deep-q-networks-dqn/challenges-neural-networks-rl"
title: "强化学习中神经网络的难题"
description: "讨论在强化学习中训练神经网络时，由相关数据和移动目标引起的不稳定性问题。"
order: 2
plots: []
sourceHash: "e562a40069342b733bf8a5037c4969c1a9552e37a5c3929847f42a3d1fb60b29"
sourceCorrections: []
---

虽然使用深度神经网络 (neural network)近似动作价值函数 $Q(s, a; \theta)$ 看起来像是对标准函数近似技术的自然延伸，但将它们直接应用于 Q-learning 框架会带来显著的障碍。神经网络表现良好的标准监督学习 (supervised learning)方法，通常依赖于在强化学习 (reinforcement learning)环境中不成立的假设。训练稳定性成为一个主要顾虑，原因有两方面：数据的序列性以及目标值不断变化。

### 相关数据的问题

在典型的监督学习 (supervised learning)场景中，训练数据由被认为是独立同分布（i.i.d.）的样本组成。这种独立性使得随机梯度下降 (gradient descent)（SGD）等优化算法能够通过采样小批量数据稳步进行，这些数据提供了对真实梯度的相对无偏估计。

然而，强化学习 (reinforcement learning)会按顺序生成数据。智能体随时间与环境交互，产生一系列经验：$(s_t, a_t, r_{t+1}, s_{t+1}), (s_{t+1}, a_{t+1}, r_{t+2}, s_{t+2}), \dots$。这个序列中连续的样本通常高度相关。状态 $s_{t+1}$ 强烈依赖于 $s_t$ 和 $a_t$。如果我们根据经验的到来顺序训练神经网络 (neural network)，就会违反 i.i.d. 假设。

为什么这种相关性不利？

1. **学习效率低下：** 在高度相关的样本上训练会导致学习效率低下。网络可能会过度拟合最近的经验，忘记它从过去可能不同情境中学到的东西。想象一下，学习玩一个游戏时只反复练习第一关；你可能在那一关变得非常出色，但在第二关却会完全失败。相关的更新可以将网络参数 (parameter)推向状态-动作空间中与时间紧密相连的一小部分区域，可能会抵消之前的学习。
2. **梯度方差：** 尽管长期来看，相关的样本不一定会使梯度估计产生偏差，但它们会使小批量数据中的方差明显增加。这使得训练过程更嘈杂，收敛速度可能更慢。

### 移动目标的不稳定性

第二个主要难题来自于 Q-learning 更新其估计值的方式。回顾 Q-learning 更新中用于转换 $(s, a, r, s')$ 的目标值：


$$
\text{目标} = r + \gamma \max_{a'} Q(s', a'; \theta)
$$


这里，$\theta$ 表示我们近似 Q 函数的神经网络 (neural network)的参数 (parameter)。我们希望调整网络的预测 $Q(s, a; \theta)$，使其更接近这个目标值。损失函数 (loss function)，通常是均方误差（MSE），会是下面这样：


$$
L(\theta) = \mathbb{E} \left[ \left( \underbrace{r + \gamma \max_{a'} Q(s', a'; \theta)}_{\text{目标}} - \underbrace{Q(s, a; \theta)}_{\text{预测}} \right)^2 \right]
$$


请注意，参数 $\theta$ 既出现在预测 $Q(s, a; \theta)$ 中，也出现在目标 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 中。当我们执行梯度下降 (gradient descent)步骤来最小化这个损失时，我们正在调整 $\theta$。但是当 $\theta$ 改变时，目标值本身也会发生变化，因为它也依赖于 $\theta$。

这就产生了“移动目标”问题。我们正试图让网络的预测与一个每一步更新都会改变的目标相匹配。这种耦合可能导致反馈循环和不稳定：旨在减少误差的更新可能无意中以一种在下一步中增加误差的方式移动目标，可能导致网络参数的振荡甚至发散。这就像你根据上一次射击调整瞄准时，目标每次都会突然移动一样。

这两个问题，相关数据和非平稳目标，意味着天真地将标准 Q-learning 与深度神经网络结合通常无法收敛或产生不稳定结果。应对这些难题对于使深度强化学习 (reinforcement learning)变得实用非常重要，从而促成了经验回放（Experience Replay）和固定 Q 目标（Fixed Q-Targets）等技术的发展，我们接下来会讨论这些技术。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press; Pages: Chapters 6, 11, 16
  这本权威教材的第二版为Q学习和函数近似提供了坚实基础，详细阐述了它们结合时出现的问题，并为深度强化学习奠定了基础。
- [UCL Course on Reinforcement Learning - Lecture 6: Deep Q-Networks](https://github.com/zyxue/youtube_RL_course_by_David_Silver) — David Silver (2015)
  本讲座清晰地解释了深度Q学习中相关数据和非平稳目标的问题，以及DQN如何解决这些问题。
