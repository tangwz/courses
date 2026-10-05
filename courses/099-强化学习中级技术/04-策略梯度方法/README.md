# 第 4 章：策略梯度方法

来源：[原章节](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods)

[返回课程目录](../README.md)

前几章着重介绍了像深度Q网络（DQN）这样的方法，它们学习在状态下采取行动的价值$Q(s, a)$。尽管有效，但这些基于价值的方法在某些情况下可能会遇到困难，例如在连续动作空间的环境中，或者当随机策略本身是必要时。

本章将介绍策略梯度方法，这是一种截然不同的策略。在这里，我们直接学习一个参数化的策略$\pi(a|s; \theta)$来选择行动，而不依赖中间的价值函数估计来决定行动。我们将首先讨论基于价值方法的局限性，正是这些局限性促使了这种替代方法的出现。

接着，我们将阐述策略梯度背后的主要思想：调整策略参数$\theta$以最大化预期回报。这需要了解策略梯度定理，它是这些方法的理论依据。您将学习实现REINFORCE算法，这是一种蒙特卡洛策略梯度的基本技术。我们还将解决REINFORCE算法中常遇到的高方差挑战，并提出使用基线来提高稳定性和收敛速度的方法。本章最后将提供REINFORCE算法的实践练习。

## 小节

- 1. [基于价值方法的局限性](01-%E5%9F%BA%E4%BA%8E%E4%BB%B7%E5%80%BC%E6%96%B9%E6%B3%95%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 2. [策略直接参数化](02-%E7%AD%96%E7%95%A5%E7%9B%B4%E6%8E%A5%E5%8F%82%E6%95%B0%E5%8C%96.md)
- 3. [策略梯度定理](03-%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E5%AE%9A%E7%90%86.md)
- 4. [REINFORCE 算法](04-REINFORCE%20%E7%AE%97%E6%B3%95.md)
- 5. [理解策略梯度中的方差](05-%E7%90%86%E8%A7%A3%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E4%B8%AD%E7%9A%84%E6%96%B9%E5%B7%AE.md)
- 6. [方差减少的基线](06-%E6%96%B9%E5%B7%AE%E5%87%8F%E5%B0%91%E7%9A%84%E5%9F%BA%E7%BA%BF.md)
- 7. [动手实践：实现REINFORCE算法](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0REINFORCE%E7%AE%97%E6%B3%95.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-reinforcement-learning/chapter-4-policy-gradient-methods/quiz)
