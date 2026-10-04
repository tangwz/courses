---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "linear-methods-vfa"
sourceId: 1653
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/linear-methods-vfa"
title: "价值函数逼近的线性方法"
description: "描述如何使用线性模型来逼近价值函数。"
order: 4
plots: ["plots/1653-0.json"]
sourceHash: "c346291ca00bda93ad77944dcf8583dcbefdbca9911aa50192411ac2bac85092"
sourceCorrections: []
---

对大规模问题，逼近价值函数是必要的，状态通常使用特征向量 (vector) $\mathbf{x}(s)$ 来表示。处理这些问题的一个最简单、最基本的方法是使用线性函数。

不同于为每个状态 $s$ 存储唯一价值的表格方法，我们通过状态特征的线性组合来逼近状态价值函数 $v_\pi(s)$。我们将这种逼近表示为 $\hat{v}(s, \mathbf{w})$，其中 $\mathbf{w}$ 是一个我们需要学习的权重 (weight)或参数 (parameter)向量。

### 状态价值的线性逼近

状态价值函数的线性逼近定义为权重 (weight)向量 (vector) $\mathbf{w}$ 与该状态的特征向量 $\mathbf{x}(s)$ 的点积：


$$
\hat{v}(s, \mathbf{w}) \doteq \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^{d} w_i x_i(s)
$$


此处：

- $\hat{v}(s, \mathbf{w})$ 是我们对状态 $s$ 的估计价值。
- $\mathbf{w} = [w_1, w_2, ..., w_d]^T$ 是权重列向量。这些是我们学习过程中调整的参数 (parameter)，以使我们的逼近 $\hat{v}$ 更接近真实价值 $v_\pi$。
- $\mathbf{x}(s) = [x_1(s), x_2(s), ..., x_d(s)]^T$ 是表示状态 $s$ 的特征列向量，如前一节所述。
- $d$ 是特征的数量。重要的是，$d$ 通常远小于状态总数 $|\mathcal{S}|$，这使得此方法对大规模状态空间可行。我们放弃了精确表示每个状态价值的能力，以换取一种更紧凑、可以在不同状态间泛化的表示方法。

核心思想是，具有相似特征向量的状态应具有相似的估计价值，由学习到的权重 $\mathbf{w}$ 确定。

### 动作价值的线性逼近

类似地，我们可以使用线性函数逼近动作价值函数 $q_\pi(s, a)$。这需要依赖于状态 $s$ 和动作 $a$ 的特征。我们将状态-动作对的特征向量 (vector)表示为 $\mathbf{x}(s, a)$。

动作价值函数的线性逼近是：


$$
\hat{q}(s, a, \mathbf{w}) \doteq \mathbf{w}^T \mathbf{x}(s, a) = \sum_{i=1}^{d} w_i x_i(s, a)
$$


此处：

- $\hat{q}(s, a, \mathbf{w})$ 是在状态 $s$ 中采取动作 $a$ 的估计价值。
- $\mathbf{w}$ 仍然是我们需要学习的权重 (weight)向量。注意，同样的权重可能被用于逼近所有动作的价值，或者有时会根据具体的特征构建使用不同的权重。目前，假设使用一个单一的权重向量。
- $\mathbf{x}(s, a)$ 是表示状态-动作对的特征向量。构建这些特征通常涉及将状态特征与关于动作的信息结合起来。例如，如果动作是离散的，可以有仅在考虑特定动作时才激活的单独特征，或者特征可以表示状态属性和动作选择之间的相互影响。

这种表述使我们能够估计 Q 价值，这对于在使用函数逼近时的 Q-learning 或 SARSA 等控制算法非常重要。

### 线性方法的优点

线性函数逼近具有多项优点，使其成为一种有价值的工具，并且常作为良好的起始选择：

1. **简洁性：** 数学形式简单明了，使其相对容易理解、实现和调试。
2. **计算效率：** 计算逼近价值涉及简单的点积运算。基于梯度下降 (gradient descent)的学习算法（我们接下来会讲到）对于线性模型也具有计算效率。
3. **良好的收敛特性：** 当与适当的学习算法（如基于梯度的方法）结合时，线性函数逼近通常具有更易于理解的收敛行为，相比更复杂的非线性逼近器。尽管在强化学习 (reinforcement learning)环境中（例如，由于自举）无法始终保证收敛到全局最优权重 (weight)，但线性方法通常是稳定的。
4. **可解释性：** 有时，学习到的权重 $w_i$ 可以提供关于不同特征 $x_i(s)$ 在决定价值方面的重要性的见解。



![线性价值函数逼近（一维示例）](plots/1653-0.json)



> 一个简单的线性函数逼近单个状态特征与其估计价值之间的关系。虚线代表 $\hat{v}(s, \mathbf{w})$，试图拟合从经验中获得的采样状态价值（蓝色圆点）。

### 特征的作用

需要记住的是，线性函数逼近的有效性很大程度上依赖于所选择的特征 $\mathbf{x}(s)$ 或 $\mathbf{x}(s, a)$。如果特征能捕捉真正影响价值的状态（或状态-动作对）的方面，那么线性组合可能是一个好的逼近。然而，如果特征选择不当，或者底层真实价值函数具有复杂、非线性的结构，那么线性模型将难以提供准确的估计，无论我们学习权重 (weight) $\mathbf{w}$ 的效果如何。瓦片编码、多项式基或傅里叶基等方法是构建特征的常用方式，这些特征可以帮助线性方法捕捉某些非线性。

### 局限性

线性方法的主要局限是其表示能力有限。它们只能表示特征和价值之间的线性关系。如果真实价值函数是高度非线性的，即使使用最佳的可能权重 (weight)，线性逼近 $\hat{v}(s, \mathbf{w})$ 或 $\hat{q}(s, a, \mathbf{w})$ 也可能与真实价值 $v_\pi(s)$ 或 $q_\pi(s, a)$ 明显不同。这一局限性促使我们使用非线性函数逼近器，例如神经网络 (neural network)，我们将在本章后面介绍。

此外，设计好的特征通常需要大量的专业知识和工程投入。

线性方法为函数逼近在强化学习 (reinforcement learning)中如何运作提供了基本理解。下一步是学习如何实际找到使我们的逼近尽可能准确的权重向量 (vector) $\mathbf{w}$，这通常涉及使用基于梯度的优化方法。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: The MIT Press
  强化学习领域的权威教科书。第9章和第10章对函数逼近，包括用于价值函数逼近的线性方法和常用特征构建技术，提供了基础性介绍。
- [Neuro-Dynamic Programming](https://www.athenasc.com) — Dimitri P. Bertsekas and John N. Tsitsiklis (1996)
  Publisher: Athena Scientific
  近似动态规划领域的开创性著作，提供了关于函数逼近，包括线性方法及其收敛性质的理论见解，对于理解强化学习算法的稳定性至关重要。
