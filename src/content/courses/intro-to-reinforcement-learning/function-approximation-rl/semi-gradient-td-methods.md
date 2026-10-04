---
course: "intro-to-reinforcement-learning"
chapter: "function-approximation-rl"
lesson: "semi-gradient-td-methods"
sourceId: 1655
sourceUrl: "https://apxml.com/zh/courses/intro-to-reinforcement-learning/chapter-6-function-approximation-rl/semi-gradient-td-methods"
title: "半梯度 TD 方法"
description: "说明为什么带有函数逼近的 TD 方法使用半梯度更新。"
order: 6
plots: []
sourceHash: "f399ffb8107d414234c24140e03a8a8196181fb8a93099c4bdf6729e60ac5f22"
sourceCorrections: []
---

梯度下降 (gradient descent)是一种用于学习函数逼近器 $\hat{v}(s, \theta)$ 的参数 (parameter) $\theta$ 的方法。其目的是最小化预测值 $\hat{v}(S_t, \theta)$ 与某个目标值 $U_t$ 之间的平方误差。对于蒙特卡洛方法，这个目标 $U_t$ 是该幕的实际回报 $G_t$，它不依赖于当前的价值估计。这使得我们能够执行真正的梯度下降。

现在，我们来考虑时序差分 (TD) 学习。回顾第5章，TD 方法根据观测到的奖励 $R_{t+1}$ 和*下一*状态 $S_{t+1}$ 的估计价值来更新状态 $S_t$ 的价值估计。时间步 $t$ 更新的 TD 目标是：


$$
Y_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t)
$$


这里，$\hat{v}(S_{t+1}, \theta_t)$ 是使用当前参数 $\theta_t$ 对下一状态价值的*当前估计值*。这便是与函数逼近结合时产生有趣之处。

如果我们尝试应用之前相同的梯度下降方法，旨在最小化预测值 $\hat{v}(S_t, \theta)$ 与 TD 目标 $Y_t$ 之间的均方误差 (MSE)，我们会遇到一个细微的问题。单次转换的损失函数 (loss function)如下所示：


$$
L(\theta) = \frac{1}{2} [ Y_t - \hat{v}(S_t, \theta) ]^2 = \frac{1}{2} [ R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta) - \hat{v}(S_t, \theta) ]^2
$$


注意到目标 $Y_t$ 本身依赖于参数 $\theta$，因为它包含项 $\hat{v}(S_{t+1}, \theta)$。一个真正的梯度下降更新需要对*整个*表达式关于 $\theta$ 求梯度。这涉及计算目标值 $\hat{v}(S_{t+1}, \theta)$ 的梯度，这可能复杂且计算成本高。

更重要地是，目标 $Y_t$ 基于一个本质上嘈杂且有偏的*估计值*（因为它依赖于当前可能不准确的权重 (weight) $\theta$）。基于这个可能存在缺陷的目标的梯度来更新我们的参数，可能导致不稳定或收敛缓慢。

### 半梯度方法

为了解决这个问题，带有函数逼近的 TD 方法通常采用所谓的**半梯度**方法。主要思想简单但有效：在计算更新的梯度时，我们假定 TD 目标 $Y_t$ 是一个固定的观测值，就像蒙特卡洛方法中的回报 $G_t$ 一样。我们忽略了 $Y_t$ 依赖于当前参数 (parameter) $\theta_t$ 的事实。

本质上，我们仅计算关于我们的预测 $\hat{v}(S_t, \theta)$ 的梯度，而不是目标部分 $R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t)$。

简化损失（将 $Y_t$ 视为常数）的梯度是：


$$
\nabla_{\theta} L(\theta) \approx \nabla_{\theta} \frac{1}{2} [ (R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t)) - \hat{v}(S_t, \theta) ]^2
$$


$$
\nabla_{\theta} L(\theta) \approx - [ R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t) - \hat{v}(S_t, \theta) ] \nabla_{\theta} \hat{v}(S_t, \theta)
$$


记住梯度下降 (gradient descent)以梯度的*相反*方向更新参数。因此，权重 (weight) $\theta$ 的更新规则变为：


$$
\theta_{t+1} \leftarrow \theta_t - \alpha \nabla_{\theta} L(\theta)
$$


$$
\theta_{t+1} \leftarrow \theta_t + \alpha [ R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t) - \hat{v}(S_t, \theta_t) ] \nabla_{\theta} \hat{v}(S_t, \theta_t)
$$


让我们分析一下：

1. **TD 误差 ($\delta_t$)**：方括号中的项 $R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t) - \hat{v}(S_t, \theta_t)$，是我们熟悉的在表格型 TD(0) 中遇到的 TD 误差。它表示 $S_t$ 的估计价值与从即时奖励和下一状态价值得出的更好估计之间的差异。
2. **价值函数的梯度 ($\nabla_{\theta} \hat{v}(S_t, \theta_t)$)**：该项告诉我们改变 $\theta$ 中的每个参数将如何影响*当前*状态 $S_t$ 的价值估计。它指导更新方向，使其趋向于对当前预测影响最大的参数。
3. **学习率 ($\alpha$)**：控制更新的步长。

这被称为“半梯度”方法，因为我们只使用了真正的梯度的一部分。我们计算了我们的预测 $\hat{v}(S_t, \theta)$ 的梯度，但忽略了目标 $\hat{v}(S_{t+1}, \theta)$ 的梯度。

### 线性函数逼近的半梯度 TD(0)

让我们以线性函数逼近为例来具体化，其中我们的价值估计是 $\hat{v}(s, \theta) = \theta^T x(s)$，而 $x(s)$ 是状态 $s$ 的特征向量 (vector)。

如我们之前所见，线性价值函数关于参数 (parameter) $\theta$ 的梯度就是特征向量本身：


$$
\nabla_{\theta} \hat{v}(s, \theta) = x(s)
$$


将其代入通用半梯度 TD 更新规则，我们得到了**线性半梯度 TD(0)** 的更新规则：


$$
\theta_{t+1} \leftarrow \theta_t + \alpha [ R_{t+1} + \gamma \theta_t^T x(S_{t+1}) - \theta_t^T x(S_t) ] x(S_t)
$$


这个更新规则计算高效，并且在实践中通常表现良好。在观察到一次转换 $(S_t, A_t, R_{t+1}, S_{t+1})$ 后，我们：

1. 获取当前状态 $x(S_t)$ 和下一状态 $x(S_{t+1})$ 的特征向量。
2. 计算当前价值估计：$\hat{v}(S_t, \theta_t) = \theta_t^T x(S_t)$ 和 $\hat{v}(S_{t+1}, \theta_t) = \theta_t^T x(S_{t+1})$。
3. 计算 TD 误差：$\delta_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \theta_t) - \hat{v}(S_t, \theta_t)$。
4. 更新权重 (weight)向量：$\theta_{t+1} \leftarrow \theta_t + \alpha \delta_t x(S_t)$。

下图展示了半梯度 TD 更新一个步骤中的信息流。

> 带有函数逼近的半梯度 TD(0) 中单次更新步骤的流程图。

### 为何称“半”？收敛性考量

尽管半梯度方法被广泛使用且通常有效，但重要的是要理解它们并非针对贝尔曼误差的真正的梯度下降 (gradient descent)方法。由于我们忽略了目标中的梯度依赖性，因此失去了与在固定目标函数上进行标准梯度下降相关的理论收敛保证。

在某些情况下，尤其是在使用非线性函数逼近器或离策略学习（例如尝试在遵循不同行为策略 $b$ 的同时学习目标策略 $\pi$ 的 Q 值）时，半梯度方法可能变得不稳定，参数 (parameter)也可能发散。这通常被称为“致命三元组”：函数逼近、自举（TD 更新）和离策略学习。

然而，对于带有线性函数逼近的在策略 TD 学习，半梯度方法通常稳定且可靠收敛，不一定收敛到绝对最好的可能权重 (weight)，而是在所选特征和函数逼近器的限制内，收敛到一个接近最优解的合理近似值。收敛通常到一个最小化*投影*贝尔曼误差的固定点，对此的详细讨论超出了我们目前的范围，但这表明了一个明确的目标。

半梯度方法提供了一种实用且计算上可行的方式，将 TD 学习的能力与大规模问题中函数逼近的必要性结合起来。它们构成了许多先进算法的基础，包括我们稍后会讲到的深度 Q 网络。接下来，我们将简要考虑使用更强大的非线性函数逼近器，如神经网络 (neural network)。

## 参考资料

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton and Andrew G. Barto (2018)
  Publisher: MIT Press; Pages: Chapters 9-10
  这本教材是强化学习领域的权威资料。第9章和第10章深入解释了函数逼近、半梯度TD方法及其收敛特性。
- [Gradient Descent for Reinforcement Learning](https://neurips.cc/paper/1994/file/f7e6a71e1a5342a6c8e03b44dd92d53c-Paper.pdf) — Leemon C. Baird (1994)
  Journal: Advances in Neural Information Processing Systems 7; Publisher: MIT Press; Volume: 7; Pages: 1007-1014; DOI: [10.5555/2984950.2985068](https://doi.org/10.5555/2984950.2985068)
  这篇基础性论文强调了“致命三元组”（函数逼近、自举和离策略学习），以及使用函数逼近的TD学习中可能出现发散的问题，解释了半梯度方法所应对的挑战。
- [CS234: Reinforcement Learning (Winter 2024)](http://cs234.stanford.edu/) — Emma Brunskill (2025)
  Publisher: Stanford University
  斯坦福大学的强化学习课程提供了关于半梯度TD方法和函数逼近的讲义和视频，提供了另一种教学视角。
