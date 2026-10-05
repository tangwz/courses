# AdaGrad：调整每个参数的学习率

来源：[原文](https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/adagrad-optimizer)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随机梯度下降 (gradient descent)（SGD）及其变体（如Momentum）通常使用单一学习率来控制神经网络 (neural network)中所有参数 (parameter)的更新步骤。尽管学习率调度等方法可以随时间调整此全局速率，但它们没有考虑到不同参数可能需要不同更新幅度这一情况。设想一个稀疏特征的场景：一些与稀有特征相关的参数可能只偶尔接收到有用的梯度，而频繁特征的参数则持续接收更新。统一的学习率对于不频繁的参数可能过小，无法有效训练；对于频繁的参数则可能过大，导致不稳定。

AdaGrad（自适应梯度算法）优化器通过为每个参数维护一个根据该参数梯度历史进行调整的学习率来解决此问题。其思路简单明了：频繁遇到大梯度的参数，其学习率应被降低以避免超调；而与不频繁更新（以及历史上可能较小梯度）相关的参数，应保持较大的学习率以利于训练。

### AdaGrad机制

AdaGrad通过累积每个参数 (parameter)在所有先前时间步的梯度平方来实现这种调整。设$g_{t,i}$是目标函数在时间步$t$时对参数$\theta_i$的梯度。AdaGrad为每个参数$\theta_i$维护一个状态变量，我们称之为$G_{t,i}$，它是截至时间步$t$时对$\theta_i$的梯度平方和。

$G_{t,i}$的更新规则是：


$$
G_{t,i} = G_{t-1,i} + g_{t,i}^2
$$


通常，$G_{0,i}$被初始化为零。这个值$G_{t,i}$表示为参数$\theta_i$累积的历史梯度信息。

### 参数 (parameter)更新规则

更新实际参数$\theta_i$时，AdaGrad通过将全局学习率$\eta$除以累积和$G_{t,i}$的平方根来修改标准梯度下降 (gradient descent)更新。分母中会加上一个小的常数$\epsilon$（epsilon，例如$10^{-8}$）以保持数值稳定性，避免除以零，尤其是在$G_{t,i}$可能为零的初始步骤。

参数$\theta_i$在时间步$t$的更新是：


$$
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t,i} + \epsilon}} g_{t,i}
$$


我们来分解一下：

1. $g_{t,i}$：参数$\theta_i$的当前梯度。
2. $\eta$：一个全局学习率，类似于SGD中使用的学习率。
3. $G_{t,i}$：参数$\theta_i$截至当前步骤的梯度平方累积和。
4. $\epsilon$：一个小的平滑项，用于防止除以零。

请注意，每个参数$\theta_i$的有效学习率，即$\frac{\eta}{\sqrt{G_{t,i} + \epsilon}}$，是动态变化的。如果一个参数持续收到大梯度，$G_{t,i}$将快速增长，导致该参数的有效学习率下降。相反，如果一个参数收到小梯度或不频繁的梯度，$G_{t,i}$将缓慢增长，有效学习率将保持相对较高。

### 向量 (vector)化形式

实践中，这些操作在向量和矩阵上按元素进行。如果$\theta_t$表示时间步$t$时所有参数 (parameter)的向量，$g_t$是梯度向量，$G_t$是累积梯度元素平方的向量，则更新规则为：


$$
G_t = G_{t-1} + g_t \odot g_t
$$


$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot g_t
$$


其中$\odot$表示按元素乘法，平方根和$\epsilon$的加法也是按元素应用的。

### AdaGrad的优点

- **学习率自动调整：** AdaGrad有效地为每个参数 (parameter)单独调整学习率，减少了对全局学习率$\eta$进行大量手动设定的必要。
- **适用于稀疏数据：** 它在梯度稀疏的场景中表现良好，例如自然语言处理或推荐系统，其中某些特征（如单词或项目ID）出现不频繁。与频繁特征相比，这些不频繁特征会得到更大的更新。

### 注意事项

尽管有创新，AdaGrad有一个重要的属性需要记住：梯度平方的累积和$G_t$只随时间增长。由于$G_t$出现在分母中，每个参数 (parameter)的有效学习率在训练过程中会单调递减。在某些情况下，这种衰减可能变得过于剧烈，导致学习率收缩到接近零，从而过早地停止训练。这一局限性促使了后续自适应算法（如RMSprop和Adam）的发展，我们接下来会介绍它们。

## 参考资料

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 12; Pages: 2121-2159; DOI: [10.1162/jmlr.2011.12.2121](https://doi.org/10.1162/jmlr.2011.12.2121)
  介绍AdaGrad算法的原始研究论文，阐述其理论基础和应用。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 296-300 (Chapter 8.5.1)
  深度学习领域的经典教科书，全面解释了包括AdaGrad在内的优化算法及其在神经网络训练中的作用。
- [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/) — Justin Johnson, Andrej Karpathy, and Fei-Fei Li (2023)
  Journal: Stanford University CS231n Course Notes; Publisher: Stanford University
  提供了对包括AdaGrad在内的各种优化算法的易懂和实用性解释，常作为深度学习的教育资源。

---

[上一节](01-%E8%87%AA%E9%80%82%E5%BA%94%E5%AD%A6%E4%B9%A0%E7%8E%87%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md) · [下一节](03-AdaGrad%20%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7%EF%BC%9A%E5%AD%A6%E4%B9%A0%E7%8E%87%E8%A1%B0%E5%87%8F.md)
