---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "need-adaptive-learning-rates"
sourceId: 4988
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/need-adaptive-learning-rates"
title: "自适应学习率的必要性"
description: "阐明自适应学习率方法的发展原因。"
order: 1
plots: ["plots/4988-0.json"]
sourceHash: "f8169f4a5808dfb00d5f95a8566fc1244dded12762c5f7b2fb21be782da5ae22"
sourceCorrections: []
---

随机梯度下降 (gradient descent) (SGD)、动量 (Momentum) 和 Nesterov 加速梯度 (NAG) 等优化器是训练深度学习 (deep learning)模型的常用方法。这些方法之间的一个共同点是使用单一的全局学习率，通常表示为 $\alpha$。此学习率根据计算出的梯度，决定了更新模型**所有**参数 (parameter)的步长。虽然学习率调度等技术（我们将在后面讨论）可以随时间调整此全局学习率，但在单次更新步中，它对每个参数保持不变。

然而，这种统一的方法在训练复杂的深度神经网络 (neural network)时可能低效甚至存在问题。为什么？因为“理想”的步长可能不适用于所有参数。我们来思考几个单一学习率表现不足的情况。

### 单一学习率遇到的问题

1. **特征重要性和频率差异**：想象在数据上训练一个模型，其中某些输入特征非常常见，而另一些则很少出现。与稀有特征相关的参数 (parameter)可能偶尔才能接收到有用的梯度。适合常见特征的小全局学习率，可能导致这些参数更新速度极慢，阻碍模型从那些稀有但可能重要的信号中进行学习的能力。反之，过大的学习率可能导致与常见特征相关的参数越过其最优值。
2. **层间梯度尺度差异**：深度网络不同层之间的梯度幅度通常差异很大。与前几层相比，后几层的梯度可能大得多（或者反之，取决于网络架构和激活函数 (activation function)）。一个对某一层有效的学习率，对另一层可能过大（导致发散）或过小（导致学习缓慢）。在所有地方应用相同的学习率会迫使我们做出折衷，这可能对网络的任何部分都不是最优的。
3. **复杂损失函数 (loss function)形态**：深度网络的损失函数很少形成简单的对称碗状。通常，它呈现出复杂的形状，带有狭长的山谷（峡谷）、平坦高原和鞍点。

思考一个细长损失曲面的简单可视化实例：



![非球形损失曲面示例](plots/4988-0.json)



> 此等高线图显示了一个损失函数，其中沿水平轴 (w1) 的进展远比沿陡峭垂直轴 (w2) 的进展容易。

在这种情况下，梯度沿 $w_2$ 方向比沿 $w_1$ 方向陡峭得多。如果我们使用全局学习率 $\alpha$:

- 如果 $\alpha$ 足够大，可以在浅 $w_1$ 方向上取得良好进展，它对于陡峭的 $w_2$ 方向可能过大，导致更新在山谷中剧烈来回振荡，可能导致发散。
- 如果 $\alpha$ 足够小以避免 $w_2$ 方向的振荡，沿 $w_1$ 方向的进展将非常缓慢。

带有动量或 NAG 的 SGD 可以帮助平滑振荡并加速沿浅方向的进展，但它们仍然使用单一学习率，限制了它们在高度非球形曲面上的有效性。

### 自适应方法

理想情况下，我们希望有一种优化算法，能够独立地为每个参数 (parameter)调整步长。它应该能够：

- 对于与平缓斜坡或不常见特征相关的参数，采取更大的步长。
- 对于与陡峭斜坡或非常常见特征相关的参数，采取更小的步长。

这正是**自适应学习率算法**的原理所在。AdaGrad、RMSprop 和 Adam 等方法维护每个参数的过往梯度信息，并使用这些历史数据单独调整学习率。它们有效地为每个参数提供了在训练期间动态变化的独立学习率。

通过逐参数调整学习率，这些算法通常能更有效地应对复杂的损失曲面，从而加快收敛速度，有时能找到更好的解，与标准 SGD 或其动量变体相比，特别是在默认超参数 (hyperparameter)设置下表现更优。

在接下来的章节中，我们将了解 AdaGrad、RMSprop 和 Adam 的具体机制，以明白它们如何达到这种自适应调整行为。

## 参考资料

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, and Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing; Volume: 12; Pages: 2121-2159; DOI: [10.5555/1953048.2078174](https://doi.org/10.5555/1953048.2078174)
  介绍了AdaGrad，这是一种基础的自适应学习率算法，通过将参数更新与过去梯度平方和的平方根成反比进行缩放。这篇论文有助于理解逐参数学习率缩放的起源。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Diederik P. Kingma, Jimmy Ba (2014)
  Journal: 3rd International Conference for Learning Representations, San Diego, 2015; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  介绍了Adam，这是一种广泛使用的自适应优化算法，它结合了RMSprop和Momentum的优点，通过估计梯度的一阶和二阶矩，为不同参数提供独立的自适应学习率。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 278-319
  第8章“深度模型训练的优化”全面概述了优化算法，包括自适应学习率的动机以及对AdaGrad、RMSprop和Adam的详细讨论。
