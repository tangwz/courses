---
course: "deep-learning-regularization-optimization"
chapter: "adaptive-optimizers"
lesson: "adagrad-limitations"
sourceId: 4990
sourceUrl: "https://apxml.com/zh/courses/deep-learning-regularization-optimization/chapter-6-adaptive-optimizers/adagrad-limitations"
title: "AdaGrad 的局限性：学习率衰减"
description: "讨论 AdaGrad 中学习率激进衰减的问题。"
order: 3
plots: ["plots/4990-0.json"]
sourceHash: "5630eb5a01307a550f1fb417c5f1cf85bad910509bf8a3a1857bc8045ef42b38"
sourceCorrections: []
---

虽然 AdaGrad 引入了有用的每参数 (parameter)自适应学习率思想，但其累积梯度历史的特定机制导致了一个显著的实际问题。AdaGrad 的更新规则将参数 $\theta_i$ 在时间步 $t$ 的有效学习率通过该参数历史梯度平方和的平方根的倒数进行缩放：


$$
\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}} g_{t, i}
$$


在此，$g_{t, i}$ 是损失函数 (loss function)在时间步 $t$ 对参数 $\theta_i$ 的梯度，$\eta$ 是全局学习率，$\epsilon$ 是一个用于数值稳定的小常数，而 $G_{t, ii}$ 是从时间步 1 到 $t$ 对 $\theta_i$ 的梯度平方和：


$$
G_{t, ii} = \sum_{\tau=1}^{t} g_{\tau, i}^2
$$


核心局限性源于累积项 $G_{t, ii}$。因为 $g_{\tau, i}^2$ 总是非负的（作为一个平方值），所以假设梯度非零，在整个训练过程中，和 $G_{t, ii}$ 将单调递增。它从不减小。

随着训练的进行，$G_{t, ii}$ 不断增大。因此，分母项 $\sqrt{G_{t, ii} + \epsilon}$ 也随之增大。这导致有效学习率 $\frac{\eta}{\sqrt{G_{t, ii} + \epsilon}}$ 单调趋近于零。



![AdaGrad：有效学习率衰减](plots/4990-0.json)



> 示例说明了使用 AdaGrad 时，两个不同参数的有效学习率在训练迭代过程中如何降低。经历较大梯度的参数（参数 1）比梯度较小的参数（参数 2）衰减快得多，但两者都趋近于零。初始全局学习率 $\eta=0.01$。

这种过于激进的衰减会带来问题。在深度学习 (deep learning)中，优化空间复杂且非凸，训练即使在后期也常需要进行探究。如果学习率过早变得小到可以忽略不计，优化器可能实际上在达到满意的最小值之前很久就停止进展。模型的学习能力可能过早地被中止。

虽然 AdaGrad 是朝着自适应学习率迈出的重要一步，但这一局限性促使了算法的出现，这些算法能够在不导致学习率如此激进衰减的情况下调整学习率。像 RMSprop 这样的方法，我们接下来会查看，它们修改了累积机制以阻止这种无限制的增长。

## 参考资料

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) — John Duchi, Elad Hazan, Yoram Singer (2011)
  Journal: Journal of Machine Learning Research; Volume: 12; Pages: 2121-2159
  这篇基础论文介绍了 AdaGrad 算法，详细阐述了其基于过去梯度平方累加的自适应学习率机制。
- [Lecture 6e: RmsProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — Tijmen Tieleman, Geoffrey Hinton (2012)
  该内容在 Coursera 课程《神经网络机器学习》中展示，介绍了 RMSprop，这是一种旨在解决 AdaGrad 学习率过快衰减问题的优化器。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教材全面讨论了深度学习中的优化算法，包括对 AdaGrad 特性和其局限性的分析。
