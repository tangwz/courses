---
course: "calculus-fundamentals-machine-learning"
chapter: "calculus-action-optimization"
lesson: "gradient-descent-step-conceptual"
sourceId: 2313
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/gradient-descent-step-conceptual"
title: "执行梯度下降的单次更新"
description: "逐步讲解如何使用计算出的梯度进行梯度下降的单次更新。"
order: 5
plots: []
sourceHash: "875f25c39f891135d3deaf9e0849ec77456a5ea859c2fda8e6203364df8648ed"
sourceCorrections: []
---

一个简单的线性回归模型 $y = mx + b$，使用其成本函数 $J(m, b)$ 来衡量直线与数据的拟合程度。此成本函数的梯度被计算为一个包含偏导数 $\frac{\partial J}{\partial m}$ 和 $\frac{\partial J}{\partial b}$ 的向量 (vector)。这些导数表明了当我们微调 (fine-tuning) $m$ 或 $b$ 时成本如何变化。

那么，我们究竟如何运用这些梯度来改进模型参数 (parameter)并最小化成本呢？这就是梯度下降 (gradient descent)算法发挥作用的地方。

可以将成本函数 $J(m, b)$ 看作定义了一个曲面，也许像一个有山有谷的地形。我们的目标是找到这个地形中的最低点，即最小成本。参数 $m$ 和 $b$ 定义了我们在这个曲面上的当前位置。

梯度 $\nabla J = \left[ \frac{\partial J}{\partial m}, \frac{\partial J}{\partial b} \right]$ 在我们当前位置指向的是*最陡峭的上升*方向，即*上坡*最快的路径。由于我们想要最小化成本，所以需要*下坡*。因此，我们朝着与梯度*相反*的方向迈出一步。

这便引出了梯度下降的核心更新规则。对于每个参数，我们通过减去一个与其偏导数成比例的小量来调整其当前值：

- 更新斜率 $m$：
  $m_{\text{新}} = m_{\text{旧}} - \alpha \frac{\partial J}{\partial m}$
- 更新 y 轴截距 $b$：
  $b_{\text{新}} = b_{\text{旧}} - \alpha \frac{\partial J}{\partial b}$

这里，$m_{\text{旧}}$ 和 $b_{\text{旧}}$ 是更新步骤*之前*的参数值，而 $m_{\text{新}}$ 和 $b_{\text{新}}$ 是更新步骤*之后*的值。梯度 $\frac{\partial J}{\partial m}$ 和 $\frac{\partial J}{\partial b}$ 是使用*当前*值（$m_{\text{旧}}$，$b_{\text{旧}}$）计算的。

请注意更新规则中的符号 $\alpha$（阿尔法）。这是**学习率**，我们将在下一节中进行讨论。现在，可以将其视为控制我们下坡所迈步的*大小*。它是一个小的正数（例如，0.01，0.1）。

- 如果梯度分量（例如 $\frac{\partial J}{\partial m}$）为正，则表示增加 $m$ 会增加成本。因此，更新规则会减去一个正值（$\alpha \times \text{正梯度}$），从而有效地*减小* $m$。
- 如果梯度分量为负，则增加 $m$ 会*减小*成本。更新规则会减去一个负值（$\alpha \times \text{负梯度}$），这意味着我们*增加* $m$，使其朝着降低成本的方向移动。

对 $m$ 和 $b$ 的这种单一计算和更新构成了梯度下降的*一步*。

> 该图显示了梯度下降中单次更新的流程。从当前模型参数开始，使用这些参数计算成本函数的梯度，然后通过沿梯度相反的方向微小移动来更新参数。此过程通常会重复多次。

理解梯度下降是一种迭代算法很重要。通常，一步不足以达到最小成本。我们多次重复这个过程，计算梯度并更新参数。每一步，我们（希望）都会更接近最小化成本函数所需的 $m$ 和 $b$ 值，从而得到一个拟合度更好的线性回归模型。

这一步，在计算出的导数引导下，是许多机器学习 (machine learning)模型从数据中学习的基本机制。通过重复调整参数以减少误差（成本）的方向，模型逐步提高其预测能力。

## 参考资料

- [CS229 Lecture Notes: Supervised Learning, Discriminative Algorithms: Linear Regression](https://github.com/mxc19912008/Andrew-Ng-Machine-Learning-Notes/blob/master/cs229-notes1.pdf) — Andrew Ng (2019)
  Publisher: Stanford University
  涵盖线性回归和梯度下降算法的基础知识，包括其迭代更新规则，正如一门广受认可的机器学习课程中所教。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  为优化算法提供了全面的理论解释，包括梯度下降的推导和运行原理，适用于多种机器学习模型。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 3; DOI: [10.1007/b94608](https://doi.org/10.1007/b94608)
  统计学习方法的标准参考书，提供了关于线性回归和用于拟合此类模型的优化技术的全面统计学视角。
