---
course: "linear-algebra-essentials-ml"
chapter: "solving-linear-systems-matrix-inverses"
lesson: "linear-systems-ml-models"
sourceId: 1477
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-3-solving-linear-systems-matrix-inverses/linear-systems-ml-models"
title: "机器学习模型中的线性系统"
description: "了解求解线性系统对于训练线性回归等模型有多么核心。"
order: 1
plots: []
sourceHash: "1a1914e3517b35ab5b390c9dbea4a004404c604a6f60f09a5ee70d15e455aac0"
sourceCorrections: []
---

您已经了解向量 (vector)如何表示单个数据点或特征，以及矩阵如何组织整个数据集或表示线性变换。现在，我们将这些内容与机器学习 (machine learning)中的一个常见任务关联起来：为模型找到最优参数 (parameter)。通常，此任务归结为求解线性方程组。

考虑最基本的一个模型：线性回归。其目标是根据一组输入特征 $x_1, x_2, ..., x_n$ 预测目标值 $y$。该模型假设存在线性关系：


$$
y \approx \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$


这里，$\theta_0, \theta_1, \dots, \theta_n$ 是模型参数（或权重 (weight)、系数），我们需要根据训练数据来确定它们。如果我们有 $m$ 个数据点，可以将特征表示为矩阵 $X$（其中每行是一个数据点，通常会添加一列1以表示截距项 $\theta_0$），并将目标值表示为向量 $y$。参数构成一个向量 $\theta$。

线性回归的目标通常是使预测值与实际值之间的平方差之和最小化。微积分和线性代数表明，实现此最小化的最优参数向量 $\theta$ 满足*正规方程*：


$$
X^T X \theta = X^T y
$$


仔细看这个方程。

- $X^T X$ 得到一个方阵（我们称之为 $A$）。
- $\theta$ 是我们想找到的未知参数向量（我们称之为 $x$，与本章的符号一致）。
- $X^T y$ 得到一个向量（我们称之为 $b$）。

突然间，训练线性回归模型的问题转变为求解熟悉的矩阵方程：


$$
A x = b
$$


这正是本章关注的 $Ax=b$ 形式。找到线性回归的最优参数需要求解此方程组，以得到未知向量 $x$（它代表 $\theta$）。

这种模式不局限于简单的线性回归。

- **正则化 (regularization)回归：** 像岭回归这样的方法会略微修改方程以防止过拟合 (overfitting)，从而常得到诸如 $(X^T X + \lambda I) \theta = X^T y$ 的方程组，其中 $\lambda$ 是正则化参数，$I$ 是单位矩阵。这本质上仍然是一个 $Ax=b$ 问题，只是矩阵 $A$ 不同。
- **优化算法：** 即使对于更复杂的模型，其解不能通过单个矩阵方程找到，求解线性系统也经常作为训练过程中使用的迭代优化算法（如牛顿法）中的子问题出现。

因此，理解如何表示和求解像 $Ax=b$ 这样的线性方程组不仅仅是线性代数中的理论练习。它是实现和理解几种重要机器学习算法背后机制的实际要求。在阐明了求解这些系统在机器学习中为何相关之后，接下来的部分将讨论用于找到解向量 $x$ 的方法，从矩阵逆的思路开始。

## 参考资料

- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  涵盖机器学习的基础线性代数概念，包括线性回归和正规方程的推导。
- [An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2021)
  Publisher: Springer
  介绍线性回归、其推导以及如岭回归等正则化变体，从统计学习的角度进行说明。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  阐述适用于机器学习模型和优化的基本线性代数概念。
- [CS229 Lecture Notes: Supervised Learning](http://cs229.stanford.edu/notes2023fall/cs229-notes1.pdf) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University; Pages: 5-19
  提供机器学习背景下线性回归和正规方程的清晰推导与解释。
