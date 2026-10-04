---
course: "calculus-essentials-machine-learning"
chapter: "calculus-ml-introduction"
lesson: "optimization-concept-ml"
sourceId: 1346
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-1-calculus-ml-introduction/optimization-concept-ml"
title: "机器学习中的优化思想"
description: "弄清优化的主要内容：为机器学习模型确定适宜参数。"
order: 2
plots: ["plots/1346-0.json"]
sourceHash: "3bda96c28f6a70bd74ed0a32f03c083e0344549f6f0c022d6778064a9ddc66b9"
sourceCorrections: []
---

我们来思考一下，训练一个机器学习 (machine learning)模型究竟是什么意思。机器学习模型通常可以表示为数学函数。比如，一个简单的线性回归模型尝试通过函数 $f(x) = mx + b$ 根据输入 $x$ 预估输出 $y$。但是，我们如何从训练数据中找出参数 (parameter) $m$（斜率）和 $b$（截距）的*最恰当*值呢？这就是优化发挥作用之处。

在机器学习中，优化是指为模型寻找一组参数，使其在训练数据上达到尽可能好的表现。但是，“好”是什么意思？我们需要一种方法来评估模型运行的优劣（或者更普遍地说，劣）。这个评估指标通常被称为**成本函数**或**损失函数 (loss function)**。

成本函数，常用 $J$ 表示，它接收模型的预测值和训练数据中的真实目标值，然后计算出一个数值，用来衡量总体误差或“成本”。例如，回归问题的一个常用成本函数是均方误差（MSE），它算出预测值与实际值之间平方差的平均数。

如果我们的模型参数由向量 (vector) $\theta$ (theta) 表示，那么成本函数 $J$ 会依据这些参数：$J(\theta)$。那么，优化的目的是确定特定的参数集合 $\theta^*$，使成本函数最小化：


$$
\theta^* = \arg \min_{\theta} J(\theta)
$$


这个公式表达为：“$\theta^*$ 是使成本函数 $J(\theta)$ 最小化的自变量（参数取值）$\theta$。”

可以把成本函数想象成一个曲面，曲面上的位置由模型参数的取值决定，而曲面的高度则代表成本。我们的目的是找到这个曲面上的最低点。

设想一个非常简单的情形，我们只有一个参数，称之为 $w$。成本函数 $J(w)$ 可能看起来像一条曲线。优化就是找出这条曲线底部的 $w$ 值。



![简单成本函数与参数的关系](plots/1346-0.json)



> 当参数 $w$ 等于0时，成本 $J(w)$ 达到最小值。优化旨在确定这个最低点。

对于拥有大量参数的模型（深度学习 (deep learning)中可达百万），这个“曲面”存在于高维空间 (high-dimensional space)，使得可视化变得不可能。我们不能只看图就找出最小值。我们需要一种系统化的算法方法来遍历这个复杂的曲面，并找到最低点，或者至少是一个很低的点。

这正是微积分变得不可或缺之处。正如我们在前一节中简略地提到，导数衡量函数的变化率或斜率。通过计算成本函数相对于模型参数的导数（或其多变量对应物——梯度，我们稍后会介绍），我们可以判明成本增加最快的方向。为了使成本最小化，我们只需朝*相反*方向移动即可。这种迭代计算方向并迈进的过程，是许多机器学习优化算法的根本，其中最著名的就是梯度下降 (gradient descent)。

本质上，优化衔接了拥有模型结构和拥有一个能在数据上实际运作良好的模型之间的距离。它是推动学习过程的引擎，而微积分提供了构建并理解这个引擎的基本工具。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖机器学习中使用的基本优化概念和算法，包括深度学习背景下的成本函数和梯度下降。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-31073-2](https://doi.org/10.1007/978-0-387-31073-2)
  介绍模式识别和机器学习的基础概念，详细阐述误差函数和优化在模型参数估计中的作用。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  对统计学习方法提供了易于理解的介绍，通过线性回归和残差平方和最小化等示例解释优化。（第1版）
- [CS229 Lecture Notes: Supervised Learning](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2018)
  Publisher: Stanford University
  专门涵盖线性回归、成本函数（如MSE），以及作为优化方法的梯度下降的介绍。
