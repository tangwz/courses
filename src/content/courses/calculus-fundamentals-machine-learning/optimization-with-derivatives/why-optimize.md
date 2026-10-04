---
course: "calculus-fundamentals-machine-learning"
chapter: "optimization-with-derivatives"
lesson: "why-optimize"
sourceId: 2270
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-3-optimization-with-derivatives/why-optimize"
title: "优化：为什么最小化或最大化？"
description: "阐述优化在不同情境（包括机器学习）中的目标。"
order: 2
plots: []
sourceHash: "f4c506af151fdd347eab69a9af68069a6a082948d98376deb237e4eba170cfc4"
sourceCorrections: []
---

让我们思考一下，为什么找到这些最小值点或最大值点，也就是导数可能为零的“平坦点”，是如此重要。寻找函数最佳可能值（最小值或最大值）的过程被称为**优化**。

想象一下你在徒步旅行。你可能想找到山谷中的最低点（一个最小值）来安营扎寨，或者最高的山峰（一个最大值）以获得最佳视野。在日常生活中，你可能想最小化通勤时间或最大化储蓄。企业的目标是最小化生产成本和最大化利润。这些都是优化问题：在特定条件下找出最佳结果。

### 为什么要在机器学习 (machine learning)中进行优化？

在机器学习中，我们的目标通常是建立一个能进行预测的模型。设想一个简单的模型，它试图根据房屋面积来预测房价。该模型本质上是一个函数：你给它一个输入（房屋面积），它会产生一个输出（预测价格）。

但是，我们如何知道模型是否表现良好呢？我们需要一种方法来衡量它的表现，特别是其预测与实际房价相比的错误程度。这种“错误程度”或误差的衡量标准，我们通常称之为**成本函数**或**损失函数 (loss function)**。我们将在下一节中详细考察这些函数。

关联点如下：

1. **我们想要最佳模型：** “最佳”模型是指做出最准确预测的模型。
2. **准确性意味着低误差：** 准确的预测意味着模型预测值与实际值之间的差异很小。
3. **低误差意味着低成本：** 我们的成本函数旨在使误差小时其值小，误差大时其值大。

因此，训练许多机器学习模型的主要目标是**最小化成本函数**。通过寻找使成本函数值尽可能低的模型的设置（或参数 (parameter)），我们实际上是在寻找产生误差最小的模型，即在用于训练的数据上表现最好的模型。

虽然有时我们可能想最大化某些东西（比如给定模型的数据概率，即似然性），但最常见的情况，尤其是在初期阶段，涉及最小化误差。找出成本函数的那个最小值点是一项优化任务。

这正是导数发挥作用的地方。正如我们所见，导数帮助我们理解函数的斜率。通过查看导数（或者更准确地说，当我们有多个输入时的*梯度*，这一点我们稍后会提到），我们可以找出调整模型参数以降低成本的方向。找到导数为零的位置有助于我们确定该成本函数的潜在最小值点。

所以，我们进行优化是因为我们想要表现最佳的模型，在机器学习中，“最佳”通常意味着“最小误差”或“最小成本”。导数提供了系统性地寻找该最小值所需的数学工具。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  介绍了优化算法、损失函数以及导数/梯度在训练机器学习模型，特别是深度神经网络中的作用。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  为机器学习提供了全面基础，详细解释了损失函数、模型训练以及各种算法的优化数学原理。
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Stephen Boyd, Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press
  一本基础性教材，涵盖了优化理论与应用，解释了问题公式化以及函数中最小值和最大值的数学特性。
- [CS229 Lecture Notes: Introduction to Machine Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHd1Z_yLBwATdvreb7GmIvtPWQnsB1auEoPIIAH0yIHqaKrgk_CUUs6TLUZysmzMMT2Kjz9HeAuQp01wUzHFwaEja0t-tkcFNIaESRKNFuu-nI_SS44MR69a5eW4Aw_nBGEM5Q=) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  涵盖了机器学习的基本原理，包括成本函数的定义以及梯度下降在模型优化中的应用。
