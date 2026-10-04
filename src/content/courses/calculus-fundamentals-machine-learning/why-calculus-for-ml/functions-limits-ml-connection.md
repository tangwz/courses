---
course: "calculus-fundamentals-machine-learning"
chapter: "why-calculus-for-ml"
lesson: "functions-limits-ml-connection"
sourceId: 2246
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-1-why-calculus-for-ml/functions-limits-ml-connection"
title: "函数与极限在机器学习中的关联"
description: "简要说明函数如何代表模型，以及极限如何构成导数的依据。"
order: 6
plots: []
sourceHash: "d9d34a9f274cbbb7cb78e1742139277b9982a5100117202d95f965056c821a85"
sourceCorrections: []
---

您已了解函数是将输入映射到输出的方法，也了解了极限是趋近一个值的思想。这些与实际的机器学习 (machine learning)有何联系呢？事实证明，它们是基础构成部分。

### 函数即模型

从根本上说，许多机器学习 (machine learning)模型都可以看作是函数。考虑一个简单的任务：根据房屋面积（平方英尺）预测房价。

- **输入：** 房屋面积（例如，1500平方英尺）——我们称之为 $x$。
- **输出：** 预测价格（例如，\$300,000）——我们称之为 $y$。
- **模型：** 接收输入 $x$ 并产生输出 $y$ 的机制。

机器学习模型本质上是一个函数 $f$。它接收输入 $x$，然后给出输出 $y$，所以 $y = f(x)$。

例如，一个非常简单的线性模型可能会尝试使用熟悉的等式来捕捉这种关系：


$$
y = mx + b
$$


这里，$x$ 是房屋面积，$y$ 是预测价格，$m$ 和 $b$ 是模型需要从数据中学习的*参数 (parameter)*。代表这个模型的函数是 $f(x) = mx + b$。更复杂、多层级的模型，如神经网络 (neural network)，只是更精巧，但原理不变：它们将输入映射到输出。

### 极限作为学习的依据

所以，函数代表我们的模型。极限又在何处发挥作用呢？极限是构建*导数*的理论依据。我们尚未详细讨论导数（这将在下一章介绍！），但这里有其关联性：

导数衡量函数*瞬时变化率*。想想我们的房价模型。我们希望找到 $m$ 和 $b$ 的*最优*值，使模型的预测最准确。“准确性”通常通过衡量模型的*误差*来确定，使用一种被称为*成本函数*的东西（稍后会详细说明）。我们希望使这个成本或误差最小化。

我们如何使误差最小化？我们需要知道轻微改变 $m$ 或 $b$ 将如何影响误差。增加 $m$ 会使误差上升还是下降？会改变多少？导数恰好回答这些问题。它们告诉我们成本函数相对于参数 (parameter) $m$ 和 $b$ 的斜率或梯度。

导数是如何计算的呢？它们是通过极限定义的！具体来说，导数是通过观察函数输出的变化除以其输入的变化来确定的，当输入变化趋近于零时。这个“趋近于零”的部分正是极限的思想是关键所在。

> 一张图示，显示函数如何代表模型，极限如何构成导数的依据，以及导数如何引导用于训练模型的优化过程。

本质上：

1. **函数**为我们的机器学习 (machine learning)模型提供结构（将特征映射到预测）。
2. **极限**提供精确定义函数如何变化所需的数学依据。
3. 这种变化的定义（**导数**）使我们能够系统地调整模型参数，以提高性能（**优化**）。

不理解函数，我们就无法表示模型。不理解极限的思想，我们就无法掌握导数。如果没有导数，我们就无法有效地训练许多机器学习模型。本章为函数和极限奠定了铺垫；接下来，我们将在此基础上学习导数的主要思想。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher Bishop (2006)
  Publisher: Springer
  解释机器学习的数学基础，包括线性模型、损失函数和微积分在优化中的应用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供线性回归、成本函数和梯度下降的实用示例和解释，将其与底层数学函数和优化原理联系起来。
- [CS229: Machine Learning Lecture Notes - Main Notes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH5eSxfYbiBSfdt_E9K0TzZosdePPMYdZ7F3IpSUFnzuMODom28l0n-4KG-RKN1wf94LV6S1oJpn5y_VC5Ekl7c_6QvkcsT1ufQgDz-F0UBTLmwkMj_4WNWDZbzqTDZIDEMiE=) — Andrew Ng, Tengyu Ma (2023)
  Journal: Stanford University CS229 Course Material; Publisher: Stanford University
  涵盖线性回归的数学基础，包括作为函数的模型、成本函数以及梯度下降如何利用导数进行优化。
- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679904](https://doi.org/10.1017/9781108679904)
  全面概述机器学习所需的数学概念，其中包含关于微积分的专门章节，包括函数、极限和导数在优化背景下的应用。
