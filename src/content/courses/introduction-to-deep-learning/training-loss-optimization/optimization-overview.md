---
course: "introduction-to-deep-learning"
chapter: "training-loss-optimization"
lesson: "optimization-overview"
sourceId: 5037
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/optimization-overview"
title: "优化：寻找最优权重"
description: "介绍优化如何通过调整权重来最小化损失函数。"
order: 4
plots: ["plots/5037-0.json"]
sourceHash: "af4022d80d49ff8a55835eab7f6b535251977443c8ae12716f80f96775c7d330"
sourceCorrections: []
---

你已经了解了损失函数 (loss function)如何衡量神经网络 (neural network)的预测与实际目标之间的偏差程度。高损失表示表现不佳；低损失则表示表现更好。但我们究竟如何**借助**这个分数来改进网络呢？目的不只是衡量误差，而是要使其最小化。调整网络参数 (parameter)（其权重 (weight)和偏差）以减少损失的这一过程，称为**优化**。

可以将损失函数看作定义了一个曲面，通常称为损失曲面。对于一个只有两个权重的简单模型，你可以将其想象成一片起伏的地形。这片地形上任意一点的高度，代表了特定权重组合下的损失值。我们的目标是找到这片地形上的最低点，也就是对应于最小可能损失的点。

我们如何在这片地形中行进呢？我们从一个随机点（对应于网络的初始随机权重）开始。我们需要一种方法来确定从当前位置哪个方向是“下坡”。这就是微积分发挥作用的地方，具体来说就是**梯度**。

损失函数相对于网络参数（所有权重和偏差）的梯度，告诉我们损失**增加**最快的方向。它是一个指向“上坡”的向量 (vector)。如果我们想**减少**损失，就应该沿着梯度完全相反的方向移动。

想象你正站在那个山坡上。梯度会告诉你哪个方向是上坡最陡峭的。为了最快到达谷底，你会直接往下坡迈一步，这正好与梯度的方向相反。

这种反复计算梯度并沿相反方向迈步的迭代过程，是**梯度下降 (gradient descent)**背后的核心思想，它是深度学习 (deep learning)中最基本的优化算法。我们根据梯度的指引，反复调整权重和偏差，目标是沿着损失曲面下降到最小值。



![损失最小化](plots/5037-0.json)



> 单个参数损失曲线的简化视图。优化旨在通过沿下坡方向（与梯度相反）迈步，从起点向最小损失移动。

本质上，优化是推动神经网络学习的引擎。通过反复计算损失随每个参数（即梯度）如何变化，并沿着减少损失的方向更新这些参数，网络逐步提高其做出准确预测的能力。接下来的部分将详细阐述梯度下降算法的运行方式、学习率的重要作用，以及像随机梯度下降这类使训练大型网络成为可能的实用变体。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书为深度学习中使用的优化方法（包括梯度下降）提供了全面的理论背景。
- [Neural Networks Part 3: Learning and Optimization](http://cs231n.github.io/optimization-1/) — Stanford University CS231n Course Staff (2023)
  这些被广泛引用的讲义提供了优化技术（包括梯度下降）在训练神经网络中的实用和直观的解释。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html) — Michael Nielsen (2019)
  Pages: Chapter 2
  这本在线书籍提供了神经网络基础知识的易懂介绍，并清楚解释了梯度下降及其在学习中的作用。
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 10
  这本实用指南展示了梯度下降等优化算法在流行深度学习框架中的应用，将理论与实现联系起来。
