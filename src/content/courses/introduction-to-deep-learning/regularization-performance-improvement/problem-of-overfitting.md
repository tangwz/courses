---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "problem-of-overfitting"
sourceId: 5062
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/problem-of-overfitting"
title: "过拟合问题"
description: "了解什么是过拟合，如何发现它（通过验证曲线），以及它发生的原因。"
order: 1
plots: ["plots/5062-0.json"]
sourceHash: "0eeba17deb2f78c07087d9bcedf454ac8a16c104afa5e0b0ec30be06c8bee570"
sourceCorrections: []
---

当你训练神经网络 (neural network)时，主要目标是使训练数据集上的损失函数 (loss function)最小。我们讨论过的优化算法，比如梯度下降 (gradient descent)及其变体，正是为此目的而设。然而，仅仅在模型训练过的数据上达到很低的损失，并不能确保在实际使用中表现良好。这便引出了机器学习 (machine learning)模型开发中的一个常见难题：过拟合 (overfitting)。

## 什么是过拟合 (overfitting)？

过拟合发生在模型对训练数据学得*过*好时。模型没有掌握数据中能泛化到新样本的基本规律和关联，而是开始记忆只存在于训练集中的具体细节和噪声。想象一下，你通过背诵练习题的准确答案来备考。你可能在练习测试中取得好成绩，但遇到考查相同原理的新问题时，你可能会遇到困难，因为你没有学习基本原理。过拟合模型的表现类似；它在已见过的数据上表现非常出色，但无法泛化到未见过的数据。

这导致模型出现低*偏差*（它紧密贴合训练数据）但高*方差*（它的预测结果随不同训练集剧烈变化，在新数据上表现不佳）的情况。相反的问题是欠拟合 (underfitting)，它发生在模型过于简单，无法掌握数据的基本结构时，导致在训练集和验证集上表现均不理想（高偏差）。我们的目标通常是在这两种极端情况之间找到一个最佳点。

## 过拟合 (overfitting)为何发生？

有几个因素可能导致过拟合：

1. **模型复杂度：** 深度神经网络 (neural network)，由于其可能拥有大量的层和神经元，具备学习复杂模式的强大能力。如果模型的容量远超所处理任务的复杂程度所需，它就很容易开始拟合训练数据中的噪声，而不是真实信号。一个非常深或宽的网络可能会找到复杂的决策边界，这些边界完美地分隔训练样本，但对微小变化过于敏感。
2. **训练数据不足：** 如果训练数据量有限或无法代表真实数据分布，模型可能会错误地抓住虚假关联或仅存在于小样本中的特定例子。有了更多样的数据，模型被引导学习更具普适性的模式，这些模式在不同样本中都成立。
3. **训练时间过长：** 对模型进行过多轮次的训练也可能导致过拟合。起初，模型学习的是普适性模式，并且训练和验证表现都会提升。然而，在某个点之后，模型可能开始对训练集中的噪声和特定特性进行微调 (fine-tuning)，导致验证表现下降，即便训练表现持续提升。

## 发现过拟合 (overfitting)

最常见的方式是通过监测模型在训练过程中，在训练集和独立的验证集上的表现来发现过拟合。

- **训练表现：** 这通常通过模型训练数据上的损失函数 (loss function)和相关指标（如分类任务的准确率）来衡量。
- **验证表现：** 这使用相同的损失和指标来衡量，但在模型在梯度更新过程中未见过的独立数据集（验证集）上计算。此数据集作为未见过数据的代表。

如果模型学习良好且具有泛化能力，训练损失和验证损失都应该下降，相关指标也应该提升。然而，如果开始出现过拟合，你通常会观察到：

- 训练损失持续下降（或保持非常低）。
- 验证损失在达到一个最低点后开始上升。
- 训练指标（例如，准确率）持续提升或保持高位。
- 验证指标（例如，准确率）趋于平稳或开始下降。

训练和验证表现之间的这种差异是过拟合的一个明确迹象。可视化训练和验证损失/指标随轮次的变化是一个标准做法。



![训练损失与验证损失对比](plots/5062-0.json)



> 典型的学习曲线显示训练损失持续下降，而验证损失在大约第25轮次后开始上升，表示过拟合的出现。

构建机器学习 (machine learning)模型的最终目标是**泛化能力**：在新数据上做出准确预测的能力。过拟合是获得好的泛化能力的一个直接障碍。因此，理解、发现并缓解过拟合是任何深度学习 (deep learning)实践者的重要技能。本章后续部分将介绍几种专门用于应对过拟合并提升模型泛化表现的方法。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 5, Sections 5.1-5.3
  一本全面教科书，在深度神经网络背景下，涵盖了过拟合、泛化和偏差-方差权衡等基本机器学习概念。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 7
  统计学习领域的经典参考文献，提供了模型评估、选择和偏差-方差问题的严谨理论背景，这对于理解过拟合至关重要。
- [Machine Learning](https://www.coursera.org/learn/machine-learning) — Andrew Ng (2016)
  Journal: Coursera; Publisher: DeepLearning.AI and Stanford Online
  一门广受认可的在线入门课程，清晰解释了过拟合、欠拟合的概念以及使用训练集和验证集检测它们的实际方法。
