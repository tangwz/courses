---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "regularization-overview"
sourceId: 5063
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/regularization-overview"
title: "正则化方法概述"
description: "介绍正则化作为一种应对过拟合的方法。"
order: 2
plots: ["plots/5063-0.json"]
sourceHash: "f67d41110caa8c66f4cddda6a4db928a2f627dea2716bad8258497158c992eba"
sourceCorrections: []
---

正如我们在本章引言中谈到的，过拟合 (overfitting)是深度学习 (deep learning)模型开发中的一个主要挑战。当模型在其训练数据上表现极好，但无法很好地泛化到新的、未见过的数据时，它很可能已经过拟合了。出现这种情况是因为模型不仅学习了数据固有的模式，还学习了只存在于训练集中的噪声和特定的特征。



![过拟合：训练损失与验证损失](plots/5063-0.json)



> 一个表示过拟合的典型模式是：训练损失持续下降，而验证损失在某个点后开始上升。

为了解决这个问题，我们采用**正则化 (regularization)方法**。正则化包含了在训练过程中使用到的多种方法，它们专门用于减少过拟合并提升模型在新数据上的泛化性能。大多数正则化策略的主要思想是限制模型的复杂度，防止它过度适应训练数据。

可以从偏差-方差权衡的角度来考虑。一个过拟合的模型通常具有低偏差（它非常贴合训练数据）但高方差（它对训练数据中的微小波动敏感，导致泛化能力差）。正则化方法旨在降低这种方差，通常会略微增加偏差，以在新数据上获得更好的整体模型性能。它们通过添加约束或惩罚，有效地抑制了过度复杂的解决方案。

本章中，我们将考察几种广泛使用的正则化方法：

1. **参数 (parameter)范数惩罚（L1和L2正则化）：** 这些方法根据网络权重 (weight)的量级向模型的损失函数 (loss function)添加一个惩罚项。L1促进稀疏性（一些权重变为零），而L2则鼓励整体权重较小。这可以防止单个权重变得过大并影响预测。
2. **Dropout：** 这种技术在每次训练更新时，随机地将一部分神经元激活值设为零。它迫使网络学习到更具适应性的表示，不至于过度依赖任何一个神经元，从而有效地模拟训练一个小型网络的集合。
3. **早停法：** 这是一种简单而有效的方法，我们在训练期间监测模型在单独的验证数据集上的表现，当验证表现开始下降时就停止训练过程，即使训练损失仍在减少。
4. **批量归一化 (normalization)：** 批量归一化主要用于通过对小批量内层输入进行归一化来稳定和加速训练，但它也可以产生轻微的正则化作用，在某些情况下减少对Dropout等其他正则化方法的需求。

L2正则化、Dropout和早停法是常用的正则化策略，它们并非相互排斥；通常，将这些策略结合使用可以产生最好的效果。了解如何以及何时应用这些策略是构建在实践中表现良好的可靠深度学习模型的必要技能。对这些策略将进行详细的阐述和实际操作指导。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书全面介绍了深度学习基础知识，包含对过拟合、偏差-方差权衡以及L1/L2、Dropout和提前停止等多种正则化技术的详细讨论。
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 15; Pages: 1929-1958; DOI: [10.5555/2627435.2670313](https://doi.org/10.5555/2627435.2670313)
  介绍Dropout技术的原始文献，该技术对缓解神经网络过拟合至关重要。
- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) — Sergey Ioffe and Christian Szegedy (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Pages: 448-456; DOI: [10.48550/arXiv.1502.03167](https://doi.org/10.48550/arXiv.1502.03167)
  介绍批标准化（Batch Normalization）的论文，阐述了其在稳定和加速深度网络训练中的作用，以及其正则化效果。
- [Regularization](http://cs231n.github.io/neural-networks-2/#reg) — Stanford CS231n Course Staff (2023)
  Publisher: Stanford University
  来自广受认可的大学课程的官方笔记，提供了神经网络正则化方法的清晰解释。
