---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "early-stopping"
sourceId: 5066
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/early-stopping"
title: "早期停止"
description: "使用验证集表现，在模型开始过拟合前停止训练。"
order: 5
plots: ["plots/5066-0.json"]
sourceHash: "8b42d0b6bcfa99d91498d019a648f165a34b3df52cd9e0746dd8ad20fa0472a5"
sourceCorrections: []
---

模型训练时间过长可能导致过拟合 (overfitting)，即模型对训练数据中的具体模式和噪声学习得过于彻底，从而使其在新数据上的表现变差。虽然L1/L2正则化 (regularization)和Dropout等方法通过改变网络或损失函数 (loss function)来应对此问题，**早期停止**则提供了一种更直接、按步骤进行的方式。它是实践中使用最简单但非常有效的正则化方法之一。

核心思想很简单：在训练期间监控模型在单独的验证数据集上的表现，当验证集上的表现不再提升或开始变差时，即使训练集上的表现仍在改善，也停止训练过程。

### 早期停止的工作方式

1. **划分数据：** 在训练前，你需要将数据至少分成三个部分：训练集、验证集和测试集。模型从训练集学习。验证集用于在训练期间监控表现，以便做出早期停止等决策。测试集提供所选模型的最终无偏评估。
2. **监控验证表现：** 随着模型逐个迭代地训练，在每个迭代后（或有时在固定步数后），在验证集上计算一个选定的表现指标（最常见的是验证损失，但也可以是验证准确率或其他相关指标）。
3. **记录最佳表现：** 持续记录迄今为止达到的最佳验证指标分数以及与该迭代相关的模型权重 (weight)。
4. **停止训练：** 如果验证指标在预定数量的连续迭代（通常称为“耐心期”）内没有改善，则停止训练过程。
5. **使用最佳模型：** 用于推断的最终模型通常是保存自获得最佳验证分数那个迭代的模型，而不是训练的最后一个迭代的模型。

### 早期停止的图形呈现

训练损失、验证损失与理想停止点之间的关系可以以图形方式呈现。通常，训练损失会随着迭代次数的增加而持续下降。验证损失最初也会下降，但随后随着模型开始过拟合 (overfitting)而开始上升。早期停止的目标是在验证损失曲线的最低点附近停止训练。



![各迭代的训练与验证损失比较](plots/5066-0.json)



> 训练损失（蓝色）通常会持续下降，而验证损失（橙色）最初下降，但当模型过拟合时会开始上升。早期停止的目标是在验证损失曲线的最低点（虚线）附近停止训练。

### 其工作原理

早期停止之所以能起到正则化 (regularization)的作用，是因为它通过限制优化过程来间接约束模型的容量。通过在模型完全最小化训练损失之前停止训练，我们防止模型过于紧密地拟合训练数据的噪声和具体特征。验证损失最低点通常对应于模型在新数据上泛化能力最佳的点。

### 实际考量

- **耐心期：** 验证损失可能会有波动。在验证损失略微增加时立即停止训练可能为时过早。`patience` 参数 (parameter)允许训练再进行几个迭代，仅当在该窗口内未观察到改善时才停止。一个典型的耐心期值可能是5、10或更多迭代，具体取决于数据集和训练动态。
- **保存最佳模型：** 每当获得新的最佳验证分数时，保存模型的状态（权重 (weight)和偏差）非常必要。当训练停止时（无论是由于早期停止还是达到最大迭代次数），您都应加载保存的最佳模型状态，用于最终评估和部署。深度学习 (deep learning)框架通常提供回调或功能来自动处理此情况。
- **指标选择：** 尽管验证损失很常见，但对于某些问题（例如高度不平衡分类），监控验证准确率、F1分数或AUC可能更合适。选择最能反映模型在新数据上实际目标的指标。

早期停止是一种广泛使用、计算成本低且效果好的方法，用于防止过拟合 (overfitting)，并且通常能使模型在新数据上的泛化能力优于那些训练固定或可能过多迭代的模型。它只需要极少的配置，并且可以与其他正则化 (regularization)方法很好地配合使用。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7, Section 7.8
  对早停这一正则化方法提供了全面的理论和实践概览。
- [Early Stopping-But When?](https://doi.org/10.1007/3-540-69043-4_5) — Lutz Prechelt (1998)
  Journal: Neural Networks: Tricks of the Trade; Publisher: Springer Berlin Heidelberg; Pages: 55-69; DOI: [10.1007/3-540-69043-4_5](https://doi.org/10.1007/3-540-69043-4_5)
  一篇里程碑式的著作，讨论了神经网络训练中早停的理想停止点和实际考量。此为《神经网络：技艺窍门》一书中的一个章节。
- [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) — TensorFlow Team (2024)
  官方文档，详细说明了Keras深度学习框架中早停的实际实现，包括“耐心”等参数。
