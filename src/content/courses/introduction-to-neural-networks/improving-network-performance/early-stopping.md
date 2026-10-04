---
course: "introduction-to-neural-networks"
chapter: "improving-network-performance"
lesson: "early-stopping"
sourceId: 2034
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-6-improving-network-performance/early-stopping"
title: "早期停止：优化训练终止"
description: "使用验证表现，在过拟合发生前停止训练。"
order: 5
plots: ["plots/2034-0.json"]
sourceHash: "b7e610a4cd2d0043172fd0f6ac83f224426a8c05d648963039476765285d84f6"
sourceCorrections: []
---

在训练神经网络 (neural network)时，您通常会发现训练损失持续下降。这是合理的；模型在拟合训练数据方面表现越来越好。然而，如本章前面所说，仅仅最小化训练损失并非最终目标。我们希望模型能够良好地泛化到新的、未见过的数据。我们如何知道何时停止训练以达到最佳泛化效果？训练的轮次过少可能导致欠拟合 (underfitting)，而训练的轮次过多常会导致过拟合 (overfitting)。

早期停止提供了解决此问题的简单而有效的方法。其主要思路是，在训练过程中，监控模型在单独验证集上的表现，并在验证集上的表现不再提升或开始变差时停止训练，即使训练损失仍在下降。

### 监控验证表现

在每个训练轮次之后（或有时在固定数量的批次之后），需要评估模型在训练数据和验证集上的表现。在验证集上，通常会追踪一个特定的衡量指标，最常见的是验证损失，但它也可以是准确率、F1分数或其他根据特定问题而定的相关指标。

最初，随着模型学习，训练损失和验证损失都趋于下降。然而，在某个时刻，模型可能开始过度适应训练数据的特定模式和噪声。这是过拟合 (overfitting)的开始。当这种情况发生时，您通常会发现训练损失继续下降，而验证损失趋于平稳，或者更明显地，开始上升。这种分歧是一个明确的信号，表明模型的泛化能力正在下降。

下图说明了这种常见模式：



![训练损失与验证损失对比](plots/2034-0.json)



> 训练损失通常随训练轮次增加而下降。验证损失最初下降，但当模型开始过拟合时会开始上升。早期停止的目标是在验证损失最小的点（由标记 (token)指示）附近停止训练。

### 早期停止机制

实施早期停止涉及几个重要的组成部分：

1. **监控指标：** 选择一个在验证集上追踪的指标（例如，验证损失、验证准确率）。
2. **频率：** 决定多久在验证集上评估一次模型（例如，每个训练轮次之后）。
3. **最佳表现追踪：** 记录训练期间目前观察到的最佳验证指标值。
4. **模型检查点：** 每当取得新的最佳验证表现时，保存模型的当前状态（权重 (weight)和偏置 (bias)）。这很重要，因为*最佳*模型可能不是停止前的最后一个训练轮次所对应的模型。
5. **容忍度：** 引入一个“容忍度”参数 (parameter)。这是在停止训练前，等待验证指标提升的训练轮次数量。例如，如果容忍度设置为10，那么如果验证指标连续10个训练轮次没有提升，训练就会停止。容忍度有助于避免因验证指标的噪声波动而过早停止。
6. **停止标准：** 如果监控的验证指标在指定的`容忍度`训练轮次内没有提升，则停止训练。
7. **恢复最佳权重：** 训练停止后，丢弃最后一个训练轮次的模型权重，并加载训练期间观察到的与最佳验证表现相对应的已保存权重。这确保您最终得到泛化表现最好的模型。

### 实践考量

- **选择指标：** 验证损失通常是一个好的默认选择，因为它与模型优化的目标函数直接相关。然而，如果某个特定的评估指标（如准确率或F1分数）对您的应用更重要，您可以选择监控该指标。请注意，像准确率这样的指标可能不如损失对微小提升那么敏感，尤其是在训练早期或不平衡数据集上。
- **设置容忍度：** 最佳的`容忍度`值取决于数据集、模型复杂度和批次大小。值过低可能导致训练过早停止；值过高可能在停止前允许过拟合 (overfitting)发生。5到20之间的值是常见的起始点，但通常需要进行实验。
- **最小变化量：** 一些实现包含一个`min_delta`参数 (parameter)，它定义了监控量需要变化的最小值才能算作提升。这有助于忽略可能只是噪声的微不足道的提升。

早期停止是一种广泛采用且效果好的方法，用于防止过拟合。它通过训练时长隐式地控制模型的容量，起到了一种正则化 (regularization)作用。它计算效率高，相对容易实现，并且通常在泛化表现上带来显著提升，且调优工作量较少，相比于明确添加$L1$或$L2$等正则化项。大多数深度学习 (deep learning)框架提供了便于将早期停止集成到训练循环中的方式。

## 参考资料

- [Early Stopping - But When?](https://doi.org/10.1007/3-540-49430-8_3) — Lutz Prechelt (1996)
  Journal: Neural Networks: Tricks of the Trade; Publisher: Springer, Berlin, Heidelberg; Volume: 1524; Pages: 55-69; DOI: [10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3)
  探讨了早期停止的理论和实践方面，包括停止标准和经验评估。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 247-249
  提供了深度学习的全面理论和实践介绍，其中包含作为正则化方法的早期停止。
- [tf.keras.callbacks.EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping) — TensorFlow Contributors (2023)
  Keras深度学习框架中实现早期停止的官方文档，详细说明了耐心值、最小变化量和监控指标等参数。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Lecture Notes on Regularization](http://cs231n.github.io/neural-networks-3/#reg) — Stanford CS231n Course Staff (2023)
  关于正则化技术的部分，其中包含早期停止的清晰解释和实用见解。
