---
course: "introduction-to-deep-learning"
chapter: "building-training-dnns"
lesson: "monitoring-training-progress"
sourceId: 5059
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-5-building-training-dnns/monitoring-training-progress"
title: "监控训练进展（损失与指标）"
description: "在训练期间追踪损失和准确率（或其他指标）以评估性能。"
order: 8
plots: ["plots/5059-0.json", "plots/5059-1.json", "plots/5059-2.json", "plots/5059-3.json"]
sourceHash: "d6e59ef1513acbefe380a0c104139a083ddefeea68c7a9f5a0a67f1f7a84b380"
sourceCorrections: []
---

一旦您启动训练过程，通常是通过调用Keras中的`fit`函数或在PyTorch中管理训练循环来完成，框架便会根据训练数据开始迭代调整模型权重 (weight)。然而，训练神经网络 (neural network)并非一项可以简单启动后就放任不管的任务。密切关注训练的进展情况非常必要，以便了解模型是否正在有效学习、学习速度是否过慢，或者是否过度学习了训练数据，从而牺牲了对新数据的泛化能力（过拟合 (overfitting)）。

### 为何要监控？

监控训练可提供对学习动态的重要观察。通过随着时间（通常是跨训练周期）追踪重要指标，您可以诊断问题，就何时停止训练做出明智决定，并收集改进模型或训练设置的线索。您将监控的两种主要指标是损失和性能指标。

### 追踪损失：训练损失与验证损失

损失函数 (loss function)量化 (quantization)了模型预测值与训练期间实际目标值之间的差距。损失越低，通常表示模型在*正在评估的数据上*表现越好。监控两种类型的损失是标准做法：

1. **训练损失：** 根据当前训练步骤中用于梯度计算和权重 (weight)更新的数据批次进行计算。训练损失的降低表明模型正在学习适应训练数据。
2. **验证损失：** 根据模型*未曾训练过*的独立验证数据集进行计算。这提供了模型对未见数据的泛化能力的估计。比较训练损失和验证损失是诊断过拟合 (overfitting)的基本方法。

### 追踪性能指标：评估损失

尽管损失指导着优化过程，但它可能并非总是最直观的性能衡量标准。例如，知道交叉熵损失是0.1并不能立即告诉您有多少分类是正确的。因此，我们还会追踪与任务相关的性能指标。

1. **训练指标：** 在训练数据上计算，与训练损失类似。常见例子包括分类任务的准确率或回归任务的平均绝对误差（MAE）。
2. **验证指标：** 在验证数据集上计算。这些指标（例如，验证准确率）在训练期间提供了模型在未见数据上性能的更实际评估。

大多数机器学习 (machine learning)框架在您配置或编译模型时，可以轻松指定除损失函数 (loss function)外还需要追踪哪些指标。

### 框架如何报告进展

在训练期间，Keras等框架或PyTorch Lightning等库通常会在每个训练周期结束时打印损失和指标值。您可能会看到如下输出：

```
Epoch 1/20
1500/1500 [==============================] - 5s 3ms/step - loss: 0.4521 - accuracy: 0.8534 - val_loss: 0.2105 - val_accuracy: 0.9312
Epoch 2/20
1500/1500 [==============================] - 4s 3ms/step - loss: 0.1855 - accuracy: 0.9432 - val_loss: 0.1520 - val_accuracy: 0.9558
...
Epoch 20/20
1500/1500 [==============================] - 4s 3ms/step - loss: 0.0412 - accuracy: 0.9870 - val_loss: 0.0950 - val_accuracy: 0.9715
```

此输出显示了训练周期数、周期内进度、耗时、训练损失、训练准确率、验证损失和验证准确率。

此外，训练函数通常会返回一个`history`对象（名称可能因框架而异）。此对象存储了每个训练周期记录的损失和指标值，让您能够在过程完成后分析和可视化训练趋势。

```python
# 使用Keras history对象的示例
# history = model.fit(train_data, train_labels, epochs=20, validation_data=(val_data, val_labels))

# 访问记录的数据
# training_loss = history.history['loss']
# validation_loss = history.history['val_loss']
# training_accuracy = history.history['accuracy']
# validation_accuracy = history.history['val_accuracy']

# 之后您可以使用Matplotlib或Plotly等库来绘制这些值
```

### 解释训练曲线

将训练损失/指标和验证损失/指标随训练周期绘制成图，是理解训练动态最有效的方法。以下是一些常见模式：

#### 正常收敛

理想情况下，训练损失和验证损失都稳定下降并趋于收敛，而训练指标和验证指标则增加并趋于收敛。这表明模型学习良好并能有效泛化。



![正常收敛：损失](plots/5059-0.json)



> 训练损失和验证损失一同下降并趋于平稳。



![正常收敛：准确率](plots/5059-1.json)



> 训练准确率和验证准确率一同上升并趋于平稳。

#### 过拟合 (overfitting)

一种常见问题，模型过度具体地学习训练数据，包括其噪声和特殊性。这导致模型在新的、未见数据上表现不佳。过拟合通常在以下情况被识别：

- 训练损失持续下降。
- 验证损失在某个点后开始增加。
- 训练指标可能继续改善。
- 验证指标趋于平稳或开始下降。



![过拟合：损失](plots/5059-2.json)



> 训练损失持续下降，而验证损失开始增加，表明出现过拟合。

#### 欠拟合 (underfitting)

这发生在模型过于简单而无法捕捉数据中隐藏模式，或者训练周期不足时。迹象包括：

- 训练损失和验证损失都保持较高水平或迅速在高值处趋于平稳。
- 训练指标和验证指标都较低，且未能明显改善。



![欠拟合：损失](plots/5059-3.json)



> 训练损失和验证损失都在高值处趋于平稳，表明模型无法充分学习数据模式。

#### 其他问题

- **噪声曲线：** 非常锯齿状或不稳定的损失/指标曲线可能表明学习率过高，导致优化过程冲过极小值点。
- **收敛缓慢：** 如果损失下降非常缓慢，学习率可能过低，或者优化算法可能停滞不前。

### 验证集的作用

在独立于训练集和最终测试集的验证集上监控性能非常必要。验证集提供了模型在训练*期间*泛化情况的无偏估计。它有助于早期检测过拟合 (overfitting)，并为何时停止训练（一种称为“早期停止”的技术，稍后讨论）提供信息。最终测试集只能在训练完成且模型选择/调优（基于验证集）结束后使用，以便对所选模型的性能进行最终、无偏的评估。

在即将到来的实践练习中，您将在MNIST数据集上训练一个分类器，请密切关注这些曲线（`loss`、`accuracy`、`val_loss`、`val_accuracy`）。观察这些趋势是应用神经网络 (neural network)学习中的一项基本技能。理解这些模式是解决潜在问题的第一步，这通常涉及正则化 (regularization)或超参数 (parameter) (hyperparameter)调优等技术，我们很快将介绍这些话题。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  关于“深度模型训练优化”、“深度学习正则化”和“深度模型经验评估”的章节提供了关于损失函数、指标、过拟合以及验证集作用的全面理论和实践见解。
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  第3版。本书提供了使用 Keras 和 TensorFlow 实现和监控深度学习模型的实践指导，包括对训练/验证损失/指标、学习曲线的解释以及诊断过拟合和欠拟合等常见问题的详细说明。
- [Guide to training and evaluation](https://www.tensorflow.org/guide/keras/train_and_evaluate) — fchollet (2023)
  Publisher: TensorFlow
  Keras 官方文档提供了关于如何配置和监控模型训练的直接且最新的指南，包括使用 `model.fit`、解释 `History` 对象以及指定损失函数和指标。
