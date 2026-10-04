---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "the-gradient-boosting-machine"
lesson: "gbm-classification-loss-functions"
sourceId: 7582
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-2-the-gradient-boosting-machine/gbm-classification-loss-functions"
title: "分类任务的损失函数"
description: "了解分类任务的损失函数，如对数损失（二元交叉熵），并理解它们如何在GBM框架中应用。"
order: 5
plots: ["plots/7582-0.json"]
sourceHash: "94ad647c563cb9a03bd0d78d713e09c17d4d562456a5711388446734ad7826c4"
sourceCorrections: []
---

对于回归问题，梯度提升（Gradient Boosting）通常通过将新树拟合到残差来工作。然而，将其应用于分类任务需要不同的方式。分类任务涉及类别标签（例如，0或1）作为目标变量，因此无法简单地计算像 $y - \hat{y}$ 这样的残差。此外，模型的原始输出不是类别标签，而是一个连续分数，需要将其转换为概率。

为了解决这个问题，我们调整此框架以处理概率。主要思想是让模型预测正类的*对数几率*，这是一个从负无穷到正无穷的值。然后，我们可以使用逻辑（或S型）函数将此对数几率分数转换为介于0和1之间的概率。

这种设置使我们能够为分类问题使用更合适的损失函数 (loss function)，即**对数损失**，也称为二元交叉熵。

### 对数损失函数 (loss function)

对于二元分类问题，其中真实标签 $y$ 为0或1，单个观测值的对数损失定义为：


$$
L(y, p) = -[y \log(p) + (1 - y) \log(1 - p)]
$$


这里，$p$ 是正类（$y=1$）的预测概率。

分析此函数：

- 如果真实标签 $y=1$，损失变为 $-\log(p)$。当预测概率 $p$ 接近1时，损失较小；当 $p$ 接近0时，损失无限增大。
- 如果真实标签 $y=0$，损失变为 $-\log(1-p)$。当 $p$ 接近0时（即 $1-p$ 接近1时），损失较小；当 $p$ 接近1时，损失无限增大。

该函数对自信但错误的预测惩罚远大于对自信且正确的预测奖励，这是分类损失函数的一个理想特性。



![单个观测值的对数损失](plots/7582-0.json)



> 对数损失函数惩罚那些自信但错误的预测。当真实标签为1（蓝线）时，随着预测概率接近0，损失趋于无穷大。同样，当真实标签为0（红线）时，随着概率接近1，损失增大。

### 从损失到梯度

我们知道，梯度提升根据损失函数 (loss function)的负梯度训练新模型。对于使用MSE的回归问题，这个梯度仅仅是残差。现在，我们来找到使用对数损失的分类问题的对应部分。

首先，设 $F(x)$ 是我们当前集成模型对观测值 $x$ 的原始输出。此输出在对数几率空间中。我们使用逻辑函数将其转换为概率 $p$：


$$
p = \frac{1}{1 + e^{-F(x)}}
$$


现在，我们需要找到对数损失函数 $L(y, p)$ 相对于模型原始输出 $F(x)$ 的导数。使用链式法则，导数是：


$$
\frac{\partial L}{\partial F(x)} = \frac{\partial L}{\partial p} \cdot \frac{\partial p}{\partial F(x)}
$$


在完成导数推导（机器学习 (machine learning)课程中的常见练习）后，我们得到一个非常简单的结果：


$$
\frac{\partial L}{\partial F(x)} = p - y
$$


因此，负梯度，即我们的下一棵树将以此为训练依据的值，是：


$$
-\frac{\partial L}{\partial F(x)} = - (p - y) = y - p
$$


这个结果 $y - p$ 是分类的**伪残差**。它是实际标签（0或1）与模型当前预测概率之间的差值。例如，如果真实标签是1，模型预测概率是0.3，则伪残差是 $1 - 0.3 = 0.7$。下一棵树将训练以预测这个值，使整体模型的预测更接近正确答案。

这个巧妙的结果展示了梯度提升框架的效用。通过选择正确的损失函数，我们得到一个“类似残差”的量，这使得相同的顺序误差校正过程对分类问题有效，就像对回归问题一样。

### 多类别分类

此框架直接扩展到多类别分类问题。模型不再输出单个对数几率值，而是生成一个对数几率向量 (vector)，每个类别一个。然后，使用 **softmax函数** 将这些值转换为概率分布。

相应的损失函数 (loss function)是**多项式偏差**，常被称为类别交叉熵。过程保持不变：计算此损失函数对每个类别的负梯度，得到一组伪残差。通常在每个提升迭代中为每个类别训练一个单独的树，以预测这些伪残差，更新模型对每个类别的对数几率分数。虽然实现细节更为复杂，但将树拟合到梯度的基本机制保持不变。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1009210624) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1009210624](https://doi.org/10.1214/aos/1009210624)
  介绍了梯度提升机框架，详细阐述了其通用原理并将其应用于分类损失函数。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本涵盖统计学习方法的教科书，包括梯度提升、逻辑回归、交叉熵以及各种损失函数梯度的推导（第二版）。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher Bishop (2006)
  Publisher: Springer
  详细阐述了概率机器学习，包括逻辑回归、softmax和交叉熵误差函数。
