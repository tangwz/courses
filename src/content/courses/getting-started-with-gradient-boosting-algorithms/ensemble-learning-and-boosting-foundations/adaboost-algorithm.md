---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "ensemble-learning-and-boosting-foundations"
lesson: "adaboost-algorithm"
sourceId: 7575
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations/adaboost-algorithm"
title: "AdaBoost算法：梯度提升算法的前身"
description: "详细了解自适应提升（AdaBoost）算法、其赋权机制以及它在现代提升方法中的核心作用。"
order: 4
plots: []
sourceHash: "644491389e47a0799eed511b6ad9ef691d0ccaa7e23822e8997163b37e52620a"
sourceCorrections: []
---

提升法的一般原理是按顺序训练模型以纠正之前的错误。具体*如何*进行这些纠正的机制决定了算法的特点。AdaBoost，即自适应提升的简称，是最早且最具影响力的提升算法之一，它为这种顺序学习过程提供了一个具体的方法体系。它为理解更普遍的梯度提升方法提供了很好的起点。

AdaBoost的“自适应”特性源于其在每一步中重新赋予数据点权重 (weight)的方法。它不是在整个训练过程中将每个数据点视为同样重要，而是根据它发现难以分类的点来调整其关注点。

### 核心思想：侧重于错误

想象一下你在备考。你可能首先平均回顾所有主题。练习测试后，你识别出自己的薄弱环节。在下一次学习时，你自然会花更多时间在你做错的主题上。AdaBoost也遵循类似原理。

该过程始于为训练数据集中每个样本分配一个相等的权重 (weight)。然后，一个弱学习器（通常是一个决策桩，即只包含一次分割的决策树）在该数据上进行训练。第一轮之后，算法会评估学习器的表现。

这是主要步骤：*被错分*的数据点的权重会增加，而被正确分类的点的权重则会减少。在下一次迭代中，会训练一个新的弱学习器，但这一次它必须更加关注权重较高的样本，即前一个模型难以处理的样本。这促使新的学习器侧重于更难的案例。这种训练、评估和重新赋权的循环会重复预设的迭代次数。

> AdaBoost的顺序过程。每次迭代都会在已重新赋权的数据上训练一个新的弱学习器，以突出前一个学习器的错误。最终模型通过加权投票组合所有学习器。

### 加权投票：让模型拥有“发言权”

AdaBoost不仅仅是通过简单的多数投票来组合弱学习器。有些学习器必然比其他学习器更准确。为此，每个弱学习器都会被分配一个权重 (weight)，或者说“发言权份额”，根据其在训练轮中的整体表现。具有较低加权错误率的模型在最终预测中会有更大的发言权，而表现仅略好于随机猜测的模型则发言权很小。

最终预测由所有已训练弱学习器的加权投票决定。这确保了更准确的模型对结果有更大的影响。

### 算法步骤概述

我们来概述一下二元分类问题的过程，其中输出为 -1 和 1。

1. **初始化样本权重 (weight)：** 对于包含 $N$ 个样本的数据集，每个样本 $i$ 被赋予初始权重 $w_i = 1/N$。
2. **迭代与训练：** 对于 $M$ 次迭代中的每一次（其中 $M$ 是要训练的弱学习器数量）：

   - **a. 训练一个弱学习器：** 将弱分类器 $h_m(x)$ 拟合到训练数据，以最小化加权分类误差。
   - **b. 计算加权误差 ($\epsilon_m$)：** 对所有错分样本的权重求和。

   
   $$
   \epsilon_m = \sum_{i=1}^{N} w_i \cdot \mathbb{I}(y_i \neq h_m(x_i))
   $$
   

   其中 $\mathbb{I}$ 是一个指示函数，如果条件为真则为 1，否则为 0。

   - **c. 计算模型的权重 ($\alpha_m$)：** 此值代表模型的“发言权”。它根据错误率计算。

   
   $$
   \alpha_m = \frac{1}{2} \ln \left( \frac{1 - \epsilon_m}{\epsilon_m} \right)
   $$
   

   较低的误差 $\epsilon_m$ 会得到更高、正值的 $\alpha_m$。误差为 0.5（随机猜测）会导致 $\alpha_m = 0$，使模型没有发言权。

   - **d. 更新样本权重：** 使用模型的权重 $\alpha_m$，增加错分样本的权重，减少正确分类样本的权重。

   
   $$
   w_i \leftarrow w_i \cdot \exp(-\alpha_m y_i h_m(x_i))
   $$
   

   此步骤之后，权重通常会归一化 (normalization)，使其总和为 1。项 $y_i h_m(x_i)$ 对于错分是 -1，对于正确分类是 +1，这推动了权重的更新。
3. **形成最终模型：** 最终强分类器 $H(x)$ 结合了所有弱学习器的预测，根据它们各自的 $\alpha_m$ 值进行加权。结果和的符号决定了最终的类别预测。

   
   $$
   H(x) = \text{符号}\left(\sum_{m=1}^{M} \alpha_m h_m(x)\right)
   $$
   

### 从AdaBoost到梯度提升

AdaBoost为顺序错误校正提供了一个清晰的示例。它通过调整*数据样本*的权重 (weight)来工作，以强制后续模型侧重于难分类的情况。这种机制非常有效，但主要为分类任务设计。

梯度提升沿用了这个核心思想并将其泛化。梯度提升模型不侧重于样本权重，而是训练来纠正前一个模型的*残差错误*。对于回归问题，残差就是实际值与预测值之间的差值。对于分类问题，“误差”是一个更复杂的值，它源自损失函数 (loss function)的梯度。

这种从重新赋权样本到拟合残差错误的转变，使得梯度提升能够应用于各类问题，包括回归，并能使用任何可微的损失函数。AdaBoost的样本赋权方法可以看作是这种更通用错误校正方法体系的一个具体例子。理解AdaBoost如何通过重新赋权迭代地最小化错误，为在下一章掌握梯度提升如何通过追踪梯度做同样的事情，提供了一个有力的直观依据。

## 参考资料

- [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009990168X) — Yoav Freund and Robert E. Schapire (1999)
  Journal: Journal of Computer and System Sciences; Publisher: Academic Press; Volume: 55; Pages: 119-139; DOI: [10.1006/jcss.1999.1681](https://doi.org/10.1006/jcss.1999.1681)
  介绍了原始的AdaBoost算法及其理论框架。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本标准教科书，全面地从统计学角度分析了AdaBoost及其与其他集成和提升方法的关系。
- [Boosting](https://cs229.stanford.edu/notes2023fall/cs229-notes-boosting.pdf) — Andrew Ng (2023)
  Publisher: Stanford University
  斯坦福大学CS229机器学习课程的讲义，清晰地解释了AdaBoost的工作原理。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了梯度提升框架，展示了它如何从AdaBoost等算法中得到推广。
