---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-algorithm-depth"
lesson: "deriving-generic-gbm"
sourceId: 1933
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/deriving-generic-gbm"
title: "推导通用GBM算法"
description: "梯度提升算法的逐步推导。"
order: 2
plots: []
sourceHash: "bcabbfab7c00d4a39d9462114960e909b64fc4fbd2347ccc32b73b73b4c1e787"
sourceCorrections: []
---

加性建模框架中，模型是顺序构建的。标准梯度提升机（GBM）算法通过使用类似于梯度下降 (gradient descent)的过程（但在函数空间中运行）来最小化损失函数 (loss function)，从而构建这种序列。

### 目标：用加性模型最小化损失

我们的目的是找到一个函数 $F(x)$，它能最小化某个指定的损失函数 (loss function) $L(y, F(x))$ 的期望值，其中 $y$ 是真实目标值，$F(x)$ 是我们模型的预测。由于我们使用有限的训练集 $\{(x_i, y_i)\}_{i=1}^N$，因此我们旨在最小化经验损失：


$$
\mathcal{L}(F) = \sum_{i=1}^{N} L(y_i, F(x_i))
$$


梯度提升以加性方式构建模型 $F(x)$：


$$
F(x) = F_M(x) = \sum_{m=0}^{M} h_m(x) = F_0(x) + \sum_{m=1}^{M} h_m(x)
$$


在这里，$F_0(x)$ 是模型的初始估计（对于回归，通常是目标值的均值；对于分类，是对数几率），而 $h_m(x)$ 是在每次迭代 $m$ 中顺序添加的基础学习器（通常是决策树）。表示步长或权重 (weight)的项 $\beta_m$ 通常被吸收进学习器 $h_m(x)$，或由单独的学习率参数 (parameter)处理，后续会看到。

### 迭代步骤：函数梯度下降 (gradient descent)

假设我们已经构建了模型直到迭代 $m-1$，得到 $F_{m-1}(x)$。我们想找到下一个基础学习器 $h_m(x)$，使得将其添加到当前模型后，拟合度得以改善并减少整体损失：


$$
F_m(x) = F_{m-1}(x) + h_m(x)
$$


理想情况下，我们希望 $h_m(x)$ 指向能最大程度减少损失函数 (loss function) $\mathcal{L}$ 的方向。考虑迭代 $m$ 时的损失：


$$
\mathcal{L}(F_m) = \sum_{i=1}^{N} L(y_i, F_{m-1}(x_i) + h_m(x_i))
$$


可以将其视为在函数空间中从当前位置 $F_{m-1}(x)$ 迈出一步 $h_m(x)$。在标准梯度下降中，我们通过沿着损失函数相对于参数 (parameter)的负梯度方向移动来更新参数。在梯度提升中，我们做类似的事情：我们找到一个函数 $h_m(x)$，它指向损失函数 $\mathcal{L}$ 的负梯度方向，该负梯度是对于每个数据点 $i$，相对于当前模型的预测 $F_{m-1}(x_i)$ 评估得到的。

让我们计算损失函数 $L$ 相对于在每个点 $x_i$ 处的函数值 $F(x_i)$ 的梯度，该梯度在当前模型 $F_{m-1}$ 处评估得到：


$$
g_{im} = \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$


对于每个观测 $i$，我们想要移动的方向是负梯度，即 $-g_{im}$。这些负梯度值通常被称为 **伪残差**：


$$
r_{im} = - g_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$


**为什么是伪残差？** 如果我们使用平方误差损失 $L(y, F) = \frac{1}{2}(y - F)^2$，梯度为 $\frac{\partial L}{\partial F} = -(y - F)$。那么负梯度是 $-( -(y-F) ) = y - F$，这正是残差（真实值与预测值之间的差）。对于其他损失函数，$-g_{im}$ 表现得像一个残差，根据该特定损失函数，指示每个点的错误方向和大小。

### 将基础学习器拟合到伪残差

核心思想是拟合下一个基础学习器 $h_m(x)$，以近似这些伪残差 $\{r_{im}\}_{i=1}^N$。也就是说，我们使用原始特征 $x_i$ 但以 $r_{im}$ 作为目标变量来训练 $h_m(x)$：


$$
h_m \approx \arg \min_{h} \sum_{i=1}^{N} (r_{im} - h(x_i))^2
$$


通常，$h_m(x)$ 被限制为浅层决策树（例如CART）。该树被构建用于基于输入特征预测伪残差。

### 优化贡献（叶节点）

一旦树 $h_m(x)$ 的结构（即分裂）通过拟合 (overfitting)伪残差被确定，就需要确定树的叶节点（终端节点）$R_{jm}$ 的最优值 $\gamma_{jm}$。与其简单地使用每个叶子中的平均伪残差，我们可以找到每个叶子 $j$ 的常数值 $\gamma_{jm}$，它能最小化落入该叶子中的样本的原始损失函数 (loss function) $\mathcal{L}$：


$$
\gamma_{jm} = \arg \min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)
$$


这一步确保了新树的贡献直接最小化损失函数，考虑到当前模型 $F_{m-1}$。对于某些损失函数（如平方误差），这简化为叶子中伪残差的平均值，但对于其他函数（如对数损失或绝对误差），它需要不同的计算方法（例如，绝对误差的中间值）。然后，基础学习器 $h_m(x)$ 被定义为：如果 $x_i \in R_{jm}$ 则 $h_m(x_i) = \gamma_{jm}$。

### 使用收缩更新模型

最后，我们通过添加新训练的基础学习器 $h_m(x)$ 并乘以学习率 $\nu$（也称为收缩）来更新整体模型 $F(x)$：


$$
F_m(x) = F_{m-1}(x) + \nu h_m(x)
$$


学习率 $\nu$（通常是0.01到0.3之间的小值）缩放每棵新树的贡献。这减少了单棵树的影响，有助于防止过拟合 (overfitting)，迫使模型在多次迭代中更缓慢地构建其预测。它有效地提供了正则化 (regularization)。

### 通用GBM算法概述

我们现在可以概述通用梯度提升算法的步骤：

1. **初始化模型：** 从一个初始常数预测开始 $F_0(x) = \arg \min_{\gamma} \sum_{i=1}^{N} L(y_i, \gamma)$。

   - 对于使用平方误差损失的回归问题，$F_0(x)$ 是 $y$ 的均值。
   - 对于使用对数损失的二元分类问题，$F_0(x)$ 是与整体概率对应的对数几率。
2. **迭代 $m = 1$ 到 $M$（树的数量）：**
   a. **计算伪残差：** 对于每个样本 $i = 1, ..., N$：
   $r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$
   b. **拟合基础学习器：** 将基础学习器（例如回归树）$h_m(x)$ 拟合到伪残差 $\{ (x_i, r_{im}) \}_{i=1}^N$。这确定了树的区域（叶子）$R_{jm}$。
   c. **计算最优叶值：** 对于树 $h_m(x)$ 的每个叶节点（终端节点）$j = 1, ..., J_m$：
   $\gamma_{jm} = \arg \min_{\gamma} \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$
   d. **更新模型：** 使用学习率 $\nu$ 更新集成模型：
   $F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$
   （其中 $I(\cdot)$ 是指示函数。实际上，$h_m(x)$ 现在表示具有最优叶值 $\gamma_{jm}$ 的树。）
3. **输出最终模型：** 最终预测由 $F_M(x)$ 给出。对于分类，这可能被转换为概率（例如，对于二元分类，使用Sigmoid函数）。

> 梯度提升机（GBM）算法的迭代过程。每次迭代都包括计算梯度（伪残差）、将基础学习器拟合到这些梯度、优化学习器的贡献以及更新集成模型。

这种逐步推导说明了GBM如何通过顺序添加经过训练的基础学习器来迭代逐步改进其预测，这些学习器旨在纠正现有集成的错误（由损失函数 (loss function)的负梯度定义）。损失函数 $L$ 的选择决定了伪残差的具体形式以及可能的叶子优化步骤，使得GBM能够适用于各种回归和分类任务，后续我们将对此进行讨论。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇基础性论文介绍了梯度提升机（GBM）算法，详细阐述了其数学推导和函数梯度下降的通用框架。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/ESLII.pdf) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 745
  这本综合性教材深入阐述了各种机器学习算法的统计学解释，其中包括关于梯度提升及其理论基础的详细章节。（第二版）
- [Lecture Notes: Boosting (CS229, Machine Learning)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEVBpZbtJF-WDl8hhEt_ydoYluE4iLo0KzDwrL6Ri1gkAhl4cFGlEWQayQfDviYnR3It5nldiA3A5jHQ5GAGk7sIKxuSNd81Z3q9Utp2hJYvaIuDEwGvgu61AJdGKtdZv04_9Vy0eyYjeeVX_ZS) — Andrew Ng (2009)
  Publisher: Stanford University
  这些讲义清晰简洁地解释了提升算法，包括梯度提升的教学性推导，适合进行入门但严谨的理解。
