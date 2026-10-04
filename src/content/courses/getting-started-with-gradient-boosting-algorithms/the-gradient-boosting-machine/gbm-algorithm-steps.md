---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "the-gradient-boosting-machine"
lesson: "gbm-algorithm-steps"
sourceId: 7580
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-2-the-gradient-boosting-machine/gbm-algorithm-steps"
title: "梯度提升机算法分步详解"
description: "按照清晰、分步的说明了解梯度提升机算法，从初始化到最终预测。"
order: 3
plots: []
sourceHash: "f944205e6fb01b6b61f5b8b2eccf79002f6e6222e67cedbb7959388721186af5"
sourceCorrections: []
---

了解了梯度提升如何通过损失函数 (loss function)的负梯度训练模型后，我们现在可以将这些组成部分整合起来，形成一个完整的、分步进行的算法。按顺序查看这个过程有助于弄清每棵新树如何改进整体模型。

该算法迭代地改进其预测。它从一个简单的初始猜测开始，然后，在指定轮次内，构建一棵新树，旨在纠正当前模型产生的错误。

> 梯度提升机算法的迭代过程。每个循环都会添加一棵新树，旨在纠正当前集成模型中的错误。

我们来详细分析每一步。为了说明，我们将专注于使用均方误差 (MSE) 作为损失函数的回归问题，在此问题中，该过程最直接明了。

### 步骤 1：用一个常数值初始化模型

在我们开始纠正错误之前，需要一个初始预测。在没有任何特征的情况下，我们能为所有样本做出的最佳单一预测是什么？对于 MSE，使总体误差最小的常数值是目标变量 $y$ 的均值。

因此，我们的初始模型 $F_0(x)$ 就是这个平均值：


$$
F_0(x) = \bar{y}
$$


这个单一值作为训练集中每个观测值的起点或“零次”预测。

### 步骤 2：迭代并构建树

现在，我们进入一个循环，它将运行预设的 $M$ 次迭代，每次迭代都会为我们的模型添加一棵新树。我们来看看第 $m$ 次迭代（从 $1$ 到 $M$）循环内部发生的情况。

#### 2a. 计算伪残差

梯度提升的主要部分是在之前模型的误差上训练新模型。正如我们所确定的，这种“误差”在形式上是损失函数 (loss function)的负梯度。我们将这些值称为**伪残差**。

对于每个样本 $i$，伪残差 $r_{im}$ 是根据前一个模型的预测 $F_{m-1}(x_i)$ 计算的：


$$
r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x)=F_{m-1}(x)}
$$


对于 MSE 损失函数，其中 $L(y_i, F(x_i)) = \frac{1}{2}(y_i - F(x_i))^2$，这个导数可以很好地简化。负梯度变为：


$$
r_{im} = y_i - F_{m-1}(x_i)
$$


这仅仅是实际值减去预测值，也就是标准的残差。对于第一次迭代（$m=1$），伪残差就是目标值减去总体均值：$r_{i1} = y_i - \bar{y}$。

#### 2b. 将基础学习器（决策树）拟合到伪残差上

接下来，我们拟合一个新的弱学习器，我们称之为 $h_m(x)$，来预测我们刚刚计算的伪残差。这是过程的主要部分：**新树不是用来预测原始目标 $y$，而是预测当前的残差。**

- **输入特征：** 来自我们数据集的原始特征 $X$。
- **目标：** 来自步骤 2a 的伪残差 $\{r_{1m}, r_{2m}, \dots, r_{Nm}\}$。

由此产生的树 $h_m(x)$ 会学习特征与当前模型残差之间的关系。例如，如果模型持续低估了某一类样本，该树将学会为它们输出一个正值。

#### 2c. 更新集成模型

我们现在通过将新树的预测添加到我们之前的模型来更新整体模型。但是，我们不仅仅是添加完整的预测。我们通过一个称为**学习率**（通常表示为 $\eta$，eta 或 `alpha`）的小因子对其进行缩放。

更新规则是：


$$
F_m(x) = F_{m-1}(x) + \eta h_m(x)
$$


学习率是一种正则化 (regularization)技术。较小的学习率会减小每棵树的贡献，这需要集成模型中包含更多的树，但通常会带来更好的泛化能力。它防止模型因添加一棵树而发生剧烈变化，使学习过程更稳定。$\eta$ 的常见值在 0.01 到 0.3 之间。

### 步骤 3：输出最终模型

循环完成后（即，我们已构建所有 $M$ 棵树），我们的最终模型是初始预测与所有树的按比例贡献的总和。

新观测值的最终预测由以下公式给出：


$$
\hat{y} = F_M(x) = F_0(x) + \sum_{m=1}^{M} \eta h_m(x)
$$


这个最终模型代表了一个复杂的函数，它是迭代构建的，每个组成部分都擅长纠正之前模型序列中遗留的错误。通过沿着负梯度方向小心地迈出小步，模型逐渐减小了总体损失。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/euclid.aos/1013203450) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203450](https://doi.org/10.1214/aos/1013203450)
  这篇原始论文介绍了梯度提升机算法，详细阐述了其数学基础和迭代过程。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  第二版。第10章对梯度提升进行了全面解释，包括逐步算法和各种损失函数。
- [1.11. Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) — scikit-learn developers (2024)
  scikit-learn官方文档在集成方法部分清晰地概述了梯度树提升（GBM），包括算法的一般步骤。
