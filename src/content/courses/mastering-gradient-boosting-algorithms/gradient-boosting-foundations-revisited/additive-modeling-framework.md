---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-foundations-revisited"
lesson: "additive-modeling-framework"
sourceId: 1918
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-1-gradient-boosting-foundations-revisited/additive-modeling-framework"
title: "加法模型框架"
description: "回顾顺序构建模型的思想。"
order: 3
plots: []
sourceHash: "4fba2d48bc57cf31df5320c413658c5f829aae2c9b2709a5eaa526dd056ce84c"
sourceCorrections: []
---

**加法模型**形式化了作为增强（boosting）算法基础的顺序学习过程。这个框架涉及集成方法，通常使用决策树作为基础学习器。与Bagging等方法不同，Bagging中基础学习器通常独立并行训练，而加法模型是迭代构建的。模型中添加的每个新组件都针对现有集成的不足之处。

### 顺序特性

设想您正在一步步构建一个预测模型。您从一个初始的、通常非常简单的预测开始。然后，您评估这个初始预测的不足之处。基于此评估，您添加一个新的组件，一个简单的模型，它专门用于弥补迄今为止所犯的错误。您重复此过程，通过添加处理剩余错误的组件，逐步改进整体模型。

这种迭代改进是加法模型的要旨。最终预测是按顺序构建的所有组件预测的总和（或相加）。

### 框架的形式化

从数学角度看，一个加法模型 $F_M(x)$ 在 $M$ 个阶段（或迭代）后对输入 $x$ 进行预测，可以表示为：


$$
F_M(x) = F_0(x) + \sum_{m=1}^{M} \beta_m h_m(x)
$$


让我们分解这个方程：

- $F_M(x)$: 在 $M$ 步之后，集成模型的最终预测。
- $F_0(x)$: 一个初始的基础模型或起始预测。对于许多问题，这只是训练集中目标变量的平均值（对于回归）或对数几率（对于分类）。它代表了我们在添加任何复杂学习器之前的最佳猜测。
- $h_m(x)$: 在第 $m$ 步添加的第 $m$ 个基础学习器（通常是决策树）。这个学习器经过专门训练，以处理构建到第 $m-1$ 步的模型（表示为 $F_{m-1}(x)$）的错误或不足。
- $\beta_m$: 应用于第 $m$ 个基础学习器预测的系数或权重 (weight)。该项控制每个新学习器对整体集成的贡献。在梯度增强中，这与学习率或收缩参数 (parameter)密切相关，后者通过减弱每个单独步骤的影响来帮助防止过拟合 (overfitting)。
- $M$: 基础学习器（或增强阶段）的总数。

### 迭代过程

模型构建过程如下：

1. **初始化：** 从初始模型 $F_0(x)$ 开始。这通常是一个常数值，它在训练数据上最小化损失函数 (loss function)（例如，平方误差损失的平均值）。
2. **迭代 $m=1$ 到 $M$：**
   - 计算当前集成模型 $F_{m-1}(x)$ 所产生的误差（残差或梯度，我们稍后会看到）。这些误差代表目标变量中“未被解释”的部分。
   - 训练一个新的基础学习器 $h_m(x)$ 来预测前一步的这些误差。 $h_m(x)$ 的目标是捕捉剩余误差中的模式。
   - 确定这个新学习器的最优系数 $\beta_m$（通常涉及最小化整体损失函数）。
   - 更新集成模型：$F_m(x) = F_{m-1}(x) + \beta_m h_m(x)$。

这个迭代过程可视化如下：

> 加法模型过程：从初始模型（$F_0$）开始，计算误差，将新的基础学习器（$h_m$）拟合到这些误差上，并将其添加到集成模型中（$F_m = F_{m-1} + \beta_m h_m$）。重复 $M$ 步。

### 为何使用加法模型？

这个框架的效能在于其灵活性和专注度。通过专注于前一个模型的误差，每个新学习器都能解决当前集成模型认为最困难的问题方面。这使得模型能够逐步提升性能，有可能捕捉到单个模型可能遗漏的复杂模式。

梯度增强是一个非常成功的算法系列，它在此加法模型框架内运作。它提供了一种具体的、有数学依据的方法来确定*如何*训练每个新的基础学习器 $h_m(x)$，以最佳地纠正 $F_{m-1}(x)$ 的误差，这其中运用了函数空间中的梯度下降 (gradient descent)思想。随着我们继续学习，我们将详细分析这种联系。理解这种加法结构是掌握GBM、XGBoost、LightGBM和CatBoost机制的基本。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本涵盖统计学习方法（包括集成方法、加性模型和提升算法）的综合性教科书。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了梯度提升机（GBM）框架，形式化了如何通过迭代地将基础学习器拟合到损失函数的负梯度来构建加性模型。
