---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-foundations-revisited"
lesson: "introducing-gbm"
sourceId: 1926
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-1-gradient-boosting-foundations-revisited/introducing-gbm"
title: "介绍梯度提升机 (GBM)"
description: "原始梯度提升机算法概述。"
order: 5
plots: []
sourceHash: "9c5a4fa9c5e2a3e8734212c67d8bb460aa7f0871599f9e3d93790ccaf6e56296"
sourceCorrections: []
---

Jerome Friedman 提出的原始梯度提升机 (GBM) 算法整合了集成方法、决策树的作用、加性建模和梯度下降 (gradient descent)的思想。GBM 提供了一个具体且强大的框架，用于迭代地构建加性模型。

其核心在于，GBM 是梯度下降优化的一种应用，但它不是在固定的模型结构中优化参数 (parameter)，而是在函数空间中进行优化。设想您有一个经过 $m-1$ 次迭代构建的现有模型 $F_{m-1}(x)$。为了改进它，我们希望添加一个新函数（一个基学习器，通常是决策树）$h_m(x)$，使得新模型 $F_m(x) = F_{m-1}(x) + h_m(x)$ 最小化总体损失 $L(y, F(x))$。

GBM 的核心思想是选择新函数 $h_m(x)$ 指向损失函数 (loss function) $L$ 相对于当前模型的预测 $F_{m-1}(x)$ 的负梯度方向。对于给定的观测 $(x_i, y_i)$，负梯度，通常称为 *伪残差*，计算方式如下：


$$
r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$


这个 $r_{im}$ 表示预测 $F(x_i)$ 应该朝着哪个“方向”移动，才能最佳地减小观测 $i$ 的损失。算法然后使用输入特征 $x_i$ 拟合一个基学习器 $h_m(x)$ (例如，CART 回归树) 来预测这些伪残差。本质上，新树学习当前集成模型 $F_{m-1}(x)$ 所犯的错误，这些错误以损失函数梯度的形式表示。

整个GBM算法步骤如下：

1. **初始化:** 从一个简单的初始模型 $F_0(x)$ 开始。这通常是一个常数值，它在训练数据上使损失函数最小化。例如，对于平方误差损失，可以是目标值的均值；对于对数损失，可以是对数几率。
   
   $$
   F_0(x) = \arg \min_{\gamma} \sum_{i=1}^N L(y_i, \gamma)
   $$
   
2. **迭代:** 对于 $m = 1$ 到 $M$（其中 $M$ 是基学习器的总数）：
   a. **计算伪残差:** 基于当前模型 $F_{m-1}(x)$，计算所有观测 $i=1, ..., N$ 的负梯度（伪残差）$r_{im}$。
   b. **拟合基学习器:** 使用输入特征 $x_i$ 和计算出的伪残差 $r_{im}$ 作为目标，训练一个基学习器 $h_m(x)$（例如，回归树）。该树被拟合用于近似负梯度。
   c. **计算步长（乘数）:** 找到新添加的树 $h_m(x)$ 的最佳乘数 $\gamma_m$。这通常通过线搜索完成，以在添加新树时最小化损失：
   $\gamma_m = \arg \min_{\gamma} \sum_{i=1}^N L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$
   实际中，特别是结合收缩（稍后讨论）等技术时，通常使用固定学习率 $\eta$ 来代替或结合线搜索，因此更新中会包含 $\eta \cdot h_m(x)$ 或 $\eta \cdot \gamma_m \cdot h_m(x)$。
   d. **更新模型:** 通过添加按比例缩放的基学习器来更新集成模型：
   $F_m(x) = F_{m-1}(x) + \gamma_m h_m(x) \quad \text{（或通常 } F_m(x) = F_{m-1}(x) + \eta \gamma_m h_m(x) \text{）}$
3. **输出:** 最终模型是 $F_M(x)$。

> 梯度提升机算法的迭代过程。

GBM 的灵活性来自于损失函数 $L$ 的选择和基学习器 $h_m$ 的类型。虽然可以使用其他学习器，但回归树被广泛采用，因为它们自然地处理预测伪残差（连续值）的任务，并且能够捕获数据中复杂的非线性和交互。树本身的结构（分裂、深度）在步骤 2b 的拟合过程中确定。

常见的损失函数包括回归的均方误差 ($L(y, F) = \frac{1}{2}(y-F)^2$)，这会导致标准残差 ($r_{im} = y_i - F_{m-1}(x_i)$)；以及二元分类的对数损失（偏差）。损失函数的选择直接影响伪残差的计算方式，并决定模型如何优先修正不同类型的错误。

这一基本 GBM 算法提供了一种强大且通用的提升方法。然而，它存在一些局限性，特别是在计算效率、正则化 (regularization)和处理特定数据类型方面。这些局限性促使了 XGBoost、LightGBM 和 CatBoost 等高级算法的发展，我们将在后续章节中详细分析它们。它们在此核心 GBM 框架的基础上构建，通过引入复杂的正则化技术、优化的树构建算法以及针对各种数据特点的专门处理。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇奠基性论文介绍了梯度提升机（GBM）算法，详细阐述了其作为函数空间中贪婪优化的理论基础及其通用应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapter 10, "Boosting and Additive Trees"; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  这本综合性教材涵盖了各种统计学习方法，其中有一章专门对包括GBM在内的提升算法进行了详细的理论和实践解释。
- [Applied Predictive Modeling](https://doi.org/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; Pages: Chapter 8, "Boosting"; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  这本书从实践应用的角度介绍了预测建模技术，对GBM进行了清晰的解释，并探讨了在实际场景中实施GBM的考量。
