---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-xgboost"
lesson: "xgboost-regularization"
sourceId: 7593
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/xgboost-regularization"
title: "XGBoost 中的正则化 (L1 和 L2)"
description: "了解 XGBoost 如何将其 L1 (Lasso) 和 L2 (Ridge) 正则化都纳入目标函数以对抗过拟合。"
order: 3
plots: []
sourceHash: "b6dcc80a94fd0856f442bc60bb2916706622a0303f36732f2370a5e8a964a92d"
sourceCorrections: []
---

XGBoost 采用形式化的正则化 (regularization)方法，这使其区别于许多传统梯度提升算法。不同于 Scikit-Learn 的 `GradientBoostingClassifier` 等方法通常通过 `max_depth` 和 `subsample` 等超参数 (parameter) (hyperparameter)来控制过拟合 (overfitting)，XGBoost 将正则化直接整合到其目标函数中。这种整合使得算法能够在树构建过程的每一步对模型复杂程度进行惩罚，从而得到泛化能力更强的模型。

正如章节介绍中提到的，XGBoost 目标函数由两部分组成：损失函数 (loss function)和正则化项。

$Obj(\Theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)$

第一部分 $l(y_i, \hat{y}_i)$ 用于衡量实际目标值 $y_i$ 与预测值 $\hat{y}_i$ 之间的差异。第二部分 $\Omega(f_k)$ 是正则化项，它对集成模型中每棵树 $f_k$ 的结构复杂程度进行惩罚。接下来，我们来仔细分析一下这个正则化项。

### XGBoost 中模型结构复杂程度的组成部分

XGBoost 中的正则化 (regularization)项 $\Omega$ 定义为：

$\Omega(f_k) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2 + \alpha \sum_{j=1}^{T} |w_j|$

这个公式可能看起来有些复杂，但它由一些简单的部分构成，这些部分控制着树结构的不同方面。

- **$T$**: 树中终端节点（即叶子节点）的数量。
- **$w_j$**: 分配给第 $j$ 个叶子节点的分数（或权重 (weight)）。对于落入此叶子节点的任何数据点，此值都将被添加到总预测中。
- **$\gamma$ (gamma)**: 一个超参数 (parameter) (hyperparameter)，用于控制增加更多叶子节点到树中的惩罚。
- **$\lambda$ (lambda)**: L2 正则化超参数。它惩罚叶子节点权重的数值大小。
- **$\alpha$ (alpha)**: L1 正则化超参数。它也惩罚叶子节点权重的数值大小，但使用不同的数学形式。

接下来我们看看这些组成部分如何帮助防止过拟合 (overfitting)。

### 使用 Gamma 惩罚叶子节点数量

第一项 $\gamma T$ 直接处理树中叶子节点的数量。叶子节点多的树结构更复杂，更容易拟合训练数据中的噪声。`gamma` 超参数 (parameter) (hyperparameter)（在 XGBoost 库中通常称为 `min_split_loss`）设定了添加新分割点的阈值。

当 XGBoost 评估一个潜在的分割点时，它会计算该分割点能带来的损失函数 (loss function)降低量。只有当这个损失降低量大于 `gamma` 值时，该分割点才会被接受。

- **较低的 `gamma` 值**（例如 0）意味着没有惩罚，算法会在每次损失降低时进行节点分裂，这可能导致过拟合 (overfitting)。
- **较高的 `gamma` 值**使得算法更为保守。它要求损失有显著降低才会创建新分支，从而在树生长时有效地进行剪枝。

这提供了一种更有原则的方式来控制树的大小，而不是简单地通过 `max_depth` 设置一个硬性限制。

### 使用 L2 和 L1 正则化 (regularization)平滑预测

正则化公式中的另外两项关注于叶子节点的权重 (weight) $w_j$。这些权重是每个弱学习器对最终预测贡献的实际值。如果这些值非常大，则表示单棵树有很强的影响力，使得模型对训练数据中的微小变化敏感。

#### L2 正则化 (lambda)

项 $\frac{1}{2} \lambda \sum w_j^2$ 是 L2 正则化惩罚，类似于岭回归中使用的形式。它惩罚叶子节点权重的平方大小。

- 通过惩罚较大的权重，L2 正则化促使叶子节点分数更小且分布更均匀。
- 这使得模型的预测更平滑，对任何单棵树的依赖性更小，从而提高其对新数据的泛化能力。
- `lambda` 超参数 (parameter) (hyperparameter)控制此惩罚的强度。默认值为 1，提供适度的正则化。增加 `lambda` 会使模型更趋于保守。

#### L1 正则化 (alpha)

项 $\alpha \sum |w_j|$ 是 L1 正则化惩罚，类似于 Lasso 回归。它惩罚权重的绝对值。

- 与 L2 类似，L1 正则化也抑制较大的叶子节点权重。
- L1 的一个独特之处在于它可以将权重推至恰好为零。尽管在参数调整中不如 `lambda` 常用，但在需要更稀疏模型的场景中它很有用。
- `alpha` 超参数控制此惩罚的强度。默认值为 0，这意味着除非指定一个值，否则它不会被应用。

下图展示了这些正则化惩罚在一个简单决策树上的应用位置。

> 这棵树的总结构复杂程度惩罚是施加于叶子节点数量 (T=3) 的 gamma 惩罚与施加于每个叶子节点权重 (-0.2, 0.3, 0.15) 的 L1/L2 惩罚之和。

通过将这些惩罚直接集成到目标函数中，XGBoost 在拟合训练数据和保持模型简单、泛化能力强之间取得了更明智的平衡。这种内置的防止过拟合 (overfitting)机制是其在比赛和生产系统中都非常有效和受欢迎的主要原因。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://dl.acm.org/doi/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍了XGBoost的基本设计，阐述了其目标函数中包含的正则化项，以提升模型泛化能力。
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) — XGBoost Contributors (2024)
  提供了XGBoost正则化超参数（包括gamma、lambda和alpha）的实际细节和配置指南。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本关于统计学习方法的综合资源，涵盖了正则化技术（L1和L2）以及集成方法（如梯度提升）。
- [Regression Shrinkage and Selection Via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) — Robert Tibshirani (1996)
  Journal: Journal of the Royal Statistical Society: Series B (Methodological); Publisher: Royal Statistical Society; Volume: 58; Pages: 267-288; DOI: [10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
  介绍了Lasso方法，阐述了L1正则化及其在权重收缩和模型稀疏性方面的特性。
