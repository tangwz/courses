---
course: "applied-data-science"
chapter: "building-tuning-predictive-models"
lesson: "supervised-learning-review"
sourceId: 1865
sourceUrl: "https://apxml.com/zh/courses/applied-data-science/chapter-3-building-tuning-predictive-models/supervised-learning-review"
title: "常用监督学习算法回顾"
description: "快速回顾常用回归和分类算法的思路。"
order: 1
plots: []
sourceHash: "3c8098eb70745f6d6d3b5fd47ba0947c265d3ad276938844cd322b9eff2cef34"
sourceCorrections: []
---

监督学习 (supervised learning)算法是构建预测模型的核心。这类算法旨在构建一个模型，该模型基于一组示例对 $(X, y)$，学习从输入特征（表示为 $X$）到输出目标变量 ($y$) 的映射。此处的“监督”来源于训练数据中已知的输出标签。

监督学习问题通常分为两大类别：

1. **回归：** 预测连续数值。例子包括预测房价、股票价值或温度。
2. **分类：** 预测离散类别或类别标签。例子包括将电子邮件分类为垃圾邮件或非垃圾邮件、识别猫或狗的图片，或预测客户流失（是/否）。

本章侧重于这两种类型问题的模型实现和调优。以下是所提及算法的简要回顾：

### 线性模型

线性模型因其简单性和可解释性而在机器学习 (machine learning)中非常基础。它们将目标变量建模为输入特征的线性组合。

#### 线性回归

这是许多回归任务的首选算法。它假设输入特征 ($X$) 与连续目标变量 ($y$) 之间存在线性关系。该模型旨在找到穿过数据点的最优拟合直线（或高维空间 (high-dimensional space)中的超平面）。

该关系由以下方程建模：


$$
y \approx \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
$$


这里，$y$ 是预测值，$x_1, \dots, x_n$ 是输入特征，而 $\beta_1, \dots, \beta_n$ 是模型学习到的系数（或权重 (weight)），表示在其他特征保持不变的情况下，对应特征每单位变化引起的 $y$ 的变化。$\beta_0$ 是截距项，表示当所有特征均为零时 $y$ 的期望值。

训练过程涉及找到使成本函数最小化的系数 $\beta$，成本函数通常是预测值与实际值之间的平方差之和（残差平方和，或称 RSS）。尽管方法直接，但其有效性依赖于某些假设，例如特征与目标之间的线性关系，以及误差的独立性。

#### 逻辑回归

尽管其名称如此，逻辑回归用于**分类**任务，主要是二元分类（两个类别，例如 0 或 1，是或否）。它对输入 $X$ 属于某个特定类别的概率进行建模。

它通过将输入的线性组合馈入 Sigmoid（或逻辑）函数来调整线性回归：


$$
P(y=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
$$


Sigmoid 函数将线性方程的输出压缩到 $[0, 1]$ 范围内，使其可以解释为概率。然后使用一个阈值（通常为 0.5）将此概率转换为类别预测：如果 $P(y=1|X) > 0.5$，则预测类别 1，否则预测类别 0。

训练过程涉及找到最大化训练数据中实际类别标签似然性的系数 $\beta$，通常使用梯度下降 (gradient descent)等技术。它提供了与结果的对数几率相关的可解释系数。

### 基于树的模型

基于树的模型将特征空间划分为一组矩形，并在每个矩形中拟合一个简单模型（例如一个常数）。它们具有多功能性，可以处理回归和分类任务。

#### 决策树

决策树构建的模型呈树形结构。它将数据集分解为越来越小的子集，同时逐步构建相关的决策树。

该过程从根节点开始，并根据最能分离目标变量的特征值递归地分割数据。常见的分裂标准包括分类任务中的基尼不纯度或信息增益（熵），以及回归任务中的方差减小。分裂持续进行，直到满足停止条件（例如，最大深度、每叶节点最小样本数）。终端节点被称为叶节点，它们包含预测结果（分类任务中的多数类别，或回归任务中的平均值）。

> 分类决策树结构的一个简单表示。每个内部节点测试一个特征，分支代表测试结果，叶节点分配一个类别标签。

决策树相对容易理解和可视化。然而，单个决策树容易出现过拟合 (overfitting)，这意味着它们会过度学习训练数据（包括其中的噪声），并且可能无法很好地泛化到新的、未见过的数据。

#### 随机森林

随机森林通过构建多个决策树的**集成**来解决单个决策树的过拟合问题。它是一种装袋（自举聚合）方法，并增加了一层随机性。

核心思路如下：

1. **自举抽样：** 通过从原始数据集中有放回抽样，创建多个不同的训练数据集。
2. **特征随机性：** 对于每棵树，在每个分裂点，仅考虑特征的一个随机子集，而非所有可用特征。
3. **构建树：** 使用随机特征子集，在每个自举数据集上训练一棵决策树。这些树通常生长得很深。
4. **聚合预测结果：** 对于回归任务，平均所有树的预测结果。对于分类任务，使用多数投票（所有树中预测最多的类别）。

自举和特征随机性的这种结合有助于降低个体树之间的相关性。集成模型通常比任何单个树都更准确，显著降低了方差和过拟合。其权衡之处在于，与单个决策树相比，它损失了直接可解释性。

### 梯度提升机 (GBM)

梯度提升是另一种强大的集成技术，它顺序地构建模型，每个新模型都试图纠正先前模型所犯的错误。与随机森林中树独立构建不同，GBM 是附加地构建树。

核心思路包括：

1. 从一个简单的初始模型开始（例如，回归预测均值，或分类预测对数几率）。
2. 迭代地向集成中添加新树。
3. 每棵新树都经过训练，用于预测前一个模型的**残差**（实际值与当前集成模型预测之间的差异）。
4. 这些新树的预测结果被添加到集成预测中，通常通过学习率（一个小的步长）进行缩放，以防止过拟合 (overfitting)。

这种顺序的、纠错的过程使得 GBM 能够非常有效地拟合数据，并且通常能实现高预测精度。流行且高效的实现包括：

- **XGBoost (极限梯度提升):** 以其速度、应对过拟合的正则化 (regularization)选项（L1 和 L2）、内置缺失值处理和交叉验证能力而著称。
- **LightGBM (轻量级梯度提升机):** 也侧重于性能，使用诸如基于梯度的单侧抽样（GOSS）和独占特征捆绑（EFB）等技术来加快训练速度，特别是在大型数据集上。它通常采用叶子生长策略而非层级生长策略。

“这些算法因其性能而在机器学习 (machine learning)竞赛和应用中常用。然而，与线性回归或单个决策树等简单模型相比，它们有更多超参数 (parameter) (hyperparameter)需要调整。”

本次回顾提供了基础。在接下来的章节中，我们将进入使用 scikit-learn 和其他相关 Python 库进行这些监督学习 (supervised learning)模型的实际实现、评估和调优阶段。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本标准教科书，为许多机器学习算法（包括线性模型、决策树以及集成方法如Bagging和Boosting）提供了统计学框架。
- [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324) — Leo Breiman (2001)
  Journal: Machine Learning; Publisher: Springer; Volume: 45; Pages: 5-32; DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A%3A1010933404324)
  介绍随机森林算法的原始学术论文，详细阐述了其构建和在减少过拟合方面的优势。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  本文介绍了XGBoost，这是一种流行且高效的梯度提升实现，专注于可伸缩性和正则化。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 3146-3154
  介绍LightGBM的研究论文，强调了其在梯度提升模型训练速度和效率方面的进步，尤其适用于大型数据集。
