---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-xgboost"
lesson: "xgboost-architectural-improvements"
sourceId: 7592
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-4-advanced-gradient-boosting-xgboost/xgboost-architectural-improvements"
title: "相较于标准GBM的架构改进"
description: "查看XGBoost的技术改进，包括其优化的树构建算法和系统层面的效率。"
order: 2
plots: []
sourceHash: "8fe5e78799b43c241345c15f0686e2a5ad94887d0f066b7b3bf30e27d3bfdf6b"
sourceCorrections: []
---

虽然Scikit-Learn的`GradientBoostingClassifier`提供了经典算法的实现，但XGBoost从一开始就被设计为具有更高的效率和预测准确性。它的优点不仅是优化代码的结果，更源于显著的算法提升。这些改进解决了标准梯度提升机（GBM）的一些实际限制，使XGBoost成为一个更强大、更灵活的工具。

让我们看看使XGBoost区别于标准GBM实现的主要架构变化。

### 正则化 (regularization)的更系统方法

在标准GBM中，正则化通常是间接应用的。我们通过调整`max_depth`等超参数 (parameter) (hyperparameter)来限制树的大小，`subsample`来为每棵树使用部分数据，以及`learning_rate`来缩小每棵树的贡献，从而控制模型复杂度。尽管有效，但这本质上是启发式的。

XGBoost通过将其直接包含在它试图最小化的目标函数中来规范正则化。如本章概述中所述，目标函数包含两部分：训练损失和正则化项。


$$
Obj(\Theta) = \sum_{i=1}^n l(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$


第一部分，$l(y_i, \hat{y}_i)$，是衡量真实标签$y_i$与预测值$\hat{y}_i$之间差异的损失函数 (loss function)。第二部分，$\Omega(f_k)$，是XGBoost的创新之处。这一项惩罚添加到模型中的每棵树$f_k$的复杂度。这一惩罚的具体公式是：


$$
\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2
$$


让我们分解一下：

- **`T`** 是树中的叶子数量。
- **$\gamma$** (gamma) 是一个超参数，它控制添加新叶子的惩罚。较高的gamma值使算法更加保守，有效地充当了划分节点的复杂度成本。它促使剪除那些损失减少不足的节点。
- **`w_j`** 是第j个叶子的得分（或权重 (weight)）。
- **$\lambda$** (lambda) 是L2正则化超参数。这一项将叶子权重推向零，非常类似于岭回归。这有助于防止模型过于依赖任何一棵树的预测，使整个模型更稳定。

通过最小化这个组合目标，XGBoost在很好地拟合训练数据和保持模型简单之间做出了直接权衡。这种内置的正则化是它在对抗过拟合 (overfitting)方面表现出色的主要原因。

### 稀疏感知的分裂查找

实际数据集通常是稀疏的，包含许多缺失值或零项。大多数机器学习 (machine learning)算法要求你事先处理缺失值，例如，通过用平均值、中位数或常数来填充它们。

XGBoost通过其内置的**稀疏感知分裂查找**算法简化了这一过程。当在节点遇到缺失值时，XGBoost不会失败或要求填充。相反，在训练期间，它会为每个节点学习一个**默认方向**。

其工作原理如下：对于每个潜在的分裂点，算法通过评估两种情况来计算增益：

1. 将该特征所有缺失值的实例分配到左子节点。
2. 将该特征所有缺失值的实例分配到右子节点。

然后它选择产生更高增益（损失函数 (loss function)更大减少）的方向。当对含有缺失值的新数据进行预测时，它会将实例沿着该节点学习到的默认路径发送。这种方法比简单的填充更精巧，因为模型从数据本身学习处理缺失值的最佳方式。

### 使用近似分裂的高效树构建

对于具有许多连续特征的数据集，找到最优分裂可能计算量很大。贪婪算法需要评估每个特征的每个可能分裂点。对于大型数据集，这会成为性能瓶颈。

XGBoost采用**近似分裂查找算法**来加速这一过程。它不是枚举所有可能的分裂，而是首先根据特征分布的分位数提出一组有限的候选分裂点。然后算法只评估这些候选分裂以找到最好的一个。

这一过程通过**加权分位数草图**算法进一步优化。在梯度提升中，数据点并非一视同仁。那些被先前树预测较差的实例具有更大的梯度。XGBoost使用这些梯度（特别是二阶梯度，即Hessian）作为实例权重 (weight)。加权分位数草图算法在生成候选分裂时考虑这些权重，从而确保提议的分裂对模型难以处理的数据点更敏感。

### 追求可伸缩性和速度的系统设计

XGBoost库在算法层面上被设计为高性能。

- **缓存感知访问：** XGBoost将内存中的数据组织成名为“块”的特殊结构。它以与分裂查找算法一致的方式将数据预取到CPU缓存中。这最大限度地减少了缓存未命中，而缓存未命中是常见的性能瓶颈，从而使CPU能够更快地处理数据。
- **核外计算：** XGBoost可以处理过大而无法完全放入RAM的数据集。它可以从磁盘分块处理数据，使用数据压缩和分片来优化磁盘I/O。这使得在单台机器上训练大型数据集模型成为可能。

下图总结了标准GBM与XGBoost增强架构之间的主要差异。

> 标准梯度提升机和XGBoost架构方法的比较。XGBoost引入了正则化 (regularization)目标、原生缺失数据处理和优化的分裂查找作为主要特性。

总而言之，这些算法和系统层面的改进使得XGBoost成为一个快速、准确且可伸缩的梯度提升实现，非常适合多种机器学习 (machine learning)任务。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939671.2939785) — Tianqi Chen and Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939671.2939785](https://doi.org/10.1145/2939671.2939785)
  这篇基础论文介绍了XGBoost算法，详细阐述了其正则化目标函数、稀疏感知分裂查找和系统优化。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  这本教科书为梯度提升机和正则化技术提供了全面的理论背景，是理解XGBoost改进的基础。
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/) — XGBoost Contributors (2024)
  官方文档提供了XGBoost特性、超参数和实现细节的实用信息。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125004/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这是一本通过示例解释XGBoost算法改进和系统设计的实用书籍，使复杂主题变得易于理解。
