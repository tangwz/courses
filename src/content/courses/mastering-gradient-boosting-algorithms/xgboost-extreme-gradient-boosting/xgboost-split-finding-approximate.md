---
course: "mastering-gradient-boosting-algorithms"
chapter: "xgboost-extreme-gradient-boosting"
lesson: "xgboost-split-finding-approximate"
sourceId: 1983
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting/xgboost-split-finding-approximate"
title: "分割查找算法：近似贪婪算法"
description: "了解大型数据集的近似算法。"
order: 4
plots: []
sourceHash: "1f932fdd0d0cced29a3cf2047fa32288cbb32dfdca81cb3b31abe90b7c7455b8"
sourceCorrections: []
---

精确贪婪算法能够确保找到连续特征所有可能值中的最佳分割点（给定当前树结构）。然而，其计算需求对于大型数据集来说可能变得难以承受。将每个独特值作为潜在分割点进行评估需要在每个节点对特征值进行排序，这种操作的扩展性较差，通常复杂度为 $O(n \log n)$，这里 $n$ 是节点中的实例数量。此外，重复访问和排序大量数据可能导致缓存未命中和内存瓶颈。

为应对这些扩展性问题，XGBoost 采用了一种近似贪婪算法。此方法通过减少评估的候选分割点数量，大幅加速了分割查找过程，尤其适用于大型数据集。

### 基于直方图的策略

近似算法的主要思想是，在训练之前或训练过程中的不同阶段，将连续特征离散化为有限数量的桶。算法不再将每个唯一的特征值视为潜在分割点，而只考虑这些离散桶*之间*的分割点。

其工作方式如下：

1. **特征分桶（量化 (quantization)）**：对于每个连续特征，算法将排序后的值分组到固定数量的离散桶中。这通常通过使用特征分布的分位数（百分位数）来完成。例如，如果我们选择 256 个桶，算法会根据该特征的值找到大约 255 个分割点，将数据分成 256 个大小大致相等的组。
2. **聚合统计量**：对于每个桶，XGBoost 计算并存储落入该桶的所有数据点的梯度之和 ($G$) 和海森之和 ($H$)。令 $I_k$ 为特征值落入桶 $k$ 的数据点索引集合。那么，桶 $k$ 的聚合统计量为：
   $G_k = \sum_{i \in I_k} g_i$
   $H_k = \sum_{i \in I_k} h_i$
3. **评估候选分割点**：算法只评估*桶边界处*的潜在分割。为了计算在桶 $j$ 和桶 $j+1$ 之间提议的分割点的增益，它高效地使用预先计算的和。左子节点（$L$，包含桶 $1...j$）和右子节点（$R$，包含桶 $j+1...N_{bins}$）的总梯度和海森简单地为：
   $G_L = \sum_{k=1}^{j} G_k, \quad H_L = \sum_{k=1}^{j} H_k$
   $G_R = \sum_{k=j+1}^{N_{bins}} G_k, \quad H_R = \sum_{k=j+1}^{N_{bins}} H_k$
   这些聚合值随后代入标准增益公式：
   $Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L+G_R)^2}{H_L+H_R + \lambda} \right] - \gamma$
   算法会找到使此增益最大的桶边界。

> 连续特征的值被映射到离散桶中。每个桶聚合其所包含实例的梯度 ($G$) 和海森 ($H$) 统计量。潜在分割点仅在这些桶的边界处进行评估。

### 提议方法：全局变体与局部变体

XGBoost 提供了两种主要方案来决定何时确定这些候选分割点（桶边界）：

1. **全局变体**：候选分割点在整个树构建过程开始时（甚至在训练开始前）*只提议一次*。相同的一组提议（桶边界）用于在特定树的生长过程中评估*所有*节点上的分割。这在计算上更经济，因为分桶过程每个特征只发生一次。
2. **局部变体**：候选分割点在*每次分割后*重新提议。也就是说，对于给定节点，算法仅考虑该节点中存在的数据点，基于此子集重新计算特征分位数，确定新的桶边界，并根据这些局部相关的边界提议分割。这在计算上更为密集，因为分桶会重复发生，但它可能带来更精确的分割，特别是在树的深层，数据分布可能与全局分布明显不同。

全局变体和局部变体之间的选择涉及一种权衡：

- **全局**：训练速度更快，内存使用量更少（提议只存储一次）。如果数据分布在树的不同分支中发生明显变化，准确性可能会降低。
- **局部**：计算成本更高，内存使用量可能更高。能更好地适应局部数据分布，可能带来更好的准确性，尤其对于更深的树。

实践中，全局变体通常足够且能大幅加速。桶的数量本身充当正则化 (regularization)参数 (parameter)；桶越少会导致更粗糙的分割，模型可能更快、正则化程度更高，而桶越多则允许更精细的分割，以增加计算成本为代价接近精确贪婪方法。XGBoost 中控制此粒度的参数通常与 `sketch_eps`（epsilon）相关，大致 `1 / sketch_eps` 表示桶的数量。

通过使用直方图并仅在桶之间评估分割，近似贪婪算法将分割查找的复杂度从依赖于唯一值的数量，变为依赖于桶的数量，而桶的数量通常要小得多。这使得 XGBoost 明显更快且内存效率更高，使其能够应对对于精确贪婪算法而言过大的数据集。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785-794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍并详细阐述了近似贪婪算法，包括基于直方图的分裂查找策略以及全局和局部变体的讨论。
- [Tree Methods](https://xgboost.readthedocs.io/en/stable/treemethod.html) — XGBoost Contributors (2024)
  解释了实现近似贪婪算法的`hist`树方法，并讨论了影响分箱的参数。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, and Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems (NIPS) 30; Publisher: Curran Associates, Inc.; Pages: 3146-3154
  介绍了另一个广泛使用的梯度提升框架，该框架高度依赖优化的基于直方图的算法来实现高效的分裂查找，为该技术提供了补充视角。
