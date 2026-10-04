---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "advanced-gradient-boosting-lightgbm-catboost"
lesson: "catboost-categorical-features"
sourceId: 7600
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost/catboost-categorical-features"
title: "CatBoost 简介：处理类别特征"
description: "了解 CatBoost 自动处理类别特征的创新方法，包括有序提升和独热编码优化。"
order: 3
plots: []
sourceHash: "89ed6149b492eac44664d837b94441762f617906f3d24cdce23a33663367bbbf"
sourceCorrections: []
---

梯度提升算法（例如 LightGBM）通过数据采样和特征捆绑等技术来优先考虑速度。相比之下，CatBoost 解决了一个不同但同样有分量的问题：有效且原生处理类别数据。由 Yandex 开发的 CatBoost（类别提升）从头开始设计，旨在解决传统方法在梯度提升模型中纳入非数值特征的不足之处。

### 树模型中类别数据的问题

大多数机器学习 (machine learning)算法，包括梯度提升，都处理数值数据。当数据集包含 `user_country`、`product_category` 或 `browser_type` 等类别特征时，这会带来问题。标准做法是将这些列预处理成数值格式，但最常见的方法对于树模型来说有明显的缺点：

- **独热编码 (OHE)：** 此方法为每个唯一类别创建一个新的二元列。虽然对低基数（少量唯一值）特征有效，但对于包含数百或数千个类别的特征来说，它会变得有问题。这种“维度灾难”会创建极其宽泛和稀疏的数据集，使得树算法难以找到最优分割点，并大大增加内存使用和训练时间。
- **标签编码：** 此方法为每个类别分配一个唯一整数（例如，`{'Chrome': 0, 'Firefox': 1, 'Safari': 2}`）。这在计算上是高效的，但会引入一种人为且误导性的序数关系。模型可能会错误地认为 `Safari > Firefox`，这种比较毫无意义，并可能损害模型性能。

CatBoost 提供了一种复杂的内置解决方案，可以避免这些问题。

### 基于目标的编码和数据泄露风险

一种更高级的技术是目标编码（或均值编码）。在这里，每个类别都被替换为一个数字，代表该类别目标变量的平均值。例如，在点击率预测任务中，类别“Firefox”可能会被所有使用 Firefox 的用户的平均点击率所取代。

这种方法很强大，因为它直接编码了特征与目标之间关系的信息。然而，它带有一个主要风险：**目标泄露**。如果您使用数据点自身的目标值来计算其编码，您就会将目标信息泄露到特征中。模型可能会轻易过拟合 (overfitting)，因为它学习将编码值直接与结果关联起来，从而无法泛化到新的、未见过的数据。

### CatBoost 的解决方案：有序目标统计

CatBoost 实现了一种巧妙的目标编码变体，称为**有序目标统计**，以防止数据泄露。它不是一次性计算整个数据集的统计数据，而是依赖于一个排序原则。核心思想是，对于任何给定的数据点，其类别特征的编码仅使用在随机打乱的数据集中位于它之前的数据点的目标统计数据来计算。

我们来细看这个过程：

1. **随机排列：** 训练数据集首先按随机顺序打乱。这种排序是临时的，仅用于计算编码。
2. **迭代编码：** 算法逐行处理打乱后的数据集。对于第 $i$ 个数据点，其类别特征的值仅使用数据集中与它属于同一类别的、位于它之前的 $i-1$ 个数据点的目标值来计算。

这种顺序依赖确保了某行的目标值永远不会被用来计算其自身的特征编码，从而防止了直接的数据泄露。下图说明了单一排列的这一流程。

> 对于随机排序数据集中的每一行，类别特征编码仅使用具有相同类别的先前行的目标值来计算。这可以防止模型使用样本自身的目标值来定义其特征。

为了使这个过程更稳定，尤其是在数据集开始阶段历史数据较少时，CatBoost 引入了一个先验值。编码的公式大致如下：


$$
\text{编码值} = \frac{\text{类别内计数} + \text{先验值}}{\text{总计数} + 1}
$$


包含权重 (weight)参数 (parameter) $a$ 和先验值 $P$ 的更通用形式是：


$$
\hat{x}_{ik} = \frac{\sum_{j=1}^{i-1} [\mathbf{x}_{jk} = \mathbf{x}_{ik}] \cdot y_j + a \cdot P}{\sum_{j=1}^{i-1} [\mathbf{x}_{jk} = \mathbf{x}_{ik}] + a}
$$


这里，$\hat{x}_{ik}$ 是第 $i$ 个样本和第 $k$ 个特征的新数值特征。该公式使用具有相同类别值的先前样本的目标值（$y_j$）。为了进一步提高健壮性，CatBoost 生成数据的多个独立随机排列，并对结果进行平均。

### 自动特征组合

CatBoost 的另一个强大功能是它能够自动生成类别特征的组合。例如，它可能会发现 `browser='Chrome'` 和 `device_type='Mobile'` 的组合是一个强大的预测因子。CatBoost 不需要您手动执行这种特征工程，而是在每次树分割时贪婪地组合类别特征，捕捉到否则可能被遗漏的复杂关系。

### 这对您意味着什么

CatBoost 方法的主要优点是极大地简化了数据预处理流程。您通常可以通过在 `cat_features` 参数 (parameter)中指定它们，将具有字符串或对象数据类型的列直接传递给模型。

```python
from catboost import CatBoostClassifier

# 带有类别特征的样本数据
train_data = [['Germany', 1], ['USA', 1], ['Germany', 0]]
train_labels = [1, 1, 0]

# 识别类别特征的索引
categorical_features_indices = [0] 

# 初始化并训练模型
model = CatBoostClassifier(iterations=10, verbose=False)

model.fit(train_data, 
          train_labels, 
          cat_features=categorical_features_indices)

# 进行预测
model.predict(['USA', 0])
```

通过自动化表格数据模型构建中最繁琐和易出错的部分之一，CatBoost 让您能够更多地关注模型调优和评估。这种内置的类别特征处理智能使其有别于其他梯度提升库，并使其成为解决多种分类和回归问题的极其有效工具。

## 参考资料

- [CatBoost: Unbiased Boosting with Categorical Features](https://papers.nips.cc/paper/2017/hash/6449f86052f20309ad6dd1a6d5ce0c16-Abstract.html) — Anna G. Dorogush, Vasily Ershov, and Andrey Gulin (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: NeurIPS; Volume: 30; DOI: [10.5591/978-1-59990-267-3_168](https://doi.org/10.5591/978-1-59990-267-3_168)
  介绍CatBoost算法、有序目标统计量以及自动特征组合的奠基性论文。
- [CatBoost Documentation](https://catboost.ai/docs/) — Yandex (2024)
  Publisher: Yandex
  CatBoost库的官方指南，包含安装、配置和使用方法，并详细解释了其参数和功能。
- [Applied Predictive Modeling](https://doi.org/10.1007/978-1-4614-6849-3) — Max Kuhn, Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  关于数据预处理和特征工程的资源，涵盖了多种分类编码方法、它们的优缺点以及数据泄露的风险。
