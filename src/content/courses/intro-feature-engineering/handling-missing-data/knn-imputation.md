---
course: "intro-feature-engineering"
chapter: "handling-missing-data"
lesson: "knn-imputation"
sourceId: 1316
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-2-handling-missing-data/knn-imputation"
title: "多变量填充：KNN填充器"
description: "使用K近邻算法，根据相似的数据点进行缺失值填充。"
order: 5
plots: []
sourceHash: "7bf8b88a204c8af3f71f64f163cd827ea365a85751b1ff275570f0ddfc64cf3f"
sourceCorrections: []
---

虽然像用均值、中位数或众数替换缺失值这样的简单填充方法快速且易于实现，但它们有一个重要的局限：它们忽略了特征之间的关联。如果一个特征的值与其他特征相关，那么使用简单的统计量可能会导致不理想或有偏差的结果。多变量填充技术通过在估计缺失数据时考虑其他特征的值来解决这个问题。

K近邻（KNN）填充器是一种常用的多变量方法。它不是仅仅使用目标列的简单统计量，而是查看*整个特征集*，以找到与包含缺失值的数据点（样本或行）最相似的数据点。然后，根据这些邻近点中观察到的值来估计缺失值。

### KNN填充的思路

想象你有一个房屋数据集，包含建筑面积、卧室数量、建造年份和销售价格等特征。假设其中一栋房屋的“建造年份”缺失。简单的填充可能会用所有房屋的平均“建造年份”来填充。然而，KNN填充会查找数据集中在建筑面积、卧室数量和销售价格方面相似的其他房屋（该行中*没有*缺失值的特征）。它识别出“k”个最相似的房屋（最近邻），然后使用它们的“建造年份”值（例如，通过取平均值）来估计目标房屋的缺失值。

核心思想基于以下假设：数据点在特征空间中可能与其邻居相似。

### KNN填充器的工作原理

1. **距离计算：** 对于特征$X_i$中包含缺失值的样本，填充器计算该样本与所有其他样本之间的距离。此计算仅使用两个被比较样本中*存在*的特征。一个常见的距离度量是欧几里得距离，它适用于处理缺失值（在实际应用中常被称为`nan_euclidean_distances`）。
2. **识别邻居：** 它识别出与包含缺失值的样本距离最小的$k$个样本（行）。这些就是“k”个最近邻。
3. **填充：** 特征$X_i$中的缺失值使用来自$k$个邻居的特征$X_i$的值进行填充。如果`weights='uniform'`，填充通常是邻居值的平均值（针对数值特征）。如果`weights='distance'`，更近的邻居在计算中拥有更大的影响，这意味着填充值是加权平均值，其中权重 (weight)与距离成反比。

### 使用Scikit-learn实现

Scikit-learn在其`impute`模块中提供了一个方便的`KNNImputer`类。

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler # KNNImputer通常需要先进行此操作

# 包含缺失值的示例数据
data = {'FeatureA': [1, 2, np.nan, 4, 5, 6, 7, 9, 10],
        'FeatureB': [2, 4, 6, 8, 10, 11, 12, np.nan, 20],
        'FeatureC': [5, 10, 15, 20, 25, 30, 35, 40, 45]}
df = pd.DataFrame(data)

print("原始DataFrame:")
print(df)

# 重要提示：KNNImputer对特征缩放很敏感
# 在应用KNNImputer之前对特征进行缩放
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 初始化KNNImputer
# n_neighbors: 要使用的邻居数量 (k)
# weights: 'uniform' 或 'distance'
imputer = KNNImputer(n_neighbors=3, weights='uniform')

# 对缩放后的数据进行拟合和转换
df_imputed_scaled = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df.columns)

# 逆转换以恢复原始比例的数据
df_imputed = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=df.columns)

print("\nKNN填充后的DataFrame:")
print(df_imputed)

# 验证填充值（例如，FeatureA，第2行）
print(f"\n索引2处FeatureA的填充值: {df_imputed.loc[2, 'FeatureA']:.2f}")
# 验证填充值（例如，FeatureB，第7行）
print(f"\n索引7处FeatureB的填充值: {df_imputed.loc[7, 'FeatureB']:.2f}")
```

在这个例子中，`n_neighbors=3`表示该算法会查找3个最近的邻居来填充缺失值。

### 参数 (parameter)

- `n_neighbors`: 用于填充的邻居样本数量 ($k$)。较小的$k$使填充对局部模式更敏感，但也可能更容易受到噪声影响。较大的$k$提供更平滑的估计，但可能掩盖局部变化。
- `weights`: 决定邻居值如何参与计算。
  - `'uniform'`: 所有$k$个邻居贡献相同（简单平均）。
  - `'distance'`: 更近的邻居有更大的影响。贡献按距离的倒数加权。
- `metric`: 用于查找邻居的距离度量。默认的`nan_euclidean`在距离计算期间适当地处理缺失值。

### KNN填充器的优点

- **考虑特征关联：** 与简单方法不同，它使用来自其他特征的信息，这可能带来更准确的填充，特别是当特征之间存在关联时。
- **非参数 (parameter)：** 相比于像`IterativeImputer`这样的基于模型的填充方法，它对数据分布的假设更少。
- **适用性广：** 如果使用适当的距离度量，理论上可以处理不同类型的数据（尽管标准实现最适合数值数据）。

### 缺点与注意事项

- **计算成本：** 查找最近邻可能计算量大，特别是对于大型数据集（许多行或许多特征）。复杂度随样本数量的增加而显著增长。
- **对缩放的敏感性：** 因为它是基于距离的，KNN填充器对特征的比例非常敏感。范围较大的特征可能会主导距离计算。**在应用`KNNImputer`*之前*，几乎总是需要对数值特征进行缩放（例如，使用`MinMaxScaler`或`StandardScaler`）。** 请记住，仅在训练数据上拟合缩放器，并转换训练集和测试集。
- **$k$的选择：** 性能取决于$k$的选择。这通常需要调整，可能通过在后续任务上使用交叉验证来完成。
- **维度灾难：** 在非常高维的空间中，“最近邻”的含义会减弱，因为距离往往变得更加一致。当特征数量相对于样本数量变得非常大时，其有效性可能会降低。
- **类别数据：** 标准`KNNImputer`需要数值输入。类别特征在应用填充器之前需要进行适当的编码（例如，独热编码）。然而，独热编码会显著增加维度，可能影响性能。

与简单策略相比，KNN填充器通过运用特征间的关联，提供了一种更精细的处理缺失数据的方法。然而，它的计算成本以及对缩放和$k$选择的敏感性意味着应谨慎使用，特别是在确保特征经过适当预处理（缩放）之后。当简单方法不足时，它通常能在填充质量和复杂性之间提供良好的平衡。

## 参考资料

- [sklearn.impute.KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html) — scikit-learn developers (2023)
  scikit-learn中KNNImputer类的官方文档，提供了详细的用法、参数和示例。
- [Missing value estimation methods for DNA microarray gene expression data](https://academic.oup.com/bioinformatics/article/17/6/520/215758) — Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor Hastie, Robert Tibshirani, David Botstein, Russ B. Altman (2001)
  Journal: Bioinformatics; Publisher: Oxford University Press; Volume: 17; Pages: 520-525; DOI: [10.1093/bioinformatics/17.6.520](https://doi.org/10.1093/bioinformatics/17.6.520)
  一篇经典论文，介绍了基于KNN的缺失数据估算方法，并在高维生物数据集中对其有效性进行了评估。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  一本广泛使用的实用指南，涵盖了数据预处理技术，包括KNN Imputer等多种插补方法，并提供了Scikit-learn的实际示例。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn, Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本涵盖数据预处理、特征工程和预测建模的教科书，讨论了各种插补策略及其注意事项。
