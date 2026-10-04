---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "filter-univariate-tests"
sourceId: 1401
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/filter-univariate-tests"
title: "过滤方法：单变量统计检验（ANOVA F值，卡方）"
description: "应用ANOVA F值（回归）和卡方（分类）等统计检验进行特征选择。"
order: 4
plots: ["plots/1401-0.json"]
sourceHash: "c82134be6bf2b66f8e706178000d9ef6e83f2041ae5cb33459d596700d5c768f"
sourceCorrections: []
---

诸如`VarianceThreshold`之类的技术有助于移除变化不大的特征。然而，这类方法不考虑特征与目标变量之间的关系。为了根据特征对目标的预测能力来选择特征，可以使用单变量统计检验。这些检验针对目标变量单独评估每个特征（单变量地），并根据其统计关系的强度给出分数。分数最高的特征被认为更具关联性。

Scikit-learn提供了`SelectKBest`（选择固定数量$k$的最佳特征）和`SelectPercentile`（选择最高百分比的特征）等便捷工具，它们与基于这些统计检验的各种评分函数结合使用。特定检验的选择取决于特征和目标变量的数据类型。

让我们来看两种用于分类任务中特征选择的常见单变量检验。

### 用于分类的ANOVA F值

**使用时机：** 当你有**数值输入特征**和**类别目标变量**时，使用方差分析（ANOVA）F值。

**原理：** ANOVA是一种统计检验，用于检查数值变量的均值在两个或多个组之间是否存在显著差异。在分类特征选择的背景下，每个“组”对应于目标变量中的一个类别。F值（或F统计量）量化 (quantization)了组间（类别间）方差与组内方差的比率。

- **高F值：** 表示数值特征的平均值在目标类别之间存在显著差异。这表明该特征有助于区分这些类别。
- **低F值：** 表明特征的平均值在所有类别中相似，暗示其区分能力较弱。

本质上，我们是在问：“这个数值特征的值是否倾向于根据目标类别而不同？” 如果答案是肯定的（高F值，低p值），该特征可能提供有用信息。

**Scikit-learn实现：**

Scikit-learn的`f_classif`函数计算数值特征与类别目标之间的ANOVA F值。你通常将其与`SelectKBest`等选择器一起使用。

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif

# 生成合成分类数据
# 100个样本，20个特征（10个有用特征，10个冗余特征）
X, y = make_classification(n_samples=100, n_features=20, n_informative=10,
                           n_redundant=5, n_repeated=0, n_classes=2,
                           n_clusters_per_class=2, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])

print("原始特征数量:", X.shape[1])

# 根据ANOVA F值选择前10个特征
# 对数值输入和分类目标使用f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 获取得分和p值
scores = selector.scores_
p_values = selector.pvalues_

# 获取选定的特征名称（需要在DataFrame上拟合选择器）
selected_mask = selector.get_support() # 布尔掩码
selected_features = X.columns[selected_mask]

print("选择的特征数量:", X_selected.shape[1])
print("选定的特征:", selected_features.tolist())
# 示例：显示前5个特征的得分
# print("得分（前5个）:", scores[:5])
# print("P值（前5个）:", p_values[:5])
```

在此示例中，`SelectKBest(f_classif, k=10)`计算了20个特征中每个特征与目标`y`的F统计量，并保留得分最高（且p值最低）的10个特征。



![特征的ANOVA F值示例](plots/1401-0.json)



> 由`f_classif`计算的ANOVA F值示例（对数刻度）。更高的分数表明与目标变量的关系更紧密。`SelectKBest`会选择特征3、5、7、13和19。

### 卡方（$χ^2$）检验

**使用时机：** 当你有**类别输入特征**和**类别目标变量**时，使用卡方（$χ^2$）检验。

**原理：** 卡方检验评估两个类别变量之间的独立性。对于特征选择，它衡量类别特征与类别目标变量之间的依赖性。它将观测频率（每个特征类别在每个目标类别中出现的频率）与期望频率（如果特征和目标是独立的，你所预期的频率）进行比较。

- **高$χ^2$统计量：** 表示观测频率和期望频率之间存在显著差异，表明该特征依赖于（关联于）目标变量。此类特征可能是有用的预测因子。
- **低$χ^2$统计量：** 表明观测频率在独立性假设下接近期望频率，意味着该特征对目标类别提供的信息可能很少。

**重要限制：** 卡方检验要求特征值是**非负的**，因为它通常应用于计数或频率。这意味着你通常在通过独热编码等方法对类别特征进行编码*之后*应用它。

**Scikit-learn实现：**

Scikit-learn的`chi2`函数计算非负特征与类别目标之间的卡方统计量。

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

# 生成合成分类数据（使用之前的X，y）
X, y = make_classification(n_samples=100, n_features=5, n_informative=3,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1, random_state=50)
X = pd.DataFrame(X, columns=[f'num_feat_{i}' for i in range(5)])

# --- 模拟添加类别特征 ---
# 为了演示，我们将数值特征离散化以模拟类别特征
# 并添加一个真正的类别特征。注意：在实际应用中，你会有实际的类别数据。

# 为数值特征创建分箱
binner = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform', subsample=None)
X_binned = pd.DataFrame(binner.fit_transform(X), columns=X.columns, dtype=int)

# 添加一个示例类别特征
np.random.seed(0)
X_binned['cat_feat_A'] = np.random.choice(['P', 'Q', 'R'], size=X.shape[0])
X_binned['cat_feat_B'] = np.random.choice(['X', 'Y'], size=X.shape[0])

print("原始特征（分箱/添加类别特征后）：\n", X_binned.head(3))

# 对分箱/类别特征进行独热编码
# 如果测试数据可能包含未见过的类别，请使用handle_unknown='ignore'
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_binned)
encoded_feature_names = encoder.get_feature_names_out(X_binned.columns)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)

print("\n独热编码后的形状:", X_encoded_df.shape)

# 使用卡方检验选择最佳特征
# 'k'应小于或等于编码特征的数量
k_best_features = 10
selector_chi2 = SelectKBest(score_func=chi2, k=k_best_features)
X_selected_chi2 = selector_chi2.fit_transform(X_encoded_df, y)

# 获取选定特征名称
selected_mask_chi2 = selector_chi2.get_support()
selected_features_chi2 = X_encoded_df.columns[selected_mask_chi2]

print(f"\n使用Chi2选择的{k_best_features}个特征:", selected_features_chi2.tolist())
```

在这种设置中，我们首先将数值特征转换为类似类别的分箱，并添加了明确的类别特征。然后，我们应用独热编码，生成多个二元列。最后，`SelectKBest(chi2, k=10)`计算了这些非负编码特征中每个特征与目标`y`的$χ^2$统计量，并选择了前10个。

### 选择和使用单变量检验

- **数值输入，类别输出：** 使用`f_classif`（ANOVA F值）。
- **类别输入，类别输出：** 使用`chi2`（卡方）。要求非负输入，通常在独热编码后应用。
- **数值输入，数值输出：** 使用`f_regression`（基于简单线性回归计算F统计量）。
- **互信息：** Scikit-learn还提供了`mutual_info_classif`和`mutual_info_regression`。这些函数衡量每个特征与目标之间的互信息。它们是非参数 (parameter)的，可以捕获非线性关系，使它们成为有效的替代方法，尽管计算量可能更大。

单变量检验计算效率高，因为它们独立检查每个特征。然而，这也是它们的主要局限：它们**不考虑特征间的相互作用**。一个特征单独看可能较弱，但与其他特征结合时预测能力很强。因此，尽管对初步评估或快速降维有用，但仅仅依赖单变量检验可能会导致丢弃那些共同作用的潜在有用特征。它们通常最好用作初步步骤，或与更复杂的封装或嵌入 (embedding)方法结合使用。

## 参考资料

- [1.13. Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) — Scikit-learn Developers (2023)
  提供Scikit-learn特征选择工具（如SelectKBest、f_classif和chi2）的官方文档和示例。
- [An Introduction to Statistical Learning: with Applications in Python](https://link.springer.com/book/10.1007/978-3-031-38747-0) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  介绍了统计学习方法，包括特征选择技术及其统计学原理。
- [Feature Selection Methods: A Survey and Review](https://dl.acm.org/doi/10.1145/3134958) — Jundong Li, Kewei Cheng, Suhang Wang, and Kai Zhang (2017)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery (ACM); Volume: 50; Pages: 1-46; DOI: [10.1145/3134958](https://doi.org/10.1145/3134958)
  提供特征选择方法的概述，对过滤、封装和嵌入方法进行分类，并讨论了其应用。
- [Applied Statistics and Probability for Engineers](https://www.wiley.com/en-us/Applied+Statistics+and+Probability+for+Engineers%2C+7th+Edition-p-9781119400350) — Douglas C. Montgomery and George C. Runger (2018)
  Publisher: Wiley
  一本标准教科书，提供扎实的统计学基础，包括假设检验、方差分析和卡方检验的详细说明。
