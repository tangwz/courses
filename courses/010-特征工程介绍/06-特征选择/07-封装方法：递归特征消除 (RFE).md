# 封装方法：递归特征消除 (RFE)

来源：[原文](https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/wrapper-rfe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

封装方法通过训练和测试特定的机器学习 (machine learning)模型来评估特征子集。这种方式不同于过滤方法，过滤方法独立于任何模型评估特征。因为封装方法使用模型表现作为评估标准，它们通常能找到更适合特定算法的特征子集，有可能捕捉到过滤方法可能遗漏的特征间关系。然而，这伴随着更高的计算成本，因为需要训练多个模型。

递归特征消除 (RFE) 是一种通过训练和测试特定机器学习模型来评估特征子集的封装方法。这种方法以递归方式移除最不重要的特征，其依据是外部评估器为特征分配的权重 (weight)或重要性得分（例如线性模型的系数或基于树模型的特征重要性）。

### RFE 如何工作

RFE 的核心思路直观明了：

1. **训练模型：** 一个评估器（例如逻辑回归、支持向量 (vector)机或随机森林）在包含所有 $N$ 个特征的初始数据集上被训练。
2. **评估特征重要性：** 每个特征的重要性从已训练的评估器中获取。对于线性模型，这通常是系数的绝对值。对于基于树的模型，它通常是 `feature_importances_` 属性。
3. **消除最不重要的特征：** 重要性得分最低的特征（或一小部分特征）从特征集中移除。
4. **迭代：** 步骤 1-3 被递归地重复，在已缩减的特征集上（现在有 $N-1$ 个特征），直到达到预设的要选择的特征数量。

这个迭代过程逐步修剪特征集，旨在保留那些对模型预测能力贡献最大的特征，根据所选评估器。

> 递归特征消除 (RFE) 的迭代过程。

### 选择评估器

在 RFE 中评估器的选择很重要，因为特征重要性排序完全取决于它。

- **线性模型：** 诸如 `LinearRegression`、`LogisticRegression` 或 `LinearSVC` 等评估器使用系数大小 (`coef_`) 来对特征进行排序。具有较大绝对值的系数的特征被认为更重要。这些模型中的正则化 (regularization)（如 L1 或 L2）会影响系数，从而影响排序。
- **基于树的模型：** 诸如 `DecisionTreeClassifier`、`RandomForestClassifier` 或 `GradientBoostingClassifier` 等评估器使用 `feature_importances_` 属性，该属性通常衡量每个特征对减少森林或集成中所有树的杂质（如基尼不纯度或熵）的贡献程度。

通常，在 RFE 中选用与您最终打算部署的模型相似的评估器是较好的做法，尽管更简单、更快的模型（如线性模型）常用于选择过程本身，以控制计算成本。

### 使用 Scikit-learn 实现

Scikit-learn 提供了通过 `sklearn.feature_selection.RFE` 类实现的便捷 RFE。

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# 生成模拟数据
X, y = make_classification(n_samples=100, n_features=10, n_informative=5,
                           n_redundant=2, n_classes=2, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

# 1. 选择一个评估器
estimator = LogisticRegression(solver='liblinear')

# 2. 初始化 RFE
# 选择最重要的 5 个特征
rfe_selector = RFE(estimator=estimator, n_features_to_select=5, step=1) # step=1 表示每次迭代移除 1 个特征

# 3. 在数据上拟合 RFE
rfe_selector = rfe_selector.fit(X, y)

# 4. 获取结果
selected_features_mask = rfe_selector.support_
feature_ranking = rfe_selector.ranking_

print("Feature Names:", X.columns.tolist())
print("Selected Mask:", selected_features_mask) # 选定特征为 True
print("Feature Ranking:", feature_ranking) # 1 表示已选，数字越大表示越早被移除

# 获取选定特征的名称
selected_feature_names = X.columns[selected_features_mask]
print("\nSelected Features:", selected_feature_names.tolist())

# 转换数据以仅保留选定特征
X_selected = rfe_selector.transform(X)
print("\nShape of original data:", X.shape)
print("Shape of data after RFE:", X_selected.shape)
```

在此示例中：

- 我们使用 `LogisticRegression` 作为基础评估器。
- 我们指定 `n_features_to_select=5`，这告诉 RFE 在只剩下 5 个最优特征时停止。
- `support_` 属性返回一个布尔掩码，表示哪些特征被选中。
- `ranking_` 属性为每个特征赋予一个排名（1 表示选中，2 表示第一个被移除的，依此类推）。
- `transform` 方法可以直接获取只包含选定特征的数据集。

### 使用 RFECV 确定特征数量

RFE 的一个常见问题是不知道要保留多少特征 (`n_features_to_select`)。选择太少可能丢弃有用信息，而选择太多可能保留噪声或冗余特征。

Scikit-learn 提供了 `RFECV` (带交叉验证的 RFE)，它自动确定最优特征数量。它在交叉验证循环中执行 RFE，评估不同数量选定特征的模型表现。

```python
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

# 使用 RFECV 寻找最优特征数量
cv_strategy = StratifiedKFold(n_splits=5) # 对于分类问题使用分层 K 折交叉验证

rfecv_selector = RFECV(estimator=estimator,
                       step=1,
                       cv=cv_strategy,
                       scoring='accuracy', # 要优化的性能指标
                       min_features_to_select=1) # 考虑的最小特征数量

rfecv_selector = rfecv_selector.fit(X, y)

print(f"\nOptimal number of features found by RFECV: {rfecv_selector.n_features_}")
print("Selected Features by RFECV:", X.columns[rfecv_selector.support_].tolist())

# 您可以选择绘制性能与特征数量的关系图
# grid_scores_ 属性（或新版本中的 cv_results_['mean_test_score']）
# 包含测试的每个特征数量的得分。
# （为简洁起见，省略了绘图代码）
```

`RFECV` 使用指定的交叉验证策略 (`cv`) 和性能指标 (`scoring`) 来寻找在各折中产生最佳平均得分的特征数量。这通常是选择特征子集大小的一种更数据驱动的方式，相比于手动设置 `n_features_to_select`。

### RFE 的优缺点

**优点：**

- **以模型为中心：** 通过所选评估器的表现，隐含地考虑特征关系。
- **潜在的更优子集：** 与过滤方法相比，通常能为所用的特定模型类型找到更具预测性的特征子集。

**缺点：**

- **计算密集：** 需要多次训练基础评估器（每移除一个特征训练一次，如果使用 `RFECV`，还可能乘以交叉验证的折数）。这对于大型数据集或复杂模型可能很慢。
- **依赖于评估器：** 结果很大程度上取决于基础评估器的选择。不同的评估器可能导致不同的特征排名和选择。
- **信息丢失：** 通过完全移除特征，可能丢弃有用的信息，特别是当特征之间存在基础评估器重要性指标未能很好捕捉的复杂非线性关系时。

RFE 是一种有价值的工具，当您怀疑特征间关系相关且计算成本可控时。使用 `RFECV` 提供了一种更自动化且通常更可靠的应用 RFE 的方式，通过集成交叉验证来确定要保留的特征数量。

## 参考资料

- [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/feature_selection.html#rfe) — scikit-learn developers (2023)
  RFE和RFECV类的官方文档，包含实际实现细节和参数。
- [Gene selection for cancer classification using support vector machines](https://link.springer.com/article/10.1023/A:1012487302797) — Isabelle Guyon, Jason Weston, Stephen Barnhill, and Ralph G. G. T. P. Jonsson (2002)
  Journal: Machine Learning; Volume: 46; Pages: 389-422; DOI: [10.1023/A:1012487302797](https://doi.org/10.1023/A%3A1012487302797)
  引入支持向量机递归特征消除的原始论文，提供了该方法的理论基础。
- [Applied Predictive Modeling](https://www.springer.com/gp/book/9781461468486) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer
  一本关于预测建模各个方面的综合教科书，其中包含专门的特征选择方法章节，包括RFE等包装方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  统计学习的经典参考文献，讨论了特征选择技术、模型构建及其理论基础。

---

[上一节](06-%E5%B0%81%E8%A3%85%E5%99%A8%E6%96%B9%E6%B3%95%E6%A6%82%E8%BF%B0.md) · [下一节](08-%E5%B0%81%E8%A3%85%E6%96%B9%E6%B3%95%EF%BC%9A%E5%BA%8F%E5%88%97%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9%20%28SFS%29.md)
