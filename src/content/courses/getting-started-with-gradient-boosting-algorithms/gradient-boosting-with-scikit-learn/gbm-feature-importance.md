---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "gbm-feature-importance"
sourceId: 7588
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/gbm-feature-importance"
title: "GBM中的特征贡献度"
description: "学习如何获取和解释训练好的梯度提升模型的特征贡献度分数，以明白哪些特征推动了预测。"
order: 5
plots: ["plots/7588-0.json"]
sourceHash: "3ebb6e5d8a1033aa5aa61ca7e22452c5f8697d374f7f0f43c94909ab0f0add40"
sourceCorrections: []
---

模型训练完成后，很自然地会问它学到了什么。哪些特征对其预测影响最大？梯度提升模型，尽管是许多决策树的集成，但并非完全的黑箱。我们可以对其进行审视，以明白哪些特征是其决策的最主要因素。此过程有助于验证模型，传递其结果，甚至指导未来的特征工程工作。

### 梯度提升模型如何衡量贡献度

特征贡献度在梯度提升等基于树的集成模型中，通常根据特征对减少模型损失或不纯度的贡献程度来计算。Scikit-Learn的实现采用了一种称为“平均不纯度下降”或基尼贡献度的方法。

以下是其高层次工作原理：

1. **节点不纯度：** 每当构建一棵树时，它会在不同的节点处分割数据。每次分割的选择都是为了使从父节点到两个子节点的不纯度下降最大化（例如，回归问题的方差，分类问题的基尼不纯度）。
2. **贡献度分数：** 通过特定特征分割所实现的不纯度减少量，即为该特征在该次分割中的“贡献度”。
3. **总贡献度：** 该特征的总贡献度是该特征在所有使用它的分割中实现的这些减少量的总和，并聚合了集成模型中所有树的结果。
4. **归一化 (normalization)：** 最后，这些总贡献度分数通常会进行归一化处理，使其总和为1.0，以便于比较。分数为0.30的特征表示其对所有树中总不纯度减少量的贡献率为30%。

分数越高，表示该特征越常用于进行有效的分割，因此模型对其的依赖程度也越大。

### 使用Scikit-Learn获取特征贡献度

在拟合 `GradientBoostingClassifier` 或 `GradientBoostingRegressor` 后，您可以通过 `feature_importances_` 属性获取特征贡献度分数。此属性返回一个NumPy数组，其中每个元素对应于特征在训练数据中出现的顺序的贡献度。

我们来看一个实际的例子。我们将在一个合成数据集上训练一个 `GradientBoostingRegressor`，然后检查特征贡献度。

```python
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# 生成一个合成数据集
# 我们使第三个特征（索引2）信息量最大
X, y = make_regression(n_samples=1000, n_features=10, n_informative=3, random_state=42)

# 初始化并训练模型
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X, y)

# 获取特征贡献度
importances = gbr.feature_importances_

# 为清晰起见创建特征名称列表
feature_names = [f'Feature {i}' for i in range(X.shape[1])]

# 打印特征贡献度
for name, score in zip(feature_names, importances):
    print(f"{name}: {score:.4f}")
```

运行此代码将产生一个输出，显示10个特征中每个特征的归一化 (normalization)贡献度分数。您可能会看到，在数据生成时指定的信息量大的特征比其他特征具有明显更高的分数。

### 可视化特征贡献度

一长串数字可能难以理解。水平条形图是一种有效的方式，可以直观地显示和比较每个特征的相对贡献度。这使得模型认为哪些特征最具预测性变得一目了然。

我们可以使用上一个示例中的 `importances` 和 `feature_names` 来创建图表。下面的图表显示了一个典型输出。



![梯度提升模型特征贡献度](plots/7588-0.json)



> 特征根据其对模型预测能力的贡献进行排名。在这个例子中，特征2显然是最具影响力的。

### 主要考量和局限性

尽管特征贡献度是一种有用的工具，但了解其局限性也很必要。

- **偏向高基数特征：** 基于树的模型有时会给连续特征或具有许多独特值的分类特征赋予更高的贡献度。这是因为这些特征提供了更多的潜在分割点，增加了找到能减少不纯度的分割的机会。
- **相关特征：** 如果两个或更多特征高度相关，它们可能会相互替代。当模型构建其树时，它可能会选择其中一个进行分割，其贡献度分数会增加。其他相关特征可能不会被那么频繁地选择，从而导致较低的贡献度分数。这些特征所承载的潜在信号的总贡献度会分散在它们之间，可能使您误以为该信号比实际更弱。
- **贡献度不代表方向：** 特征贡献度告诉您一个特征 *有多大作用*，但不会告诉您它 *如何* 作用。例如，“房屋年龄”的高贡献度分数并不能说明老旧房屋的价格更高还是更低。它只说明“房屋年龄”是一个强的预测因素。为了明白特征与预测之间的关联，我们需要其他工具，例如我们将在下一节中讨论的局部依赖图。

## 参考资料

- [Gradient Tree Boosting (Scikit-learn documentation)](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting) — scikit-learn developers (2023)
  官方文档，解释了 Scikit-learn 中梯度提升模型的实现以及如何使用 `feature_importances_` 属性获取特征重要性分数。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-4/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  引入梯度提升机 (GBM) 算法的基础性论文，为其构建和性能提供了理论基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本全面的统计学习教材，包含对决策树、集成方法以及从杂质减少中得出特征重要性概念的详细讨论。
