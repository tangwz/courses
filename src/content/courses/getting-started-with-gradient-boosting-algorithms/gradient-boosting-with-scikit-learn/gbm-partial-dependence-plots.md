---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "gbm-partial-dependence-plots"
sourceId: 7589
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/gbm-partial-dependence-plots"
title: "偏依赖图用于模型解释"
description: "通过偏依赖图（PDPs）可视化特征与模型预测之间的关系，以提升模型可解释性。"
order: 6
plots: ["plots/7589-0.json", "plots/7589-1.json"]
sourceHash: "1dd1347ce6649a2f4812a3f7e75f64b48204712b2cb0e952f2614a06fab25995"
sourceCorrections: []
---

特征重要性得分提供了梯度提升机（GBM）依赖哪些特征的概要视图。然而，这些得分只按预测能力对特征进行排名，未能解释特征与模型预测之间关系的本质。例如，特征值增加会导致更高的预测值还是更低的预测值？这种关系是线性的，还是更为复杂的？偏依赖图（PDPs）用于回答这些具体问题。

### “是什么”背后的“怎么样”

偏依赖图显示了一两个特征对模型预测结果的边际效应。简单来说，它显示了当你改变一个特征的值而保持所有其他特征不变时，模型预测值平均如何变化。这使得你能够分离并可视化模型所学到的特定输入与其输出之间的关系。

计算方法是通过平均化数据集中所有其他特征的影响。对于选定的特征，过程如下：

1. 为目标特征创建一系列值（网格）。
2. 对于网格中的每个值，将其替换为数据集中每个样本中该特征的值。
3. 模型对每个修改后的样本进行预测。
4. 对所有样本的预测值进行平均。
5. 将这个平均预测值与网格中的特征值进行绘制。

结果是一条线或一个曲面，它将预期预测值可视化为目标特征的函数。

### 使用Scikit-Learn生成偏依赖图

Scikit-Learn在`sklearn.inspection`模块中提供了一个方便且强大的工具来创建这些图表。`PartialDependenceDisplay`类及其`from_estimator`方法处理整个计算和绘图过程。

我们首先在加州住房数据集上训练一个`GradientBoostingRegressor`。

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# 加载并准备数据
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    housing.data, housing.target, test_size=0.2, random_state=42
)
feature_names = housing.feature_names

# 训练一个GBM模型
gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gbm.fit(X_train, y_train)
```

有了训练好的模型，为像中位数收入（`MedInc`）这样的单个特征创建偏依赖图直接简单。

```python
# 为'MedInc'特征创建并显示偏依赖图
fig, ax = plt.subplots(figsize=(8, 6))
PartialDependenceDisplay.from_estimator(
    gbm,
    X_train,
    features=['MedInc'], # 或使用索引: features=[0]
    feature_names=feature_names,
    ax=ax
)
plt.show()
```

### 解释单特征偏依赖图

上面的代码生成一个单特征偏依赖图，它显示了一个特征与模型预测之间的关系。`MedInc`的图表看起来像这样。



![房价对中位数收入的偏依赖](plots/7589-0.json)



> 随着中位数收入的增加，平均预测房价稳步上升，但对于非常高的收入，其效应开始趋于平稳。

从这张图表中，我们可以得出明确的结论：我们的GBM模型学到了中位数收入与房价之间存在正向、非线性关系。预测值在收入达到约9之前急剧上升，之后额外收入的边际效益减弱。这比简单地说明`MedInc`是主要的特征，提供了更为详细的解释。

### 使用双特征偏依赖图可视化特征交互

偏依赖不限于单个特征。通过同时绘制两个特征，我们可以可视化它们对预测的交互作用。这有助于回答诸如“特征A的效应是否依赖于特征B的值？”这样的问题。

我们来查看中位数收入（`MedInc`）与平均房间数（`AveRooms`）之间的交互。

```python
# 创建并显示双特征偏依赖图
fig, ax = plt.subplots(figsize=(9, 7))
PartialDependenceDisplay.from_estimator(
    gbm,
    X_train,
    features=['MedInc', 'AveRooms'],
    feature_names=feature_names,
    ax=ax
)
plt.show()
```

这将生成一个2D热力图，其中颜色代表平均预测值。



![中位数收入与平均房间数的交互作用](plots/7589-1.json)



> 最高的预测房价（最深的红色）出现在中位数收入和平均房间数都较高时。

热力图显示，最高的预测房价出现在右上角，即`MedInc`和`AveRooms`都较大时。它还表明，当平均收入已经很高时，房间数量的增加对价格有更强的正向作用。这种交互作用比单独分析每个特征，提供了更全面的模型逻辑情况。

### 主要考量

尽管偏依赖图功能强大，但它基于一个核心假设：你所绘制的特征与模型中的其他特征不相关。该方法通过改变一个特征的值，同时保持其他特征不变来运作。如果两个特征高度相关（例如，年龄和经验年限），这个过程可能会创建非常不现实甚至不可能的数据点，从而可能导致具有误导性的图表。

此外，偏依赖图显示的是整个数据集的*平均*效应。它有时会掩盖更复杂的关系，即某个特征以不同方式影响数据的不同子集。对于需要单个样本层面解释的场景，可以使用更高级的技术，例如个体条件期望（ICE）图。

尽管存在这些局限性，偏依赖图仍然是解释梯度提升机（GBM）的不可或缺的工具。它们弥合了知道*哪些*特征是主要的，以及理解它们*如何*驱动模型预测之间的差距，将一个复杂的模型变得更加透明和易于理解。

## 参考资料

- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/pdp.html) — Christoph Molnar (2024)
  Pages: 19 Partial Dependence Plot (PDP)
  提供了对部分依赖图的全面解释，包括其计算、解释、局限性以及与个体条件期望（ICE）图的关系。
- [sklearn.inspection.PartialDependenceDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.PartialDependenceDisplay.html) — scikit-learn developers (2023)
  Scikit-learn 中生成部分依赖图的官方文档，详细说明了其与 `from_estimator` 的用法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: Section 10.13.2; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础教材，介绍了部分依赖图作为模型解释技术，并为梯度提升机提供了理论基础。
