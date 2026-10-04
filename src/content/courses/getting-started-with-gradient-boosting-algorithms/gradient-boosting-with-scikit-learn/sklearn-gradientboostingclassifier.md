---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "sklearn-gradientboostingclassifier"
sourceId: 7584
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/sklearn-gradientboostingclassifier"
title: "Scikit-Learn的GradientBoostingClassifier"
description: "一份关于在Scikit-Learn中使用GradientBoostingClassifier类构建分类问题模型的指南。介绍主要参数和方法。"
order: 1
plots: []
sourceHash: "346bc29bcdaebeb3afe3a57f91b9ffc1579261960ced214020bae25abd8f1325"
sourceCorrections: []
---

对于分类问题，Scikit-Learn提供了`GradientBoostingClassifier`类，它是梯度提升机的一种实现，适用于预测分类结果。该分类器以一种前向、阶段性的方式构建一个加性模型。在每个阶段，一个回归树会拟合二项或多项偏差损失函数 (loss function)的负梯度。这个过程使得模型能够通过关注那些难以正确分类的观测值，逐步提升其性能。

其工作原理是前一章中提及原则的直接应用。`GradientBoostingClassifier`不是像在简单回归场景中那样拟合树以最小化平方误差，而是最小化一个适合分类的损失函数，例如偏差（也称为对数损失或逻辑损失）。

### 它如何用于分类

对于二元分类，模型会做出一个初始预测，通常是正类的对数几率。然后，对于每个提升阶段：

1. 计算每个观测值的对数损失梯度。这个梯度表示每个样本误差的方向和大小。
2. 训练一个新的弱学习器（一个决策树）来预测这些负梯度。
3. 这个新树的输出会乘以一个学习率，然后加到之前的预测中。

这个顺序过程优化了模型的预测，逐步将预测的对数几率推向正确的方向，以正确分类训练样本。

> 梯度提升中迭代过程的图示。每个新的弱学习器（$H$）被训练来修正前一个模型的误差（梯度），其作用在更新总体预测（$M$）之前会通过学习率（$\nu$）进行调整。

### 重要参数 (parameter)

当你实例化`GradientBoostingClassifier`时，可以配置几个对其行为有较大影响的参数。尽管我们会在后续章节详细介绍参数调优，但从一开始就了解主要参数是很必要的。

- `loss`：要优化的损失函数 (loss function)。默认值为`'log_loss'`，它支持二元和多类别分类。
- `learning_rate`：一个介于0.0和1.0之间的浮点数，用于调整每棵树的效应。较低的学习率需要更多的提升阶段才能达到相同的训练误差水平，但通常能带来更好的泛化能力。这是一种正则化 (regularization)形式。
- `n_estimators`：要执行的提升阶段数。这是集成模型中树的总数。更多的树可能在训练数据上表现更好，但如果数量过高也可能导致过拟合 (overfitting)。
- `max_depth`：单个回归估计器的最大深度。最大深度限制了树中节点的数量，从而控制其复杂程度。较小的深度可以减少方差，并有助于防止过拟合。
- `subsample`：用于拟合单个基学习器的样本比例。如果小于1.0，这将导致随机梯度提升，可以减少方差并提升模型泛化能力，但代价是增加偏差。

### 实际应用

让我们看一个使用`GradientBoostingClassifier`的简单示例。我们将使用Scikit-Learn的`make_classification`生成一个合成数据集，然后训练模型并评估其表现。

```python
# 导入所需库
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 1. 生成合成数据集
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# 2. 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 初始化GradientBoostingClassifier
# 我们将为此示例设置一些参数
gb_clf = GradientBoostingClassifier(
    n_estimators=100,       # 树的数量
    learning_rate=0.1,    # 步长收缩
    max_depth=3,            # 每棵树的最大深度
    subsample=0.8,          # 用于训练每棵树的样本比例
    random_state=42
)

# 4. 在训练数据上拟合模型
gb_clf.fit(X_train, y_train)

# 5. 对测试数据进行预测
y_pred = gb_clf.predict(X_test)

# 6. 评估模型的表现
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# 示例输出：
# Model Accuracy: 0.8900
```

在这段代码中，我们初始化了一个`GradientBoostingClassifier`，包含100棵树（`n_estimators=100`），学习率为0.1，最大树深度为3。我们还通过设置`subsample=0.8`使用了随机梯度提升，这意味着每棵树都在随机80%的训练数据上进行训练。拟合模型后，我们使用它进行预测，并发现它在我们的合成测试集上达到了不错的准确率。

这个类为处理分类任务提供了坚实的基础。下一节将介绍其用于回归问题的对应类，即`GradientBoostingRegressor`。

## 参考资料

- [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) — scikit-learn developers (2024)
  提供 GradientBoostingClassifier 的官方 API 参考、参数说明和基本使用示例。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  这篇开创性论文介绍了梯度提升机算法，阐述了其理论基础和工作机制。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/ESLII_print12.pdf) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本全面教材，详细解释了梯度提升的数学和概念原理，包括其在不同损失函数上的应用以及与统计模型的关联。第10章尤其相关。
