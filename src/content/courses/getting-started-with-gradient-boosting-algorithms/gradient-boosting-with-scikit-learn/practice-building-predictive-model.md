---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "gradient-boosting-with-scikit-learn"
lesson: "practice-building-predictive-model"
sourceId: 7590
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-3-gradient-boosting-with-scikit-learn/practice-building-predictive-model"
title: "实战演练：构建预测模型"
description: "在动手实验中运用所学知识。使用Scikit-Learn训练、评估并解释数据集上的梯度提升模型。"
order: 7
plots: ["plots/7590-0.json"]
sourceHash: "85fcab366043569e165a6e648a3896791b1cef53fcfd1ff7aea7535a1c81162a"
sourceCorrections: []
---

此动手实验聚焦于梯度提升的实践应用，通过构建、评估并解释一个用于分类问题的梯度提升模型。我们将使用Scikit-Learn的`GradientBoostingClassifier`在一个知名数据集上，亲身实践完整的工作流程。

### 数据集：葡萄酒识别

本次练习中，我们将使用葡萄酒识别数据集，该数据集在Scikit-Learn中即可方便获取。任务是根据13种化学属性预测葡萄酒的类别（三种栽培品种之一）。该数据集非常适合我们的目的，因为它只包含数值特征，并具有明确的分类目标。

### 步骤1：载入和准备数据

第一步是导入所需库并载入数据集。我们将使用pandas在DataFrame中管理数据，这使得检查和处理数据变得更容易。我们还将把数据分成训练集和测试集，以确保能够评估模型在未见过数据上的表现。

```python
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# 载入数据集
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

# 显示特征（X）的前几行
print("特征 (X):")
print(X.head())

# 显示目标变量（y）的分布
print("\n目标 (y) 值计数:")
print(y.value_counts())

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
```

`train_test_split`中的`stratify=y`参数 (parameter)确保了训练集和测试集中每种葡萄酒类别的比例与原始数据集中的比例一致。这对于分类问题而言是一种好做法，可以避免不平衡数据对模型训练和评估造成影响。

### 步骤2：训练梯度提升分类器

数据准备就绪后，我们现在可以实例化并训练`GradientBoostingClassifier`。我们将从一组基本参数 (parameter)开始。`n_estimators=100`表示我们将构建100个序列树。0.1的`learning_rate`控制每棵树的贡献，`max_depth=3`限制了每棵单独树的复杂度以避免过拟合 (overfitting)。`random_state`则确保了结果的可复现性。

```python
# 初始化梯度提升分类器
gbm = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# 将模型拟合到训练数据
gbm.fit(X_train, y_train)
```

`.fit()`方法启动训练过程。模型会按顺序添加树，每棵新树都尝试纠正由先前树组成的集成模型所犯的错误。

### 步骤3：评估模型表现

训练完成后，我们使用模型从未见过的测试集来评估其预测能力。我们使用`.predict()`方法来获取测试集中每个样本的预测类别。

```python
# 在测试集上进行预测
y_pred = gbm.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}\n")

# 显示详细的分类报告
print("分类报告:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))
```

输出结果将显示较高的准确率，以及一份包含每种葡萄酒类别对应的精确率、召回率和F1分数的分类报告。这使得我们能够全面了解模型在所有类别中正确分类样本的表现。

### 步骤4：解释模型

好的预测很有用，但了解模型做出特定预测的*原因*通常更有价值。现在，我们将通过查看特征重要性并可视化部分依赖性来解释已训练的模型。

#### 特征重要性

已训练的GBM模型的`feature_importances_`属性为每个特征提供了一个分数，说明其在构建集成模型中的决策树时有多大作用。分数越高，表示该特征被更频繁且有效地用于进行分裂，从而提高模型的纯度。

我们来可视化重要性，以便快速找出最具影响力的特征。

```python
import pandas as pd

# 获取特征重要性
importances = gbm.feature_importances_
feature_names = X.columns

# 创建DataFrame以便更轻松地可视化
feature_importance_df = pd.DataFrame(
    {'feature': feature_names, 'importance': importances}
).sort_values('importance', ascending=False)

print("前10位特征重要性:")
print(feature_importance_df.head(10))
```

为了让这些信息更易于查看，我们可以创建一个条形图。



![葡萄酒分类中的特征重要性](plots/7590-0.json)



> 根据模型，对于葡萄酒品种分类而言，最主要的特征。像`proline`和`flavanoids`这样的特征对模型的预测有很大作用。

该图清晰显示，`proline`、`flavanoids`和`color_intensity`是模型预测的三个主要驱动因素。其余特征的作用则小得多。

#### 部分依赖图

特征重要性告诉我们*哪些*特征是主要的，而部分依赖图（PDPs）则告诉我们一个特征*如何*影响模型的预测。PDP隔离了一个特征的作用，同时对所有其他特征的作用进行平均。

我们来为两个最主要的特征：`proline`和`flavanoids`创建PDP。我们将使用Scikit-Learn的`PartialDependenceDisplay`工具。这些图的y轴显示的是每个类别的预测的“中心化”对数几率。

```python
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

# 为前两个主要特征创建部分依赖图
fig, ax = plt.subplots(figsize=(12, 5))
display = PartialDependenceDisplay.from_estimator(
    gbm,
    X_train,
    features=['proline', 'flavanoids'],
    feature_names=X.columns.tolist(),
    target_names=wine.target_names,
    ax=ax
)
plt.suptitle("主要特征对葡萄酒类别预测的部分依赖性")
plt.show()
```

通过查看生成的图，可以推断出关系。例如，`proline`的PDP可能会显示，随着`proline`值的增加，葡萄酒属于'class\_0'的概率也大幅增加，而其他类别的概率则减少。类似地，`flavanoids`的图可能显示，在某个点之前，它与'class\_0'和'class\_1'呈正相关，之后效果趋于平稳或改变。这些可视化提供了每个特征对模型输出的边际效应的直接佐证。

本次实践练习带你体验了使用Scikit-Learn的梯度提升实现进行机器学习 (machine learning)项目的完整生命周期：从数据准备到模型训练、评估，最后是解释。你现在已具备扎实的根基，可以将这个强大的算法应用到你自己的数据集上，并为后续章节中更高级、更注重性能的库的学习做好了充分准备。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://www.stat.rice.edu/~riedi/552/Friedman.pdf) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机算法的原始学术论文。
- [1.11. Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) — scikit-learn developers (2023)
  scikit-learn 梯度提升实现（包括 `GradientBoostingClassifier` 及相关工具）的官方文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的实用指南，涵盖使用 scikit-learn 的机器学习工作流，包括梯度提升和模型评估。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本关于解释机器学习模型的综合在线书籍，详细介绍了特征重要性和偏依赖图。
