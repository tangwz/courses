---
course: "getting-started-with-scikit-learn"
chapter: "model-selection-evaluation"
lesson: "stratified-kfold-classification"
sourceId: 1893
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation/stratified-kfold-classification"
title: "分层K折用于分类"
description: "使用分层K折交叉验证，在分类任务中保持类别分布。"
order: 5
plots: ["plots/1893-0.json"]
sourceHash: "93536f6677c8bbeab14cf31854fe2a21eb6e05a2da96ab8c8ececb0e4b03fe59"
sourceCorrections: []
---

K折交叉验证能够比单次训练-测试划分提供更准确的模型表现估计。然而，标准K折交叉验证是随机地将数据分配到各个折叠中。尽管这种方法在许多情况下运行良好，但有时可能导致有问题的划分，特别是在处理不平衡数据集的分类任务时。

假设一个二分类问题，其中90%的样本属于A类，而只有10%属于B类。如果你使用标准K折交叉验证，例如分成10折，纯粹由于偶然，一些折叠可能只包含极少甚至没有少数类（B类）的实例。在缺少B类样本的折叠上训练模型，意味着模型将无法识别该类别。同样，在测试折叠上评估时，如果该折碰巧B类样本数量过高或过低，可能会得到不可靠的性能得分。这种折叠之间类别分布的差异会导致评估结果不稳定和有误导性。

为了解决这个问题，我们使用**分层K折交叉验证**。分层可以确保每个折叠中的类别分布与整个数据集的分布大致相同。如果你的数据集有90%的A类和10%的B类，分层K折交叉验证会确保生成的每个$k$折都保持这种90/10的划分比例（在样本数量允许的情况下尽可能接近）。

### 分层K折的工作方式

不同于纯粹的随机分配，分层K折交叉验证首先根据类别标签对数据进行分组。然后，它按比例从每个类别中抽取实例来创建折叠。这确保了在交叉验证过程的每次迭代中生成的训练和验证子集中，类别代表性都得以保持。

考虑一个小型不平衡数据集的比较：



![折叠中的类别分布（10个样本：9个A类，1个B类）](plots/1893-0.json)



> 在10个样本（9个A类，1个B类）的2折划分中，标准K折交叉验证可能会随机地将唯一的B类样本完全放在第2折中。分层K折交叉验证确保每个划分都从每个类别中接收到按比例数量的样本（尽可能接近）。

### 在Scikit-learn中实现分层K折

Scikit-learn使得使用分层K折交叉验证变得简单明了。你可以明确使用`StratifiedKFold`交叉验证器，或者依靠像`cross_val_score`这样的函数，这些函数通常在分类任务中自动使用分层。

让我们看看如何明确使用`StratifiedKFold`，然后将其与`cross_val_score`结合使用。

首先，我们需要一些示例数据和一个分类器：

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# 创建一个合成的不平衡数据集
X, y = make_classification(n_samples=100, n_features=20, n_informative=2,
                           n_redundant=10, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

print(f"数据集形状: {X.shape}")
print(f"类别分布: {np.bincount(y)}")

# 定义分类器
classifier = LogisticRegression(solver='liblinear', random_state=42)
```

现在，让我们明确创建一个`StratifiedKFold`对象，并查看它是如何生成索引的：

```python
# 初始化分层K折交叉验证
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 遍历划分并显示索引（可选：仅作说明）
print(f"\n生成 {n_splits} 个分层折叠:")
fold_num = 1
for train_index, test_index in skf.split(X, y):
    print(f"折叠 {fold_num}:")
    # print(f"  训练索引: {train_index[:10]}...") # Optional: 查看索引
    # print(f"  测试索引:  {test_index}")       # Optional: 查看索引
    print(f"  训练集大小: {len(train_index)}, 测试集大小: {len(test_index)}")
    # 验证此折叠中测试集的类别分布
    y_test_fold = y[test_index]
    print(f"  测试折叠类别分布: {np.bincount(y_test_fold)}")
    fold_num += 1
```

你会注意到，每个测试折叠都保持着与原始数据集大致相同的类别比例（大约90%和10%）。

使用分层K折交叉验证最常见的方法是将其（或者仅仅是划分的数量，让Scikit-learn选择分层）传递给`cross_val_score`：

```python
# 方法1：直接传递StratifiedKFold对象
scores_explicit_skf = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
print(f"\n使用明确的分层K折交叉验证的准确率得分 ({n_splits}折): {scores_explicit_skf}")
print(f"平均准确率: {scores_explicit_skf.mean():.4f} (+/- {scores_explicit_skf.std() * 2:.4f})")

# 方法2：传递一个整数 - cross_val_score对于分类器通常默认使用分层K折交叉验证
# 注意：为了在使用默认设置时保持结果可复现，请确保估计器（如果适用）设置了random_state，
# 并请注意，如果未通过cv对象明确控制，内部洗牌可能有所不同。
scores_implicit_skf = cross_val_score(classifier, X, y, cv=n_splits, scoring='accuracy')
print(f"\n使用隐式分层的准确率得分 ({n_splits}折): {scores_implicit_skf}")
print(f"平均准确率: {scores_implicit_skf.mean():.4f} (+/- {scores_implicit_skf.std() * 2:.4f})")
```

两种方法都达到相同的目的：使用交叉验证评估分类器，其中每个折叠都保持了整体的类别平衡。

**何时使用分层K折交叉验证：**

- **分类任务：** 对于分类问题，它通常是建议的默认选项。
- **不平衡数据集：** 在处理一个类别样本数量显著多于其他类别的数据集时，它尤其重要。
- **小型数据集：** 分层有助于确保稀有类别在所有折叠中都得到代表，这在数据有限时非常关键。

对于回归问题，标准K折交叉验证通常就足够了，因为“类别平衡”的理念不直接适用。然而，对于分类问题，使用分层K折交叉验证可以为你的模型在未见过数据上的表现提供更可靠、更稳定的估计，有助于你在模型选择和超参数 (parameter) (hyperparameter)调整时做出更好的决定。

## 参考资料

- [sklearn.model_selection.StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) — scikit-learn developers (2023)
  Scikit-learn官方文档，详细介绍了分层K折交叉验证器的参数和使用方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine/9781098125969/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用指南，涵盖了交叉验证技术，包括分层K折，并提供了Scikit-learn的使用示例。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  一本基础教材，解释了交叉验证和模型评估的原理，有助于理解分层K折的理论基础。
