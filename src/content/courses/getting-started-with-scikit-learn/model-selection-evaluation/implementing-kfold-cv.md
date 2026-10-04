---
course: "getting-started-with-scikit-learn"
chapter: "model-selection-evaluation"
lesson: "implementing-kfold-cv"
sourceId: 1892
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation/implementing-kfold-cv"
title: "实现K折交叉验证"
description: "在Scikit-learn中使用`KFold`和`cross_val_score`进行交叉验证。"
order: 4
plots: []
sourceHash: "3857db1eff7c7cf27fd6612079aae8aa365b235e6d031a0e31d156c2562b652f"
sourceCorrections: []
---

虽然将数据划分为单个训练集和测试集可以提供一种基本的防止过拟合 (overfitting)的检查方法，但由此产生的性能估计可能对*哪些*数据点最终位于训练集或测试集中很敏感。一个特别“幸运”或“不走运”的划分可能会误导性地给出模型真实泛化能力的乐观或悲观评估。交叉验证通过系统地创建多个训练-测试划分并对结果取平均值，提供了一种更可靠的方法。K折交叉验证是最常用方法之一。

### K折交叉验证过程

在K折交叉验证中，原始数据集被划分成$K$个大致等大、互不重叠的子集，称为“折”（folds）。然后，该过程迭代$K$次。在每次迭代$i$（从1到$K$）中：

1. 折叠 $i$ 被保留作为验证集。
2. 剩余的 $K-1$ 个折叠被组合起来形成训练集。
3. 机器学习 (machine learning)模型在此组合训练集上进行训练。
4. 训练好的模型在保留的验证折叠（折叠 $i$）上进行评估。
5. 此迭代的评估分数（例如，准确率、R平方）被记录下来。

在完成所有 $K$ 次迭代后，我们将得到 $K$ 个评估分数。最终的交叉验证分数通常报告为这 $K$ 个分数的平均值。与单个训练-测试划分相比，这个平均值可以提供对模型在未见数据上表现的更稳定、更可靠的估计，因为每个数据点都恰好一次出现在验证集中，并在训练集中出现 $K-1$ 次。 $K$ 的常见选择是5或10。

下面是一个展示5折交叉验证的图示：

> 5折交叉验证流程。数据被分为5个折叠。在每次迭代中，一个折叠作为验证集，其他折叠组成训练集。每次迭代的分数会被平均。

### 使用Scikit-learn实现K折交叉验证

Scikit-learn提供了工具，可以方便地实现K折交叉验证。我们首先看看如何使用`sklearn.model_selection`中的`KFold`类来生成划分的索引，然后我们将查看一个更便捷的辅助函数。

假设你已经有了特征`X`（例如，NumPy数组或Pandas DataFrame）和目标变量`y`（NumPy数组或Pandas Series）。

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression # 示例模型
from sklearn.metrics import accuracy_score # 示例度量
from sklearn.datasets import load_iris # 示例数据集

# 加载示例数据
iris = load_iris()
X, y = iris.data, iris.target

# 1. 初始化 KFold
# 我们选择 K=5 个折叠。
# 建议 shuffle=True 以在划分前随机化数据顺序。
# random_state 确保洗牌的可复现性。
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies = []
model = LogisticRegression(max_iter=1000) # Instantiate the model once

# 2. 遍历 KFold 生成的划分
print(f"正在运行 {k} 折交叉验证...")
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    # 3. 获取此折叠的训练集和验证集
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 4. 在此折叠的训练数据上训练模型
    model.fit(X_train, y_train)

    # 5. 在此折叠的验证数据上评估
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    fold_accuracies.append(accuracy)
    print(f"  折叠 {fold+1}: 验证准确率 = {accuracy:.4f}")

# 6. 计算平均性能和标准差
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\n交叉验证结果 ({k} 折):")
print(f"  各个折叠的准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
print(f"  平均验证准确率: {mean_accuracy:.4f}")
print(f"  准确率标准差: {std_accuracy:.4f}")

# 预期输出:
# 正在运行 5 折交叉验证...
#   折叠 1: 验证准确率 = 1.0000
#   折叠 2: 验证准确率 = 0.9667
#   折叠 3: 验证准确率 = 0.9333
#   折叠 4: 验证准确率 = 0.9333
#   折叠 5: 验证准确率 = 0.9667
#
# 交叉验证结果 (5 折):
#   各个折叠的准确率: ['1.0000', '0.9667', '0.9333', '0.9333', '0.9667']
#   平均验证准确率: 0.9600
#   准确率标准差: 0.0249
```

在这段代码中：

- 我们使用`n_splits=5`、`shuffle=True`和`random_state=42`初始化`KFold`。通常建议进行洗牌，除非你的数据具有需要保留的固有顺序。`random_state`使洗牌可预测，这对于获得可复现的结果很重要。
- `kf.split(X)`方法为每个折叠生成索引数组对（`train_index`，`val_index`）。
- 在循环内部，我们使用这些索引将`X`和`y`切片成当前折叠的相应训练集和验证集。
- 我们训练一个`LogisticRegression`模型，并在验证集上评估其`accuracy_score`。
- 循环结束后，我们计算在5个折叠中获得的准确率的平均值和标准差。平均准确率（这里是0.9600）为我们提供了模型性能的主要估计。标准差（0.0249）表明了性能在不同折叠中的变异性；较低的标准差表示性能更一致。

### 为求方便使用`cross_val_score`

手动遍历折叠具有指导意义，但Scikit-learn提供了`cross_val_score`函数，它可以更简洁地执行整个K折交叉验证过程（划分、训练、评估）。

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 加载示例数据
iris = load_iris()
X, y = iris.data, iris.target

# 实例化模型
model = LogisticRegression(max_iter=1000)

# 定义折叠数量 (K)
k = 5

# 使用 cross_val_score
# cv 可以是整数（用于 KFold 或 StratifiedKFold）或一个 CV 划分器对象
# scoring 指定要使用的度量（例如，'accuracy', 'neg_mean_squared_error'）
scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')

print(f"正在使用 cross_val_score 运行 {k} 折交叉验证...")
print(f"  每个折叠的分数: {scores}")
print(f"  平均准确率: {scores.mean():.4f}")
print(f"  标准差: {scores.std():.4f}")

# 预期输出（如果洗牌没有明确匹配，可能与手动循环略有不同）:
# 正在使用 cross_val_score 运行 5 折交叉验证...
#   每个折叠的分数: [0.96666667 1.         0.93333333 0.96666667 1.        ]
#   平均准确率: 0.9733
#   标准差: 0.0249
```

*(注意：如果`cv`是整数且估计器是分类器，`cross_val_score`默认使用`StratifiedKFold`，我们将在后面讨论它。对于回归任务或当`shuffle=False`时，它使用`KFold`。与手动示例相比，平均准确率的细微差异可能归因于此或一些实现细节。要精确复现带有洗牌的手动`KFold`行为，你可以直接传递`kf`对象：`scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')`)*

`cross_val_score`函数接受估计器（你的模型实例）、特征`X`、目标`y`、交叉验证策略`cv`（这里是整数5，表示使用5个折叠），以及一个`scoring`度量字符串。它返回一个包含每个折叠分数的数组。这比手动循环要紧凑得多。

**关于评分的重要说明：** Scikit-learn的评分函数遵循一个约定，即值越高越好。对于均方误差（MSE）或平均绝对误差（MAE）等值越低越好的度量，Scikit-learn提供了负值版本，如`'neg_mean_squared_error'`或`'neg_mean_absolute_error'`。当使用`cross_val_score`与这些度量时，你会得到负分数，并且“更好”的结果将更接近零（即，负值更小）。

通过使用K折交叉验证，无论是手动实现还是使用`cross_val_score`，你都可以获得对模型在新数据上表现的更可靠的估计，与单个训练-测试划分相比。这是模型评估和比较的基本方法。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本基础教材，涵盖统计学习的理论和实践，包括对交叉验证在模型评估中的详细解释。
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — Scikit-learn developers (2024)
  Scikit-learn官方用户指南中关于交叉验证的部分，详细说明了K-Fold等多种交叉验证策略，以及KFold类和cross_val_score函数的使用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 864
  一本实践指南，展示如何使用Scikit-learn库实现K-Fold交叉验证及其他机器学习技术。
