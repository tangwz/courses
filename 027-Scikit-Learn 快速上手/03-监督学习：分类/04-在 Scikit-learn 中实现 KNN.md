# 在 Scikit-learn 中实现 KNN

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/implementing-knn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

K-Nearest Neighbors (KNN) 算法根据数据点最近邻居中的多数类别进行分类。在 Scikit-learn 中实现此算法非常直接。该库通过 `sklearn.neighbors` 模块中的 `KNeighborsClassifier` 类提供了一个直接的实现方式。

### 使用 KNeighborsClassifier

Scikit-learn 中 KNN 分类主要使用 `KNeighborsClassifier`。与 Scikit-learn 其他估计器一样，它遵循常见的 `fit`/`predict` 模式。

首先，您需要导入所需的组件：

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler # 我们会看到这为什么很重要
from sklearn.datasets import load_iris # 一个示例数据集
```

让我们使用 Iris 数据集，这是一个经典的分类参照，进行演示。该数据集包含三种鸢尾花的测量值。

```python
# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 输出形状以验证分割
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
```

现在，我们实例化 `KNeighborsClassifier`。您将指定的最主要的超参数 (parameter) (hyperparameter)是 `n_neighbors`，它对应于 KNN 算法中的 $k$ 值。这决定了在进行预测时要考虑多少个邻居。让我们从 $k=5$ 开始。

```python
# 使用 k=5 实例化分类器
knn = KNeighborsClassifier(n_neighbors=5)
```

训练 KNN 模型很简单。由于 KNN 是一种基于实例的学习器，`fit` 方法主要涉及将训练数据（$X_{train}$ 和 $y_{train}$）存储在一个高效的结构中（例如默认的 Ball Tree 或 KD Tree），以便之后能够快速查询最近邻。

```python
# 训练分类器（存储训练数据）
knn.fit(X_train, y_train)
```

模型“训练”完成后（即训练数据已存储），我们就可以对新的、未见过的数据（$X_{test}$）进行预测。`predict` 方法会为测试集中的每个点在训练数据中查找其 $k$ 个最近邻，并分配这些邻居中最常见的类别。

```python
# 对测试集进行预测
y_pred = knn.predict(X_test)

# 比较预测值与实际标签
print("样本预测值:", y_pred[:10])
print("实际标签:   ", y_test[:10])
```

我们可以使用适用于分类的指标（例如准确率）来评估模型的表现。

```python
# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
```

### 特征缩放的重要性

KNN 大量依赖于计算数据点之间的距离（通常默认是欧几里德距离）。如果特征的尺度差异很大（例如，一个特征范围从 0 到 1，而另一个范围从 1000 到 50000），那么尺度较大的特征将主导距离计算。这可能导致次优的表现，因为算法可能会隐式地将更多权重 (weight)分配给值较大的特征，无论它们实际的预测意义如何。

因此，在应用 KNN 之前，几乎总是建议对特征进行缩放。一种常用技术是标准化，它将特征转换为零均值和单位方差。Scikit-learn 的 `StandardScaler` 是此目的的理想选择。

让我们应用缩放并重新训练模型：

```python
# 实例化缩放器
scaler = StandardScaler()

# 在训练数据上拟合缩放器，并转换训练和测试数据
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # 在测试数据上使用 transform，而不是 fit_transform！

# 实例化一个新的 KNN 分类器
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# 在缩放后的数据上训练
knn_scaled.fit(X_train_scaled, y_train)

# 在缩放后的测试数据上进行预测
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# 评估缩放后的模型
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)
print(f"缩放后模型准确率: {accuracy_scaled:.4f}")
```

在对特征进行缩放后，您通常会发现准确率有所提高，特别是如果原始特征的范围不同。请记住，只在训练数据上拟合缩放器，然后使用它来转换训练集和测试集，以防止测试集数据泄露到训练过程中。

### 选择合适的 $k$ 值

$k$ (`n_neighbors`) 的选择对模型的行为有很大影响。

- 较小的 $k$（例如 $k=1$）使模型对噪声和异常值敏感，可能导致决策边界复杂（高方差，低偏差）。
- 较大的 $k$ 会平滑决策边界，使模型对噪声更具抵抗力，但可能忽略局部模式（低方差，高偏差）。

寻找合适的 $k$ 值通常涉及尝试不同的值并评估它们的表现，通常使用交叉验证技术（我们将在第 5 章中介绍）。目前，请注意 $k=5$ 是一个常见的起始点，但它可能并非适用于所有数据集。

`KNeighborsClassifier` 其他可能有用的一些超参数 (parameter) (hyperparameter)包括：

- `weights`：决定邻居投票的权重 (weight)方式。`'uniform'`（默认）为所有邻居赋予相同的权重。`'distance'` 根据距离反比分配权重，这意味着距离更近的邻居具有更大的影响力。
- `metric`：指定要使用的距离度量（例如，`'minkowski'`，其中 `p=2` 代表欧几里德距离，`p=1` 代表曼哈顿距离）。

对这些超参数，特别是 $k$ 值进行尝试，是构建有效 KNN 模型的一个常规组成部分。我们将在课程的后面部分介绍调整超参数的系统方法。现在，您已经拥有使用 Scikit-learn 实现和评估一个基本 KNN 分类器的工具。

## 参考资料

- [sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) — Scikit-learn Developers (2024)
  KNeighborsClassifier 类的官方文档，详细说明其参数、方法和用法。
- [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) — Scikit-learn Developers (2024)
  StandardScaler 的官方文档，用于特征标准化，这是 KNN 的一个重要预处理步骤。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; Pages: Chapter 2-4
  一本备受推崇的教科书，涵盖 K-近邻算法、分类和相关统计学习概念的理论基础。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: Chapter 3
  一本实用指南，演示如何使用 Scikit-learn 实现和调整 K-近邻算法，包括特征缩放和超参数考量。

---

[上一节](03-K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95%20%28KNN%29.md) · [下一节](05-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%20%28SVM%29%20%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
