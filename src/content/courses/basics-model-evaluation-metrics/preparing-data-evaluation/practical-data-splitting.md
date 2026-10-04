---
course: "basics-model-evaluation-metrics"
chapter: "preparing-data-evaluation"
lesson: "practical-data-splitting"
sourceId: 4037
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation/practical-data-splitting"
title: "动手实践：数据分割"
description: "使用常用工具对样本数据集执行简单的训练-测试分割。"
order: 9
plots: []
sourceHash: "95ca78ec486c6e3d6766bac985298d39466e186737175f89d5903605438643d6"
sourceCorrections: []
---

将数据分成训练集和测试集对于可靠的模型评估是基础的。这里将演示如何实际执行这种分割，采用一种在Python中常用且广受欢迎的`scikit-learn`库所使用的方法。不过，这种做法适用于各种工具，不受具体工具的限制。

设想我们有一个小型数据集，我们想要预测一个水果是苹果（用0表示）还是橙子（用1表示），这基于两个特征：水果的重量（克）和质地（0表示光滑，1表示凹凸不平）。

我们的数据可能看起来像这样：

- **特征 (X)：** 每个水果的重量和质地。
- **标签 (y)：** 实际的水果类型（0代表苹果，1代表橙子）。

假设我们有10个水果的数据：

**特征 (X)：**
`[[150, 0], [170, 0], [140, 1], [130, 1], [160, 0], [180, 0], [125, 1], [135, 1], [190, 0], [145, 1]]`

**标签 (y)：**
`[0, 0, 1, 1, 0, 0, 1, 1, 0, 1]`

我们有10个数据点（行）。`X`中的每一行都对应着`y`中相同位置的标签。例如，第一个水果重150克，质地光滑（`[150, 0]`），并且是苹果（`0`）。

我们的目标是将这些数据分成训练集（用于训练模型）和测试集（用于评估模型学习得如何）。我们将采用常见的70/30分割比例，这表示70%的数据（7个样本）将用于训练，而30%（3个样本）将保留用于测试。

### 使用 `scikit-learn` 进行分割

在Python中，`scikit-learn`库在其`model_selection`模块中提供了一个便捷的函数，名为`train_test_split`。让我们看看如何使用它。

首先，通常需要导入函数并准备数据（通常使用NumPy等库，但对于这个简单的例子，Python列表也适用）：

```python
# 导入函数
from sklearn.model_selection import train_test_split
import numpy as np # 常用于数据表示

# 我们的特征数据（重量，质地）
X = np.array([[150, 0], [170, 0], [140, 1], [130, 1], [160, 0], 
              [180, 0], [125, 1], [135, 1], [190, 0], [145, 1]])

# 我们的标签数据（0=苹果，1=橙子）
y = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1])

# 执行分割（70% 训练，30% 测试）
# 我们设置 random_state 以获得可复现的结果
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 让我们看看结果
print("原始数据点数量：", len(X))
print("训练数据点数量：", len(X_train))
print("测试数据点数量：", len(X_test))

print("\n训练特征 (X_train)：\n", X_train)
print("\n训练标签 (y_train)：\n", y_train)

print("\n测试特征 (X_test)：\n", X_test)
print("\n测试标签 (y_test)：\n", y_test)
```

### 理解 `train_test_split` 函数

让我们分解`train_test_split(X, y, test_size=0.3, random_state=42)`的重要组成部分：

1. **`X`**: 这是我们的输入特征数据。
2. **`y`**: 这是我们对应的标签数据。该函数确保在分割过程中，特征行与其标签之间的关联得到保持。
3. **`test_size=0.3`**: 此参数 (parameter)指定数据集中用于测试分割的比例。这里，`0.3`表示30%的数据将被分配给测试集（`X_test`，`y_test`），其余70%将用于训练集（`X_train`，`y_train`）。您也可以使用`train_size=0.7`来达到相同的效果。如果您提供一个整数而不是浮点数（例如，`test_size=3`），它指定了测试样本的绝对数量。
4. **`random_state=42`**: 如前所述，分割通常涉及在划分数据之前对其进行随机打乱。将`random_state`设置为一个特定的整数（例如42、0或任何其他数字）可确保每次运行代码时都会发生*相同*的随机打乱和分割。这对于获得可复现的结果很重要。如果省略`random_state`，每次都会得到不同的分割结果，这会使调试或比较结果变得困难。

### 查看输出

运行上述代码将产生类似于这样的输出（具体行取决于使用的`random_state`）：

```
Original data points: 10
Training data points: 7
Test data points: 3

Training Features (X_train):
 [[145 1]
 [170 0]
 [140 1]
 [160 0]
 [135 1]
 [180 0]
 [125 1]]

Training Labels (y_train):
 [1 0 1 0 1 0 1]

Test Features (X_test):
 [[190 0]
 [130 1]
 [150 0]]

Test Labels (y_test):
 [0 1 0]
```

请注意：

- 我们最初有10个数据点。
- 训练集（`X_train`，`y_train`）包含7个数据点（10个数据点的70%）。
- 测试集（`X_test`，`y_test`）包含3个数据点（10个数据点的30%）。
- `X_train`中的行对应`y_train`中的标签，测试集也一样。特征和标签的原始关联在每个集合中都得到了保持。
- 数据在分割前已被打乱，这得益于`train_test_split`的特性（由`random_state`控制）。

这里是该过程的简单示意图：

> 该图展示了原始数据集通过分割函数，生成具有指定比例的独立训练集和测试集的过程。

这种数据分割的实际操作步骤很重要。您现在有了一个训练集（`X_train`，`y_train`）来训练您的模型，以及一个模型尚未见过的完全独立的测试集（`X_test`，`y_test`）。这个测试集将在之后用来公正评估您的模型如何根据我们学到的指标来适应新的、未见过的数据。

## 参考资料

- [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) — scikit-learn developers (2023)
  此为`train_test_split`函数的官方文档，详细介绍了其参数和行为。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  本书阐述了数据分割的统计学和理论依据，旨在评估模型性能和泛化误差。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850-856
  本书提供了在完整机器学习项目中，使用常用Python库应用`train_test_split`的实践示例和背景信息。
