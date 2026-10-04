---
course: "getting-started-with-scikit-learn"
chapter: "model-selection-evaluation"
lesson: "train-test-split"
sourceId: 1890
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-5-model-selection-evaluation/train-test-split"
title: "数据划分：训练集与测试集"
description: "使用 `train_test_split` 创建训练和测试数据集，以实现可靠的模型评估。"
order: 2
plots: []
sourceHash: "0ff2c83ff6fa8de27975169e96eecfe7bc6cd5d9c9326e5b440c35e03b985444"
sourceCorrections: []
---

正如本章引言中指出的，仅凭模型训练时使用的数据进行评估，会对其能力给出过于乐观且常有误导性的看法。模型可能通过记忆数据点（包括噪声和特定怪癖）而在训练集上达到高准确率，而不是学习到一般规律。这种现象被称为过拟合 (overfitting)，会导致模型在遇到新的、未见过的数据时表现不佳。

为了更实际地估计模型在实际情况中的表现，我们需要在模型训练时未见过的数据上进行评估。实现这一目的最直接的方法是将数据集分成两个独立的部分：

1. **训练集：** 用于训练模型。模型从这些数据中学习模式、关系和参数 (parameter)。
2. **测试集（或保留集）：** 用于评估已训练模型的表现。这些数据在训练期间保持独立，并作为新的、未见过数据的代表。

通过在一个子集上训练，并在另一个子集上测试，我们模拟了部署模型并评估其泛化能力的过程，即它在模型开发过程中未使用的数据上表现良好的能力。

### 使用Scikit-learn实现训练-测试数据划分

Scikit-learn 在 `model_selection` 模块中提供了一个方便的实用函数 `train_test_split`，用于将数据划分成训练集和测试集。它负责打乱数据（如果数据有固有顺序，这一点很重要）并将其分成所需比例的部分。

让我们看看如何使用它。假设你的特征存储在名为 `X` 的 NumPy 数组或 Pandas DataFrame 中，而对应的目标变量（标签或值）存储在名为 `y` 的 NumPy 数组或 Pandas Series 中。

```python
# 导入函数
from sklearn.model_selection import train_test_split
import numpy as np
# 假设 X 和 y 已经定义
# 用于演示：
# X = np.arange(20).reshape(10, 2) # 示例特征（10个样本，2个特征）
# y = np.arange(10) # 示例目标变量

# 执行划分
# 我们将25%的数据用于测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 检查结果数组的形状（示例输出）
# print(f"X_train shape: {X_train.shape}") # 如果 X 是 (10, 2) 则为 (7, 2)
# print(f"X_test shape: {X_test.shape}")   # 如果 X 是 (10, 2) 则为 (3, 2)
# print(f"y_train shape: {y_train.shape}") # 如果 y 是 (10,) 则为 (7,)
# print(f"y_test shape: {y_test.shape}")   # 如果 y 是 (10,) 则为 (3,)
```

### `train_test_split` 的参数 (parameter)

- **`X`, `y`**: 这些是分别包含你的特征和目标变量的数组或数据帧。如果需要，你也可以传入多个数组（例如，如果你有不同的特征集）。
- **`test_size`**: 此参数指定测试集中数据所占的比例。它通常是一个介于 0.0 和 1.0 之间的浮点数。例如，`test_size=0.25` 表示 25% 的数据将用于测试，其余 75% 用于训练。或者，你可以使用 `train_size` 来指定训练集所占的比例。如果只指定一项，另一项将自动推断。你也可以传入一个整数，表示测试样本的绝对数量。
- **`random_state`**: 这是一个对结果复现性很重要的参数。默认情况下，该函数在划分前会打乱数据。将 `random_state` 设置为一个整数（例如 `42`、`0`、`123`）可以确保每次运行代码时都生成相同的随机划分。这对于调试、公平比较不同模型以及确保他人能够复现你的结果都是必不可少的。如果你省略 `random_state` 或将其设置为 `None`，每次都会得到不同的划分。
- **`shuffle`**: 一个布尔参数（默认为 `True`），指示是否在划分数据前进行混洗。通常建议进行混洗，以确保训练集和测试集是具有代表性的样本，特别是当原始数据集以某种方式排序时（例如，按时间或类别排序）。
- **`stratify`**: 此参数对于分类任务特别有用，尤其是在处理不平衡数据集时（某些类别的出现频率远低于其他类别）。通过设置 `stratify=y`，该函数确保目标变量 `y` 中值的比例在训练集和测试集中都得到保留。例如，如果你的目标变量包含 80% 的 A 类和 20% 的 B 类，分层抽样可确保 `y_train` 和 `y_test` 都保持大约 80/20 的比例。如果没有分层抽样，随机划分可能会偶然地将稀有类别的几乎所有样本放入训练集或测试集，从而导致训练或评估结果出现偏差。

### 使用训练-测试划分的工作流程

一旦你划分了数据，典型的机器学习 (machine learning)工作流程如下：

1. **选择并实例化模型：** 选择一个 Scikit-learn 估计器（例如，`LinearRegression`、`LogisticRegression`、`KNeighborsClassifier`）。

   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   ```
2. **训练模型：** 仅使用训练数据（`X_train` 和 `y_train`）拟合模型。模型在此阶段绝不能看到测试数据。

   ```python
   # 仅在训练数据上拟合模型
   model.fit(X_train, y_train)
   ```
3. **进行预测：** 使用训练好的模型对*测试*特征（`X_test`）进行预测。

   ```python
   # 仅在测试数据上进行预测
   y_pred = model.predict(X_test)
   ```
4. **评估：** 使用适当的评估指标（例如，分类任务的准确率、精确率、召回率；回归任务的 MAE、MSE、R²）将预测结果（`y_pred`）与测试集中的实际目标值（`y_test`）进行比较。

   ```python
   from sklearn.metrics import accuracy_score # 分类任务示例

   # 评估模型在测试集上的表现
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Model Accuracy on Test Set: {accuracy:.4f}")
   ```

与在训练数据上进行评估相比，这种在测试集上的评估更能可靠地表明模型在新、未见过的数据上可能如何表现。

### 局限性与后续步骤

尽管训练-测试数据划分是模型评估的基本方法，但它有一个局限：性能评估结果在很大程度上取决于具体哪些数据点最终进入了训练集和测试集。如果你得到一个“幸运”或“不幸”的划分，你的评估指标可能会过于乐观或悲观。此外，通过保留一部分数据用于测试，你减少了可用于训练模型的数据量。

为了获得更可靠的性能评估结果并更好地使用现有数据，我们通常会采用**交叉验证**技术，这将在下一节中讨论。交叉验证涉及多次训练-测试划分，从而提供更平均和稳定的模型泛化能力衡量标准。

## 参考资料

- [1.1. Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html#train-test-split) — Scikit-learn developers (2024)
  Scikit-learn官方文档，介绍了train_test_split函数的作用、参数及其在模型评估中的使用方法。
- [An Introduction to Statistical Learning: With Applications in R (2nd Edition)](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1819-2](https://doi.org/10.1007/978-1-0716-1819-2)
  一本基础教科书，解释了统计学习的原理，包括模型评估、独立测试集的重要性以及过拟合问题。
