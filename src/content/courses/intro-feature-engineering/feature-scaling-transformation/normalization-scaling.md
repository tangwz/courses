---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "normalization-scaling"
sourceId: 1339
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/normalization-scaling"
title: "归一化（最小-最大值缩放）"
description: "应用 MinMaxScaler 将特征缩放至特定范围，通常是 [0, 1]。"
order: 3
plots: ["plots/1339-0.json"]
sourceHash: "54d1cdf0d4735c68c0fad44ae648402a479ff323cf6bc10d3f79969798161ead"
sourceCorrections: []
---

虽然标准化将数据中心置于零点并根据标准差进行缩放，但**归一化 (normalization)**（通常称为**最小-最大值缩放**）采用不同的方法。它的主要目的是将数值特征重新缩放到一个特定、预设的范围内，最常见的是 [0, 1]。

当您需要将特征限制在一个一致的区间内时，这种技术尤其有用。这对于不强烈依赖数据分布假设的算法（与标准化不同，标准化适用于近似高斯分布），或者对于计算距离或使用梯度下降 (gradient descent)的算法通常有益，因为特征尺度差异大可能会导致问题。图像处理常使用归一化来缩放像素强度，像素强度自然落在 [0, 255] 等范围内，并将其缩放到 [0, 1]。

### 最小-最大值缩放公式

这种转换通过以下公式对每个特征 $X$ 实现：


$$
X_{\text{归一化}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}
$$


这里：

- $X$ 是原始特征值。
- $X_{\min}$ 是该特征在训练数据中观测到的最小值。
- $X_{\max}$ 是该特征在训练数据中观测到的最大值。

这个公式将原始特征范围 $[X_{\min}, X_{\max}]$ 线性映射到新范围 [0, 1]。如果一个值等于最小值 ($X_{\min}$)，它将被映射到 0。如果它等于最大值 ($X_{\max}$)，它将被映射到 1。所有其他值按比例落在两者之间。

虽然 [0, 1] 是最常见的目标范围，但该公式可以推广以缩放到任意范围 $[a, b]$：


$$
X_{\text{归一化}} = a + \frac{(X - X_{\min})(b - a)}{X_{\max} - X_{\min}}
$$


然而，Scikit-learn 的实现默认使用 [0, 1] 范围，这足以应对大多数使用场景。

### Scikit-learn 实现

Scikit-learn 通过 `sklearn.preprocessing` 模块中的 `MinMaxScaler` 类提供便捷实现。像 Scikit-learn 中的其他转换器一样，它遵循 `fit` 和 `transform` 模式。

**重要提示：** 缩放器必须*仅*使用训练数据进行拟合。在此 `fit` 步骤中获得的最小值 ($X_{\min}$) 和最大值 ($X_{\max}$) 随后用于 `transform` 训练数据和任何后续数据（如验证集或测试集）。这避免了测试集信息渗入预处理步骤，从而确保对模型性能的准确评估。

以下是一个基本例子：

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 示例数据
data = {'Feature1': np.random.rand(100) * 100,
        'Feature2': np.random.rand(100) * 50 - 25} # 包含负值
df = pd.DataFrame(data)

# 添加一个异常值以说明敏感性
df.loc[100] = {'Feature1': 1000, 'Feature2': 200}

# 分割数据（在拟合缩放器之前必不可少）
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

# 初始化缩放器
scaler = MinMaxScaler(feature_range=(0, 1)) # 默认范围

# 仅在训练数据上拟合缩放器
scaler.fit(X_train)

# 转换训练和测试数据
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 结果是 NumPy 数组。如果需要，可转换回 DataFrame。
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("原始训练数据示例:\n", X_train.head())
# print("\n学习到的最小值:", scaler.data_min_) # 从训练数据学习到的最小值
# print("学习到的最大值:", scaler.data_max_) # 从训练数据学习到的最大值
print("\n缩放后的训练数据示例:\n", X_train_scaled_df.head())
print("\n缩放后的测试数据示例:\n", X_test_scaled_df.head())

# 验证缩放后训练数据的范围（应接近 0 和 1）
print("\n缩放后训练数据最小值:\n", X_train_scaled_df.min())
print("\n缩放后训练数据最大值:\n", X_train_scaled_df.max())
# 注意：如果测试数据包含训练期间观测范围之外的值，
# 缩放后的值可能会超出 [0, 1] 范围。这是预期行为。
print("\n缩放后测试数据最小值:\n", X_test_scaled_df.min())
print("\n缩放后测试数据最大值:\n", X_test_scaled_df.max())
```

注意 `scaler.fit()` 只调用一次，使用 `X_train`。 `X_train` 和 `X_test` 随后都使用 `scaler.transform()` 进行缩放。

### 效果可视化

最小-最大值缩放将数据压缩到 [0, 1] 范围，但保留了分布的整体形状。然而，异常值会很大程度影响其他数据点的缩放。



![最小-最大值缩放的效果 (Feature1)](plots/1339-0.json)



> 分布形状得以保留，但范围被压缩到 [0, 1]。请注意，单个大的异常值 (1000) 的存在如何将大部分数据点在缩放后的版本中挤压到 [0, 1] 范围的很小一部分。

### 优点与缺点

**优点：**

- **固定范围：** 确保所有特征都具有完全相同的尺度，通常是 [0, 1]。这有助于可视化，并对特征幅度敏感的算法有帮助。
- **保留关系：** 保持特征值之间的关系。它不改变分布本身的形状。
- **算法兼容性：** 适用于需要有界输入范围的算法，例如神经网络 (neural network)中的某些激活函数 (activation function)（如 Sigmoid 或 Tanh，尽管 ReLU 激活通常与标准化配合更好）。

**缺点：**

- **对异常值的敏感性：** 这是最主要的缺点。由于缩放直接取决于最小值和最大值，异常值会大大缩小“正常”数据点的范围，将它们压缩到一个非常小的区间。这会降低该特征对许多算法的有用性。
- **不中心化数据：** 与标准化不同，最小-最大值缩放不将数据中心置于零均值。某些算法在使用零中心化数据时表现更好。
- **新数据问题：** 如果测试集中出现的新数据点超出训练期间观测到的原始 $[X_{\min}, X_{\max}]$ 范围，缩放后的值将超出目标 [0, 1] 范围。虽然不一定是错误（它反映新数据超出了训练分布的观测边界），但这是一个需要注意的事项。

### 何时使用最小-最大值缩放

在以下情况下选择最小-最大值缩放：

1. 您知道数据很少或没有明显的异常值，或者您已经处理了它们。
2. 您需要将特征严格限制在特定范围内，例如 [0, 1] 或 [-1, 1]。
3. 您使用的算法不假设特定数据分布（如高斯分布）。例子包括 K-近邻 (KNN) 或图像处理中使用的算法。
4. 您优先考虑所有特征具有完全相同的数值尺度，而不是中心化数据或可靠地处理异常值。

如果您的数据包含明显的异常值，或者您的算法得益于零均值和单位方差的数据（如 PCA、SVM、逻辑回归、线性回归），那么**标准化**或**缩放**（将在下文讨论）通常是更好的选择。

总之，最小-最大值缩放是将特征带到共同尺度的一种直接方法，但它对异常值的敏感性需要仔细考虑。始终记住在训练数据上拟合缩放器，并一致地转换训练集和测试集。

## 参考资料

- [sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) — scikit-learn developers (2023)
  MinMaxScaler类的官方文档，详细介绍了其参数、方法和用法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了特征缩放技术（包括Min-Max缩放）的实用指导及其在机器学习工作流中的应用。
- [Data Mining: Concepts and Techniques](https://www.cs.illinois.edu/~hanj/bk3/) — Jiawei Han, Jian Pei, and Hanghang Tong (2023)
  Publisher: Morgan Kaufmann
  介绍了基本的数据预处理技术，包括数据归一化、其类型以及对数据分析的影响。
