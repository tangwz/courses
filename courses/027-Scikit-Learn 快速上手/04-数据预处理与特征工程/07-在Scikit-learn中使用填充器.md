# 在Scikit-learn中使用填充器

来源：[原文](https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-4-data-preprocessing-feature-engineering/using-imputers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据常不完整，包含缺失条目，在NumPy数组或Pandas DataFrame中通常表示为`NaN`（非数字）。许多Scikit-learn算法无法处理含有缺失值的数据集，因此需要一种处理它们的方法。简单地丢弃含有缺失数据的行或列可能会丢弃有价值的信息，特别是当数据集不是很大时。一个普遍的替代方案是**填充**：用从非缺失数据中计算出的替代值来填补缺失值。

Scikit-learn提供了一个便捷的转换器用于基础填充任务：`SimpleImputer`。它位于`sklearn.impute`模块中，可以让你使用多种方法填充缺失值。

### `SimpleImputer`的使用

`SimpleImputer`遵循标准的Scikit-learn转换器API，这意味着它具有`fit`和`transform`方法。

1. **`fit(X)`**：在`fit`步骤中，填充器会从训练数据`X`的每列非缺失值中计算统计量（例如，均值、中位数、众数）。这个计算得到的统计量会内部存储。
2. **`transform(X)`**：`transform`步骤使用`fit`期间计算出的统计量来填充输入数据`X`中的缺失值（标记 (token)为`np.nan`或其它指定标记）。

对填充器进行拟合操作时，*只*应用于训练数据，然后用它来转换训练数据和测试数据，这一点很重要。这可以防止数据泄露，从而保证测试集的信息不会影响用于训练集的填充值。

### 填充方法

`SimpleImputer`通过`strategy`参数 (parameter)支持多种填充方法：

- `mean`：使用每列的均值替换缺失值。仅适用于数值数据。可能对异常值敏感。
- `median`：使用每列的中位数替换缺失值。也仅适用于数值数据。通常比`mean`更能应对异常值。
- `most_frequent`：使用每列的众数（最常出现的值）替换缺失值。可用于数值和分类数据。
- `constant`：使用`fill_value`参数指定的固定值替换缺失值。如果缺失值有特定含义，或者你想为数值数据使用0或-1之类的占位符，或为分类数据使用“Missing”，这会比较实用。

我们来看看如何使用`median`方法应用`SimpleImputer`。

```python
import numpy as np
from sklearn.impute import SimpleImputer

# 含有缺失值的样本数据
X = np.array([[1.0, 2.0, np.nan],
              [4.0, np.nan, 6.0],
              [7.0, 8.0, 9.0],
              [np.nan, 11.0, 12.0]])

print("原始数据:")
print(X)

# 使用'median'方法初始化SimpleImputer
imputer = SimpleImputer(strategy='median')

# 在数据上拟合填充器（计算每列的中位数）
# 第0列中位数: (1+4+7)/3 -> 4.0 ([1, 4, 7]的中位数)
# 第1列中位数: (2+8+11)/3 -> 8.0 ([2, 8, 11]的中位数)
# 第2列中位数: (6+9+12)/3 -> 9.0 ([6, 9, 12]的中位数)
imputer.fit(X)

# 转换数据（填充缺失值）
X_imputed = imputer.transform(X)

print("\n填充后的数据 (中位数方法):")
print(X_imputed)

# 我们也可以使用fit_transform作为快捷方式
imputer_mean = SimpleImputer(strategy='mean')
X_imputed_mean = imputer_mean.fit_transform(X)

print("\n填充后的数据 (均值方法):")
print(X_imputed_mean)
```

运行此代码会先打印含有`NaN`值的原始数组。接着，它会展示应用中位数填充后的数组，其中`NaN`值被替换为对应的列中位数（4.0、8.0、9.0）。最后，它会展示使用均值方法的结果。

### 指示缺失值

有时，某个值缺失这一事实本身可能对机器学习 (machine learning)模型很有用。`SimpleImputer`通过将`add_indicator`参数 (parameter)设置为`True`，让你能够追踪哪些值被填充了。设置后，转换器会在输出中追加二进制指示列。每个指示列对应输入中曾有缺失值的列，其中1表示原始缺失的条目，0表示其他情况。

```python
# 再次使用样本数据
X = np.array([[1.0, 2.0, np.nan],
              [4.0, np.nan, 6.0],
              [7.0, 8.0, 9.0],
              [np.nan, 11.0, 12.0]])

# 使用中位数方法和指示器初始化SimpleImputer
indicator_imputer = SimpleImputer(strategy='median', add_indicator=True)

# 拟合并转换
X_imputed_indicated = indicator_imputer.fit_transform(X)

print("\n带缺失值指示器的填充数据:")
print(X_imputed_indicated)

# 注意形状差异
print(f"\n原始形状: {X.shape}")
print(f"带指示器填充后的形状: {X_imputed_indicated.shape}")
```

输出的`X_imputed_indicated`将在前三列（与前例中的`X_imputed`相同）包含填充值，并额外增加三个二进制列。第四列的第3行将有1（表示`X[3, 0]`缺失）。第五列的第1行将有1（表示`X[1, 1]`缺失）。第六列的第0行将有1（表示`X[0, 2]`缺失）。

### 填充的考量

虽然填充是一种常用技术，但它并非没有缺点。它对数据做出假设（例如，中位数是合理的替代），并可能扭曲变量之间的关系或减少方差。选择哪种填充方法（`mean`、`median`、`most_frequent`、`constant`）应根据你的数据性质和你打算使用的算法来决定。例如，对于含有异常值的数值数据，通常更推荐使用`median`而非`mean`。如果使用`SimpleImputer`，`most_frequent`或`constant`（带有有意义的`fill_value`）对于分类特征是必需的。

在实践中，你常常需要对不同的列应用不同的填充方法（例如，数值数据用中位数，分类数据用众数）。这通常通过Scikit-learn的`ColumnTransformer`来处理，它允许将不同的转换器应用于不同的列子集，通常在一个更大的Pipeline中。我们将在第6章详细介绍`ColumnTransformer`和Pipeline。目前，理解`SimpleImputer`的工作原理能为处理缺失数据打下良好底子。

## 参考资料

- [sklearn.impute.SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) — Scikit-learn Developers (2023)
  Publisher: Scikit-learn Project
  提供 Scikit-learn 中 SimpleImputer 类的官方 API 参考、参数详情和使用示例。
- [Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html) — Scikit-learn Developers (2023)
  Publisher: Scikit-learn Project
  官方用户指南部分，解释处理缺失值的各种策略，重点介绍 SimpleImputer 及其在 Scikit-learn 中的功能。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.vitalsource.com/products/hands-on-machine-learning-with-scikit-learn-keras-and-aurelien-geron-v9781098122461) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广泛使用的实践指南，涵盖数据预处理技术，包括缺失值插补，并常使用 Scikit-learn 提供具体示例。

---

[上一节](06-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E5%80%BC.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86.md)
