# NumPy中的统计函数

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/numpy-statistical-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对数值数据进行分析，通常需要使用统计量来归纳其主要特征。NumPy 提供了一系列函数，用于直接在 `ndarray` 对象上快速计算常见统计量。这些函数经过高度优化，是数据科学和机器学习 (machine learning)中许多分析任务的基础。熟练使用它们是数据分析工具箱中重要的一步。

这些函数中有许多可以对整个数组或沿特定轴进行数据聚合，这在处理多维数据（例如表示数据集或特征图的矩阵）时非常有用。

## 基本聚合操作

我们从总结数据的基本聚合操作开始，例如求和、最小值和最大值。

考虑一个简单数组：

```python
import numpy as np

data = np.array([1, 5, 2, 8, 3, 9, 4, 7, 6])

# 计算所有元素的和
total_sum = np.sum(data)
print(f"Sum: {total_sum}") # Output: Sum: 45

# 找到最小值和最大值
min_val = np.min(data)
max_val = np.max(data)
print(f"Min: {min_val}, Max: {max_val}") # Output: Min: 1, Max: 9
```

这些函数在一维数组上按预期运行。它们在多维数组上的实用性更明显，您可以在其中指定操作轴。`axis` 参数 (parameter)决定了函数沿哪个维度操作：`axis=0` 沿列操作，`axis=1` 沿行操作。

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 矩阵中所有元素的和
print(f"Total Sum: {np.sum(matrix)}") # Output: Total Sum: 45

# 沿列求和 (axis=0)
print(f"Column Sums: {np.sum(matrix, axis=0)}") # Output: Column Sums: [12 15 18]

# 找到每行的最大值 (axis=1)
print(f"Row Maximums: {np.max(matrix, axis=1)}") # Output: Row Maximums: [3 6 9]

# 找到每列的最小值 (axis=0)
print(f"Column Minimums: {np.min(matrix, axis=0)}") # Output: Column Minimums: [1 2 3]
```

使用 `axis` 参数可以沿着特定维度压缩信息，这在总结数据集中特征或样本时是常见需求。

## 集中趋势和离散度度量

除了简单的求和和极值，NumPy 还提供了计算集中趋势（如平均值和中位数）和离散度（如标准差和方差）度量的函数。

- `np.mean()`: 计算算术平均值。
- `np.median()`: 计算中位数（排序数据的中间值）。与平均值相比，对异常值不敏感。
- `np.std()`: 计算标准差，衡量数据围绕平均值的离散程度。
- `np.var()`: 计算方差，即标准差的平方。

```python
scores = np.array([75, 82, 88, 91, 65, 95, 88, 78])

# 计算平均值和中位数
mean_score = np.mean(scores)
median_score = np.median(scores)
print(f"Mean Score: {mean_score:.2f}")     # Output: Mean Score: 82.75
print(f"Median Score: {median_score:.2f}") # Output: Median Score: 85.00

# 计算方差和标准差
variance = np.var(scores)
std_dev = np.std(scores)
print(f"Variance: {variance:.2f}")         # Output: Variance: 97.69
print(f"Standard Deviation: {std_dev:.2f}") # Output: Standard Deviation: 9.88
```

与聚合函数一样，这些函数也接受多维数组的 `axis` 参数 (parameter)：

```python
# 使用上一个示例中的“matrix”
print(f"Mean of each column: {np.mean(matrix, axis=0)}") # Output: Mean of each column: [4. 5. 6.]
print(f"Median of each row: {np.median(matrix, axis=1)}") # Output: Median of each row: [2. 5. 8.]
print(f"Std Dev of each column: {np.std(matrix, axis=0)}") # Output: Std Dev of each column: [2.44948974 2.44948974 2.44948974]
```

## 百分位数

百分位数通过指示某个百分比的观测值低于哪个值来帮助理解数据的分布。`np.percentile()` 是实现此功能的函数。实际上，中位数就是第 50 个百分位数。

```python
data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 计算第 25 百分位数（第一四分位数）
q1 = np.percentile(data, 25)
print(f"25th Percentile (Q1): {q1}") # Output: 25th Percentile (Q1): 32.5

# 计算第 75 百分位数（第三四分位数）
q3 = np.percentile(data, 75)
print(f"75th Percentile (Q3): {q3}") # Output: 75th Percentile (Q3): 77.5

# 一次计算多个百分位数
percentiles = np.percentile(data, [10, 50, 90])
print(f"10th, 50th, 90th Percentiles: {percentiles}") # Output: 10th, 50th, 90th Percentiles: [19. 55. 91.]
```

百分位数常用于探索性数据分析，以了解数据分布范围并发现潜在的异常值，通常使用箱线图（依赖于四分位数）进行可视化。

## 相关性

理解不同变量（特征）之间的关系在机器学习 (machine learning)中很重要。相关性衡量两个变量之间的线性关联程度。NumPy 的 `np.corrcoef()` 函数计算皮尔逊相关系数矩阵。

皮尔逊相关系数的取值范围是 -1 到 +1：

- +1 表示完全正线性相关。
- -1 表示完全负线性相关。
- 0 表示没有线性相关。

该函数需要一个数组，其中每*行*表示一个变量，每*列*表示一个观测值。

```python
# 表示 5 个个体的两个变量（例如身高和体重）的数据
# 变量 1（身高）：第 0 行
# 变量 2（体重）：第 1 行
measurements = np.array([[1.7, 1.8, 1.6, 1.9, 1.5],  # 身高 (米)
                         [ 70,  85,  65,  90,  60]]) # 体重 (公斤)

# 计算相关矩阵
correlation_matrix = np.corrcoef(measurements)
print("Correlation Matrix:")
print(correlation_matrix)
# Output:
# Correlation Matrix:
# [[1.         0.98137466]  <- 身高与身高（1）和体重的相关性
#  [0.98137466 1.        ]] <- 体重与身高和体重（1）的相关性
```

所得矩阵是对称的。对角线元素始终为 1（变量与自身的相关性）。非对角线元素显示不同变量之间的相关性。在此示例数据集中，值 `0.981` 表示身高和体重之间存在很强的正线性关系。

## 处理缺失值 (NaN)

"数据集通常包含缺失值，在 NumPy 中表示为 `np.nan`（非数字）。标准统计函数通常会传播 `NaN` 值，这意味着如果任何输入元素是 `NaN`，结果也将是 `NaN`。"

```python
data_with_nan = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

print(f"Sum with NaN: {np.sum(data_with_nan)}")     # Output: Sum with NaN: nan
print(f"Mean with NaN: {np.mean(data_with_nan)}")   # Output: Mean with NaN: nan
```

为此，NumPy 提供了许多统计函数的 `nan` 安全版本（例如，`np.nansum()`、`np.nanmean()`、`np.nanmedian()`、`np.nanstd()`、`np.nanvar()`、`np.nanpercentile()`）。这些函数在计算时会忽略所有 `NaN` 值。

```python
# 使用 nan-安全函数
print(f"NaN-ignored Sum: {np.nansum(data_with_nan)}")   # Output: NaN-ignored Sum: 12.0
print(f"NaN-ignored Mean: {np.nanmean(data_with_nan)}") # Output: NaN-ignored Mean: 3.0
print(f"NaN-ignored Max: {np.nanmax(data_with_nan)}")   # Output: NaN-ignored Max: 5.0
```

在应用更复杂的插补技术之前，对数据集进行初步统计分析时，使用这些 `nan` 安全函数通常是不可或缺的。

这些 NumPy 统计函数提供了计算描述性统计量的高效方法，帮助理解数据分布，并研究变量之间的关系，构成了机器学习 (machine learning)中数据分析和预处理阶段的重要组成部分。

## 参考资料

- [NumPy v1.26 Manual: Statistical functions](https://numpy.org/doc/stable/reference/routines.statistics.html) — NumPy Developers (2023)
  提供所有NumPy统计函数的官方和最新参考，包括对`axis`等参数和处理`NaN`值的详细解释。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用NumPy和Pandas进行数据操作和统计分析的基础书籍，为数组操作、聚合和数据清洗提供了实用指导。
- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  尽管本书侧重于机器学习算法，但它阐述了统计度量（通常使用NumPy计算）如何应用于机器学习工作流中实际的数据准备和特征工程阶段。

---

[上一节](06-NumPy%20%E4%B8%AD%E7%9A%84%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%E8%BF%90%E7%AE%97.md) · [下一节](08-%E8%AF%BB%E5%8F%96%E5%92%8C%E5%86%99%E5%85%A5%E6%95%B0%E7%BB%84%E6%95%B0%E6%8D%AE%E5%88%B0%E6%96%87%E4%BB%B6.md)
