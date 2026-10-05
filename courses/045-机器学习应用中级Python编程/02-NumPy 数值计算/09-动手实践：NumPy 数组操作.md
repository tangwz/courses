# 动手实践：NumPy 数组操作

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/practice-numpy-manipulation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将 NumPy 数组的基础知识，包括索引、数学运算、广播和统计函数，付诸实践非常重要。完成实际练习有助于巩固理解并提高高效处理数值数据的信心。

本节提供旨在加强本章所讲方法的动手练习。我们将练习涉及创建数组、选取数据、进行计算和应用统计方法的示例，这些技能直接适用于机器学习 (machine learning)中的数据准备和分析。

### 问题 1：创建和分析传感器数据

假设您从三个不同的传感器在四个连续时间点收集了温度读数（摄氏度）。读数如下：传感器 1：[22.5, 23.1, 22.8, 23.5]，传感器 2：[21.8, 22.2, 22.0, 22.5]，传感器 3：[23.0, 23.3, 23.1, 23.6]。

**任务：**

1. 创建一个 2D NumPy 数组来表示这些数据，其中每行对应一个传感器，每列对应一个时间点。
2. 计算每个传感器在所有时间点上的平均温度。
3. 计算每个时间点上所有传感器的平均温度。
4. 找出总体记录的最高温度。

**解答：**

```python
import numpy as np

# 1. 创建 2D 数组
sensor_data = np.array([
    [22.5, 23.1, 22.8, 23.5],
    [21.8, 22.2, 22.0, 22.5],
    [23.0, 23.3, 23.1, 23.6]
])

print("传感器数据数组：")
print(sensor_data)
print("-" * 20)

# 2. 每个传感器的平均温度（沿列，axis=1）
avg_per_sensor = np.mean(sensor_data, axis=1)
print("每个传感器的平均温度：")
print(avg_per_sensor)
print("-" * 20)

# 3. 每个时间点的平均温度（沿行，axis=0）
avg_per_timepoint = np.mean(sensor_data, axis=0)
print("每个时间点的平均温度：")
print(avg_per_timepoint)
print("-" * 20)

# 4. 总体最高温度
max_temp = np.max(sensor_data)
print(f"总体最高温度：{max_temp:.1f}°C")
```

**说明：**

- 我们首先使用包含列表的列表通过 `np.array()` 创建 `sensor_data` 数组。
- 要计算每个传感器的平均值，我们使用 `np.mean()` 并指定 `axis=1`。这指示 NumPy 沿着水平轴（对每行的列）计算平均值。
- 类似地，指定 `axis=0` 会沿着垂直轴（对每列的行）计算平均值，从而给出每个时间点的平均温度。
- 不带 axis 参数 (parameter)的 `np.max()` 会找到整个数组中的最大值。

### 问题 2：数据标准化（Z-分数标准化）

Z-分数标准化（或称 Z-score 标准化）是机器学习 (machine learning)中常见的预处理步骤。它涉及将数据转换为均值为 0、标准差为 1 的形式。数据点 $x$ 的公式为：


$$
Z = \frac{x - \mu}{\sigma}
$$


其中 $\mu$ 是数据的均值，$\sigma$ 是标准差。

**任务：**

1. 创建一个包含以下值的一维 NumPy 数组：`[10, 15, 12, 18, 25, 11, 16]`。
2. 计算此数组的均值 ($μ$) 和标准差 ($σ$)。
3. 使用 NumPy 运算（利用广播）应用 Z-分数标准化公式来标准化数组。

**解答：**

```python
import numpy as np

# 1. 创建数组
data = np.array([10, 15, 12, 18, 25, 11, 16])
print("原始数据：")
print(data)
print("-" * 20)

# 2. 计算均值和标准差
mean_val = np.mean(data)
std_dev = np.std(data)
print(f"均值 (μ): {mean_val:.2f}")
print(f"标准差 (σ): {std_dev:.2f}")
print("-" * 20)

# 3. 使用广播应用 Z-分数标准化
normalized_data = (data - mean_val) / std_dev
print("标准化数据 (Z-分数)：")
print(normalized_data)
print("-" * 20)

# 验证：检查标准化数据的均值和标准差
print(f"标准化数据的均值：{np.mean(normalized_data):.2f}")
print(f"标准化数据的标准差：{np.std(normalized_data):.2f}")
```

**说明：**

- 我们使用 `np.mean()` 和 `np.std()` 计算均值和标准差。
- 标准化的核心发生在 `normalized_data = (data - mean_val) / std_dev` 这一行。在这里，`mean_val`（一个标量）从 `data` 数组的每个元素中减去（广播）。此减法的结果（一个数组）随后按元素除以 `std_dev`（另一个标量，再次使用广播）。这可以高效地将 Z-分数公式应用于整个数组，而无需显式循环。
- 验证步骤显示，生成的 `normalized_data` 的均值非常接近 0，标准差非常接近 1，符合预期。

### 问题 3：使用布尔索引选择和修改数据

考虑一个表示学生两次不同考试成绩的数据集：`scores = np.array([[85, 92], [78, 81], [91, 95], [60, 65], [72, 79]])`。每行代表一名学生，第 0 列是考试 1 的分数，第 1 列是考试 2 的分数。

**任务：**

1. 选择考试 1 成绩达到 80 分或以上的学生的所有分数（两次考试）。
2. 找出任一考试成绩低于 70 分的学生。
3. 假设考试 2 进行了加权，为所有在该考试中得分低于 80 分的学生加 3 分。创建一个反映此加权的新数组，而不修改原始 `scores` 数组。

**解答：**

```python
import numpy as np

scores = np.array([[85, 92], [78, 81], [91, 95], [60, 65], [72, 79]])
print("原始分数：")
print(scores)
print("-" * 20)

# 1. 选择考试 1 成绩 >= 80 的学生
high_scorers_test1 = scores[scores[:, 0] >= 80]
print("考试 1 成绩 >= 80 的学生分数：")
print(high_scorers_test1)
print("-" * 20)

# 2. 找出任一考试成绩 < 70 的学生
low_scorers_mask = (scores[:, 0] < 70) | (scores[:, 1] < 70)
low_scorers = scores[low_scorers_mask]
print("任一考试成绩 < 70 的学生分数：")
print(low_scorers)
print("-" * 20)

# 3. 对考试 2 成绩 < 80 的学生应用加权
# 创建副本以避免修改原始数组
curved_scores = scores.copy()
# 为考试 2 成绩 < 80 创建布尔掩码
test2_curve_mask = curved_scores[:, 1] < 80
# 使用掩码在相关列上添加 3 分
curved_scores[test2_curve_mask, 1] += 3

print("对考试 2 成绩 (< 80) 应用加权后的分数：")
print(curved_scores)
print("-" * 20)
print("原始分数（未更改）：")
print(scores)
```

**说明：**

- **任务 1：** `scores[:, 0]` 选择所有行 (`:`) 和第一列 (`0`)，对应考试 1 的分数。条件 `>= 80` 创建一个布尔数组（`[True, False, True, False, False]`）。使用此布尔数组作为 `scores` 的索引，只会选择条件为 `True` 的行。
- **任务 2：** 我们创建两个布尔条件：`scores[:, 0] < 70` 用于考试 1，`scores[:, 1] < 70` 用于考试 2。逻辑或运算符 (`|`) 将它们组合起来，生成一个掩码，如果学生在*至少一次*考试中得分低于 70，则该掩码为 `True`。然后使用此掩码选择相关行。
- **任务 3：** 使用 `scores.copy()` 创建 `curved_scores` 很重要。否则，修改会影响原始 `scores` 数组。我们专门为考试 2 成绩低于 80 的情况创建了一个掩码 `test2_curve_mask`。然后，`curved_scores[test2_curve_mask, 1]` 选择由掩码指示的行，*但仅在第二列中*（考试 2 成绩）。然后我们使用 `+= 3` 直接将 3 添加到这些选定的元素。

### 问题 4：重塑和基本线性代数

您会得到一个一维数组，表示灰度图像片段的像素值：`pixels = np.arange(1, 13)`。

**任务：**

1. 将此数组重塑为 3x4 矩阵（表示 3 行、4 列像素）。
2. 创建一个 4x2 矩阵 `transform_matrix`，其值为 `[[0.5, 0.5], [1, 0], [0, 1], [0.2, 0.8]]`。
3. 对重塑后的像素矩阵和 `transform_matrix` 执行矩阵乘法。结果矩阵的形状是什么？

**解答：**

```python
import numpy as np

pixels = np.arange(1, 13)
print("原始像素数组：")
print(pixels)
print("-" * 20)

# 1. 重塑数组
pixel_matrix = pixels.reshape((3, 4))
print("重塑后的像素矩阵 (3x4)：")
print(pixel_matrix)
print("-" * 20)

# 2. 创建变换矩阵
transform_matrix = np.array([
    [0.5, 0.5],
    [1, 0],
    [0, 1],
    [0.2, 0.8]
])
print("变换矩阵 (4x2)：")
print(transform_matrix)
print("-" * 20)

# 3. 执行矩阵乘法
result_matrix = np.dot(pixel_matrix, transform_matrix)
# 替代语法：result_matrix = pixel_matrix @ transform_matrix

print("矩阵乘法结果 (pixel_matrix @ transform_matrix)：")
print(result_matrix)
print("-" * 20)
print(f"结果矩阵的形状：{result_matrix.shape}")
```

**说明：**

- `np.arange(1, 13)` 创建数组 `[1, 2, ..., 12]`。
- `reshape((3, 4))` 方法将这 12 个元素重新组织成一个有 3 行 4 列的矩阵。请注意，元素总数必须保持不变（3 \* 4 = 12）。
- 矩阵乘法要求内部维度匹配。我们正在将一个 (3x4) 矩阵乘以一个 (4x2) 矩阵。内部维度（4）匹配。
- 我们使用 `np.dot()`（或 `@` 运算符）进行矩阵乘法。如果形状通过广播兼容，标准乘法 (`*`) 将执行元素级乘法，但这并非我们在此所需。
- 结果矩阵的形状由外部维度决定：(3x4) @ (4x2) 得到一个 (3x2) 矩阵。

这些练习演示了您所学的不同 NumPy 功能（创建、索引、切片、广播、数学运算和基本线性代数）如何在实际场景中协同工作。当您继续使用 Pandas 和 Scikit-learn 等库时，会发现扎实掌握这些 NumPy 操作是不可或缺的。请继续尝试不同的数组形状、运算和索引技巧，以提高熟练度。

## 参考资料

- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — NumPy Developers (2024)
  NumPy的官方文档，提供关于NumPy数组的基础知识，包括创建、索引、广播和数学运算。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Python核心库进行数据操作的实用指南，包含关于NumPy数组及其在数据准备中应用的详细章节。
- [The NumPy array object](https://scipy-lectures.org/intro/numpy/array_object.html) — Gaël Varoquaux, Emmanuelle Gouillart, and Olav Vahtras (2024)
  Journal: SciPy Lecture Notes
  一份简明实用的在线教程，介绍NumPy数组，包括创建、基本操作、索引和线性代数概念。

---

[上一节](08-%E8%AF%BB%E5%8F%96%E5%92%8C%E5%86%99%E5%85%A5%E6%95%B0%E7%BB%84%E6%95%B0%E6%8D%AE%E5%88%B0%E6%96%87%E4%BB%B6.md) · [下一节](../03-Pandas%20%E6%95%B0%E6%8D%AE%E6%93%8D%E4%BD%9C/01-Pandas%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E7%AE%80%E4%BB%8B.md)
