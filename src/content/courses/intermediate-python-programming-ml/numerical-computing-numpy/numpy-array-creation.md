---
course: "intermediate-python-programming-ml"
chapter: "numerical-computing-numpy"
lesson: "numpy-array-creation"
sourceId: 2113
sourceUrl: "https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/numpy-array-creation"
title: "数组创建方法"
description: "学习从列表、使用内置函数或从数据文件中创建 NumPy 数组的多种方法。"
order: 2
plots: []
sourceHash: "832e7458f3c43fb34c7251023ae991f034653871a198169f316bd2bcb5fb4314"
sourceCorrections: []
---

NumPy 的核心是 `ndarray`，它是 Python 列表中用于数值数据的快速且内存高效的替代品。NumPy 提供了一系列功能丰富的函数，用于以各种方式生成这些数组，以满足数据分析和机器学习 (machine learning)中的不同需求。

### 从现有 Python 数据创建数组

创建 NumPy 数组最直接的方法是使用 `np.array()` 函数，转换现有的 Python 序列型对象，例如列表或元组。

```python
import numpy as np

# 从 Python 列表创建一维数组
list_data = [1, 2, 3, 4, 5]
arr1d = np.array(list_data)
print(arr1d)
# Output: [1 2 3 4 5]
print(arr1d.dtype) # 检查数据类型
# Output: int64 (or int32 depending on your system)

# 从列表的列表创建二维数组
nested_list = [[1, 2, 3], [4, 5, 6]]
arr2d = np.array(nested_list)
print(arr2d)
# Output:
# [[1 2 3]
#  [4 5 6]]
print(arr2d.shape) # 检查维度（行，列）
# Output: (2, 3)
```

NumPy 在创建数组时，会尝试推断最合适的数据类型 (`dtype`)。但是，您可以使用 `dtype` 参数 (parameter)明确指定数据类型。这对于控制内存使用和数值精度很重要。

```python
# 指定浮点数据类型
arr_float = np.array([1, 2, 3], dtype=np.float64)
print(arr_float)
# Output: [1. 2. 3.]
print(arr_float.dtype)
# Output: float64

# 指定布尔数据类型
arr_bool = np.array([0, 1, 2, 0, 3], dtype=bool)
print(arr_bool)
# Output: [False  True  True False  True]
print(arr_bool.dtype)
# Output: bool
```

请记住，NumPy 数组是同质的；所有元素必须是相同的数据类型。如果您提供混合类型的数据（例如，整数和浮点数），NumPy 会将它们向上转换（upcast）为可以容纳所有元素的最通用类型（通常是浮点型或对象型）。

### 创建带占位符的数组

通常，您需要创建特定大小和形状的数组，但尚未确定最终值，也许是作为稍后填充的占位符。NumPy 为此提供了几个函数：

- `np.zeros()`: 创建一个完全由零填充的数组。
- `np.ones()`: 创建一个完全由一填充的数组。
- `np.full()`: 创建一个填充有指定常数值的数组。
- `np.empty()`: 创建一个初始内容是随机的数组，其值取决于内存状态。它比 `zeros` 或 `ones` 稍快，因为它避免了填充数组，但您在使用它之前必须明确地为每个元素赋值。

所有这些函数都接受一个 `shape` 参数 (parameter)，通常是一个指定数组维度的元组（例如，2D 数组的 `(行, 列)`），以及一个可选的 `dtype` 参数。

```python
# 创建一个 3x4 的零数组（默认为 float64）
zeros_arr = np.zeros((3, 4))
print(zeros_arr)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# 创建一个包含 5 个整数类型的一维数组
ones_arr = np.ones(5, dtype=np.int32)
print(ones_arr)
# Output: [1 1 1 1 1]

# 创建一个填充值为 99 的 2x2 数组
full_arr = np.full((2, 2), 99)
print(full_arr)
# Output:
# [[99 99]
#  [99 99]]

# 创建一个未初始化的 2x3 数组
empty_arr = np.empty((2, 3))
print(empty_arr) # 值将是任意的
# Output（示例，会因系统而异）：
# [[6.95190771e-310 6.95190771e-310 6.95190771e-310]
#  [6.95190771e-310 0.00000000e+000 0.00000000e+000]]
```

这些函数在机器学习 (machine learning)中经常使用，例如，在训练模型之前初始化权重 (weight)矩阵或创建结果数组。

### 创建序列数组

NumPy 提供了类似于 Python 内置 `range` 函数的函数，但设计用于直接生成 NumPy 数组：

- `np.arange()`: 返回给定区间内均匀分布的值。它接受 `start`、`stop` 和 `step` 参数 (parameter)，类似于 `range`。请注意，`stop` 值是*不包含*在内的。
- `np.linspace()`: 在指定区间内返回均匀分布的数字。它接受 `start`、`stop` 和 `num`（要生成的样本数）参数。重要的是，`stop` 值默认是*包含*在内的。

```python
# 一个从 0 到 10（不包括 10）的数组
arr_range = np.arange(10)
print(arr_range)
# Output: [0 1 2 3 4 5 6 7 8 9]

# 一个从 2 到 10（不包括 10），步长为 2 的数组
arr_range_step = np.arange(2, 10, 2)
print(arr_range_step)
# Output: [2 4 6 8]

# 0 到 1 之间（包括 1）的 5 个均匀分布的值
arr_linspace = np.linspace(0, 1, 5)
print(arr_linspace)
# Output: [0.   0.25 0.5  0.75 1.  ]

# 0 到 5 之间（包括 5）的 10 个均匀分布的值
arr_linspace_2 = np.linspace(0, 5, 10)
print(arr_linspace_2)
# Output: [0.         0.55555556 1.11111111 1.66666667 2.22222222 2.77777778
#  3.33333333 3.88888889 4.44444444 5.        ]
```

`linspace` 在您需要特定数量的点均匀分布在一个区间内时特别有用，例如，在为绘图函数生成坐标时。

### 创建随机值数组

在机器学习 (machine learning)中，生成包含随机数的数组对于各种任务来说是不可或缺的，例如初始化模型参数 (parameter)、生成合成数据或打乱数据集。NumPy 的 `random` 子模块为此提供了许多函数：

- `np.random.rand()`: 创建一个指定形状的数组，并用 [0, 1) 区间上*均匀分布*的随机样本填充它。
- `np.random.randn()`: 创建一个指定形状的数组，并用*标准正态分布*（均值 0，方差 1）的随机样本填充它。
- `np.random.randint()`: 返回指定 `low`（包含）到 `high`（不包含）边界之间的随机整数。您还可以指定输出数组的 `size`（形状）。
- `np.random.seed()`: 用于设置随机种子，这使得随机数生成可预测。这对于实验的可复现性很重要。

```python
# 设置种子以保证可复现性
np.random.seed(42)

# 创建一个 2x3 数组，其随机值来自均匀分布 [0, 1)
rand_arr = np.random.rand(2, 3)
print(rand_arr)
# Output:
# [[0.37454012 0.95071431 0.73199394]
#  [0.59865848 0.15601864 0.15599452]]

# 创建一个 3x2 数组，其随机值来自标准正态分布
randn_arr = np.random.randn(3, 2)
print(randn_arr)
# Output:
# [[ 0.05808361 -0.75634998]
#  [-0.34791215  0.1579198 ]
#  [ 0.45615031  0.99712472]]

# 生成 5 个介于 1（包含）和 10（不包含）之间的随机整数
randint_arr = np.random.randint(1, 10, size=5)
print(randint_arr)
# Output: [8 4 6 8 8]

# 生成一个 2x4 数组，其随机整数介于 0（包含）和 5（不包含）之间
randint_arr_2d = np.random.randint(0, 5, size=(2, 4))
print(randint_arr_2d)
# Output:
# [[3 0 2 3]
#  [1 1 4 2]]
```

### 其他创建方法

NumPy 还包含用于创建特定类型数组的函数：

- `np.eye()`: 创建一个二维单位矩阵（对角线上为 1，其他位置为 0）。
- `np.diag()`: 可以提取现有二维数组的对角线，或创建一个在对角线上具有指定值的二维数组。

```python
# 创建一个 3x3 单位矩阵
identity_matrix = np.eye(3)
print(identity_matrix)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# 创建一个对角线上为 [1, 2, 3] 的二维数组
diag_matrix = np.diag([1, 2, 3])
print(diag_matrix)
# Output:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
```

掌握这些数组创建方法是有效使用 NumPy 的第一步。选择正确的方法取决于您是在转换现有数据、需要占位符、需要特定序列，还是需要为模拟或初始化生成随机数据。借助这些工具，您可以高效地构建 `ndarray` 对象，它们构成了 Python 中机器学习 (machine learning)数值运算的基础。

## 参考资料

- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  涵盖从Python序列创建NumPy ndarray、dtype指定和占位符数组函数，对数据处理至关重要。
- [Introduction to Machine Learning with Python: A Guide for Data Scientists](https://www.oreilly.com/library/view/introduction-to-machine/9781449369880/) — Andreas C. Müller and Sarah Guido (2016)
  Publisher: O'Reilly Media
  为在机器学习中使用NumPy数组提供背景，涵盖数组创建和操作作为数据预处理的先决条件。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125963/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  在机器学习工作流程和示例中，通过实例说明NumPy数组的实际应用，包括它们的创建和操作。
