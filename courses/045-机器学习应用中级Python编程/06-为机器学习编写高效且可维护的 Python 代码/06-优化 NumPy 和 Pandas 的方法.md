# 优化 NumPy 和 Pandas 的方法

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/optimizing-numpy-pandas)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 NumPy 和 Pandas 为数值计算和数据处理提供了强大的抽象能力，但要达到最佳性能通常需要有意识的努力。如果实现方式未能优化，看似简单的操作可能导致计算开销大或内存占用高。在机器学习 (machine learning)场景中，这一点尤为重要，因为数据集可能很庞大，并且在实验和模型训练过程中数据处理步骤会频繁重复。专注于高效使用 NumPy 和 Pandas 可以直接带来更快的迭代，并能够处理更大的数据集。有多种技术可以用于加快代码运行速度并减少内存占用。

### 使用向量 (vector)化

优化 NumPy 和 Pandas 代码的最重要方法是**向量化 (quantization)**。向量化涉及将计算表示为对整个数组或 Series/DataFrames 的操作，而不是使用显式 Python 循环逐个元素迭代。

NumPy 的核心优势在于其通用函数 (ufuncs)。这些函数对 `ndarray` 对象进行逐元素操作，并在底层执行高度优化的 C 或 Fortran 代码。当你编写 `c = a + b`（其中 `a` 和 `b` 是 NumPy 数组）时，NumPy 不会在 Python 中逐元素循环。相反，它通过一个快速的编译循环完成加法运算。

考虑添加两个大型数组：

```python
import numpy as np
import time

size = 1_000_000
a = np.random.rand(size)
b = np.random.rand(size)

# 低效的基于循环的方法
start_loop = time.time()
c_loop = np.zeros(size)
for i in range(size):
    c_loop[i] = a[i] + b[i]
end_loop = time.time()
print(f"Python 循环耗时: {end_loop - start_loop:.6f} 秒")

# 高效的向量化方法
start_vec = time.time()
c_vec = a + b # 使用 NumPy 的向量化 '+' ufunc
end_vec = time.time()
print(f"向量化耗时: {end_vec - start_vec:.6f} 秒")

# >> Python 循环耗时: 0.178451 秒 # 示例计时
# >> 向量化耗时: 0.002135 秒   # 示例计时
```

差异巨大。向量化操作快了几个数量级，因为它避免了 Python 对每个元素进行解释循环的开销，并使用了预编译的优化代码。Pandas 操作通常构建在 NumPy 数组之上，因此同样的原则适用。总是寻找方法，将作用于数组或 Series 元素的 Python 循环替换为等效的向量化 NumPy 或 Pandas 函数（例如，算术运算符、比较运算符、数学函数如 `np.log`、`np.exp`、`df.abs()`）。

### 使用高效索引和选择

在 Pandas DataFrame 中访问和修改数据的方式会显著影响性能。虽然使用 `df.iterrows()` 等方法逐行迭代可能看起来直观，但它通常非常慢，因为它涉及为每一行创建一个新的 Series 对象。

优先使用向量 (vector)化布尔索引或基于标签/位置的 `.loc` 和 `.iloc` 索引。

- **布尔索引：** `df[df['column'] > value]` 对于基于条件筛选行非常高效。
- **`.loc`：** 通过标签（索引和列名）选择数据。`df.loc[row_labels, column_labels]`
- **`.iloc`：** 通过整数位置选择数据。`df.iloc[row_indices, column_indices]`

比较使用迭代和布尔索引进行筛选：

```python
import pandas as pd
import numpy as np
import time

# 示例 DataFrame
df = pd.DataFrame({
    'value': np.random.randint(0, 100, size=500_000),
    'category': np.random.choice(['X', 'Y', 'Z'], size=500_000)
})

# 使用 .iterrows() 的慢速迭代
start_iter = time.time()
selected_iter = []
for index, row in df.iterrows():
    if row['value'] > 50 and row['category'] == 'X':
        selected_iter.append(index)
result_iter = df.loc[selected_iter]
end_iter = time.time()
print(f".iterrows() 耗时: {end_iter - start_iter:.6f} 秒")

# 快速布尔索引
start_bool = time.time()
condition = (df['value'] > 50) & (df['category'] == 'X')
result_bool = df[condition]
end_bool = time.time()
print(f"布尔索引耗时: {end_bool - start_bool:.6f} 秒")

# >> .iterrows() 耗时: 4.351201 秒       # 示例计时
# >> 布尔索引耗时: 0.015978 秒 # 示例计时
```

同样，向量化 (quantization)方法（布尔索引）显示出明显更好的性能。当通过标签或位置选择特定行/列时，使用 `.loc` 或 `.iloc`；并使用布尔掩码进行条件筛选。

### 使用合适的数据类型优化内存占用

NumPy 和 Pandas 通常默认使用 64 位整数 (`int64`) 和浮点数 (`float64`)。虽然这些提供了高精度和广范围，但它们可能消耗比你特定数据所需更多的内存。如果你的整数值相对较小，或者你不需要 64 位浮点的全部精度，你通常可以将这些类型**降级**为更小的变体，如 `int32`、`int16`、`float32` 等。这可以带来大量内存节省，尤其对于大型数据集，进而可以加速计算，因为更多数据可以放入 CPU 缓存。

使用 `df.info(memory_usage='deep')` 来检查 DataFrame 的内存占用。你可以使用 `.astype()` 方法更改数据类型。

```python
# 假设 'data' 是上一节的 DataFrame
print("---
--- 原始内存占用 ---")
data.info(memory_usage='deep')

# 降级数据类型
data_optimized = data.copy()
data_optimized['value'] = data_optimized['value'].astype('int16') # 最大值为 99，int16 足以容纳
# 假设浮点精度允许使用 float32
# data_optimized['some_float_col'] = data_optimized['some_float_col'].astype('float32')

print("\n--- 优化后的内存占用 (整数) ---")
data_optimized.info(memory_usage='deep')
```

另一个重要的优化是在 Pandas 中对具有有限数量唯一值（低基数）的字符串列使用 **`category`** 数据类型。在内部，Pandas 使用映射到唯一字符串值的整数代码表示分类数据。这比存储重复字符串的内存效率高得多。

```python
# 继续使用前面的例子
mem_before_cat = data_optimized.memory_usage(deep=True).sum()

# 将 'category' 列转换为分类类型
data_optimized['category'] = data_optimized['category'].astype('category')

mem_after_cat = data_optimized.memory_usage(deep=True).sum()

print("\n---
--- 优化后的内存占用 (分类) ---")
data_optimized.info(memory_usage='deep')

print(f"\n转换为分类类型前内存: {mem_before_cat / (1024**2):.2f} MB")
print(f"转换为分类类型后内存:  {mem_after_cat / (1024**2):.2f} MB")
```



[交互图表：DataFrame 内存占用优化](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-6-efficient-maintainable-python-ml/optimizing-numpy-pandas#plot-1n6gg75)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "DataFrame 内存占用优化",
    "xaxis": {
      "title": "数据类型策略"
    },
    "yaxis": {
      "title": "内存占用 (MB)"
    },
    "barmode": "group"
  },
  "data": [
    {
      "type": "bar",
      "name": "原始",
      "x": [
        "整数",
        "字符串/对象"
      ],
      "y": [
        3.81,
        31.4
      ],
      "marker": {
        "color": "#fa5252"
      }
    },
    {
      "type": "bar",
      "name": "优化 (整数降级)",
      "x": [
        "整数",
        "字符串/对象"
      ],
      "y": [
        0.95,
        31.4
      ],
      "marker": {
        "color": "#ff922b"
      }
    },
    {
      "type": "bar",
      "name": "优化 (整数降级 + 分类)",
      "x": [
        "整数",
        "字符串/对象"
      ],
      "y": [
        0.95,
        0.48
      ],
      "marker": {
        "color": "#51cf66"
      }
    }
  ]
}
```

</details>



> 50 万行 DataFrame 在应用整数降级和分类转换前后的内存使用对比。实际节省取决于数据特性。

选择合适的数据类型是高效管理内存资源的简单而有效的方法。

### 避免不必要的副本（视图与副本）

NumPy 和 Pandas 操作有时返回原始数据的**视图**，有时返回**副本**。视图与原始数组或 DataFrame 共享相同的底层数据缓冲区。修改视图会修改原始对象。副本是一个全新的对象，拥有自己的数据缓冲区。

这种区别很重要。如果你修改了你认为是临时子集的东西，而它是一个视图，就可能无意中更改你的主要数据集。相反，如果你操作的是副本，而期望修改会影响原始数据，则会导致错误。

Pandas 试图通过 `SettingWithCopyWarning` 警告你可能存在的模糊情况。这通常在**链式索引**时出现，例如 `df[condition]['column'] = value`。不清楚第一个选择 (`df[condition]`) 返回的是视图还是副本。如果它是一个副本，则随后的赋值 (`['column'] = value`) 会修改这个临时副本，然后该副本被丢弃，原始 `df` 保持不变。

为避免模糊不清并确保你的赋值按预期工作：

1. 使用 `.loc` 进行基于标签/条件的赋值：`df.loc[condition, 'column'] = value`。这直接操作原始 DataFrame。
2. 如果你确实需要切片或筛选后的 DataFrame 的独立副本以进行单独操作，请显式使用 `.copy()` 方法：`subset = df[df['column'] > value].copy()`。

理解和管理视图与副本可避免一些不易察觉的错误，并确保数据操作可预测。

### 使用内置函数

NumPy 和 Pandas 为聚合（`sum`、`mean`、`median`、`std`、`count`）、转换（`cumsum`、`cumprod`）等常见操作提供了丰富的内置函数集。这些函数几乎总是用优化的 C 代码实现，并运用向量 (vector)化。

当存在向量化 (quantization)替代方案时，优先使用这些内置函数，而不是编写自己的 Python 循环或可能较慢的方法，如 `apply`。

- **聚合操作：** `df.sum()`、`df.mean()`、`df.groupby('group_col').agg(['mean', 'std'])` 都高度优化。
- **转换操作：** `df['col'].cumsum()`、`df.rank()`。

Pandas 中的 `apply` 方法（`df.apply(func, axis=...)`）可用于逐行或逐列应用自定义函数。然而，请注意 `apply` 通常在幕后涉及迭代，如果存在完全向量化的方法，这会慢得多。优先考虑向量化选项，然后再使用 `apply`。如果必须使用 `apply`，尽量确保所应用的函数本身在内部也使用了向量化操作。

通过持续应用这些方法，专注于向量化、高效索引、合适的数据类型、审慎的副本/视图处理以及内置函数，你可以显著提升 NumPy 和 Pandas 代码在机器学习 (machine learning)工作流中的性能，并减少内存占用。这会带来更快的开发周期，以及有效处理更大、更复杂数据集的能力。

## 参考资料

- [The N-dimensional array (ndarray) and Universal functions (ufunc)](https://numpy.org/doc/stable/reference/arrays.ndarray.html) — NumPy Developers (2024)
  解释NumPy核心数据结构和高度优化矢量化操作的官方文档。
- [User Guide - Enhancing performance](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancing_performance.html) — Pandas Developers (2020)
  详细说明优化Pandas代码技术的官方指南，包括索引、数据类型和避免复制。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 579
  Pandas创建者撰写的全面指南，涵盖NumPy和Pandas的高效数据操作和计算。

---

[上一节](05-Python%20%E4%BB%A3%E7%A0%81%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90.md) · [下一节](07-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E4%BB%8B%E7%BB%8D.md)
