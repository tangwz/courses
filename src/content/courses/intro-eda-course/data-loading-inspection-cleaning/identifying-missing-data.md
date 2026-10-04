---
course: "intro-eda-course"
chapter: "data-loading-inspection-cleaning"
lesson: "identifying-missing-data"
sourceId: 1142
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/identifying-missing-data"
title: "处理缺失数据：识别"
description: "检测DataFrame中缺失值(NaN)的存在、数量和分布的方法。"
order: 4
plots: ["plots/1142-0.json"]
sourceHash: "b0e3c2ca641b8a1875c766f8938ccfb7c8e0b0ac2965edd43499b90f412a61e8"
sourceCorrections: []
---

载入数据并对其结构、维度和数据类型进行初步检查后，数据清洗的下一个实际步骤是处理缺失数据。数据集很少是完整的，在您决定采取合适的处理策略之前，了解数据在哪里以及如何缺失是很必要的。如果忽略缺失值，它们可能扭曲统计汇总、使某些算法失效，并导致错误的结论。

### Pandas中什么是缺失数据？

Pandas主要使用特殊的浮点值`NaN`（Not a Number，非数值）来表示缺失数据。此值源自NumPy，并在Python的许多数据科学工具中是标准用法。需要认识到`NaN`的行为有些特别。例如，`NaN`不被认为等于自身（`np.nan == np.nan` 的结果是 `False`）。

Python内置的`None`对象也可能出现在DataFrame中，通常是在`object`数据类型的列里。旨在检测缺失数据的Pandas函数通常能同时处理`NaN`和`None`。有时，数据集可能使用自定义占位符来表示缺失值（如'?'、'missing'、999、-1）。为了让标准的Pandas方法正常工作，这些占位符需要在加载数据时或加载后转换为`NaN`（通常使用`pd.read_csv`中的`na_values`参数 (parameter)或`.replace()`方法）。

### 检测缺失值

Pandas提供了直接的方法来检测缺失值。`.isnull()`方法返回一个与原始DataFrame形状相同的布尔型DataFrame，其中`True`表示缺失值（`NaN`或`None`），`False`表示非缺失值。

```python
import pandas as pd
import numpy as np

# 示例DataFrame（假设它已加载为 'df'）
data = {'StudentID': [101, 102, 103, 104, 105],
        'Score': [85, np.nan, 77, 92, 88],
        'Grade': ['B', 'C', np.nan, 'A', 'B'],
        'Attendance': [0.95, 0.80, 0.85, np.nan, 0.92]}
df = pd.DataFrame(data)

# 检查缺失值
print(df.isnull())
```

在我们的示例`df`上运行`df.isnull()`会生成：

```
   StudentID  Score  Grade  Attendance
0      False  False  False       False
1      False   True  False       False
2      False  False   True       False
3      False  False  False        True
4      False  False  False       False
```

还有一个等效的方法`.isna()`，它执行的功能完全相同。反之，`.notnull()`（及其等效的`.notna()`）对于非缺失值返回`True`，对于缺失值返回`False`。

### 统计缺失值

虽然布尔型DataFrame很有用，但你通常需要汇总信息。对`.isnull()`的结果应用`.sum()`方法，会按列聚合`True`值（这些值被视为1），从而得到每列中缺失值的总数。

```python
# 统计每列的缺失值
missing_counts = df.isnull().sum()
print(missing_counts)
```

这将输出：

```
StudentID     0
Score         1
Grade         1
Attendance    1
dtype: int64
```

这立即告诉我们，'Score'、'Grade'和'Attendance'列各有一个缺失值，而'StudentID'列没有。

为了获取整个DataFrame中缺失值的总数，你可以再次应用`.sum()`：

```python
# DataFrame中缺失值的总数
total_missing = df.isnull().sum().sum()
print(f"Total missing values in the DataFrame: {total_missing}")
# 输出: DataFrame中缺失值的总数: 3
```

### 计算缺失数据的百分比

计数很有用，但了解缺失数据的比例通常更有助于分析，尤其在大型数据集中。在包含数百万条目的列中，少量缺失值可能微不足道，而在仅有数百条目的列中，相同数量的缺失值可能就值得注意了。

你可以通过将`.isnull().sum()`得到的每列缺失值计数除以总行数（使用`len(df)`或`df.shape[0]`获取），再乘以100来计算每列缺失值的百分比。

```python
# 计算每列缺失值的百分比
total_rows = len(df)
missing_percentage = (df.isnull().sum() / total_rows) * 100
print(missing_percentage)
```

对于我们的示例`df`：

```
StudentID      0.0
Score         20.0
Grade         20.0
Attendance    20.0
dtype: float64
```

这表明'Score'、'Grade'和'Attendance'列中20%的条目缺失。

### 缺失值数量的可视化

可视化可以快速了解缺失数据的程度，尤其在处理多列数据时。一个显示每列缺失值数量或百分比的简单条形图很有效。

你可以这样使用Plotly创建一个缺失百分比的条形图：

```python
# （计算missing_percentage的代码如上所示）

# 创建一个Plotly条形图（如果需要，请替换为实际图表生成代码）
# 使用计算出的 'missing_percentage' Series 的示例
```



![缺失值在每列中的百分比](plots/1142-0.json)



> 示例DataFrame中每列缺失值的百分比。没有缺失数据的列显示0%。

对于更复杂的数据集，可视化缺失数据的*模式*（例如，使用Seaborn中的热图，如`sns.heatmap(df.isnull(), cbar=False)`）可以显示缺失值是否倾向于在不同列的相同行中出现，这可能表明数据收集中存在潜在的关联或系统性问题。

识别数据缺失的位置和数量是必要的首要步骤。有了这些了解，你就可以继续学习下一节，该节将讨论处理这些已识别缺失值的策略，例如删除它们或使用插补技术填充它们。

## 参考资料

- [Working with missing data](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html) — pandas development team (2024)
  Pandas处理缺失值（NaN，None）的官方指南，包含isnull()和isna()等检测方法。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Pandas数据操作的基础书籍，涵盖缺失数据的识别和处理。
- [seaborn.heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) — Michael Waskom and the Seaborn development team (2024)
  创建热图的官方文档，对使用df.isnull()可视化DataFrame中缺失数据模式很有帮助。
- [Data Cleaning: A Guide to Creating Clean Data for Effective Analysis and Governance](https://www.oreilly.com/library/view/measuring-data-quality/9780123977540/) — Laura Sebastian-Coleman (2012)
  Publisher: Morgan Kaufmann; Pages: 376
  提供数据清理的原则和方法，包含数据集中缺失值的识别和重要性。
