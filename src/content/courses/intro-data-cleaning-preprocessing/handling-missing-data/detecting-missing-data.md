---
course: "intro-data-cleaning-preprocessing"
chapter: "handling-missing-data"
lesson: "detecting-missing-data"
sourceId: 3960
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-2-handling-missing-data/detecting-missing-data"
title: "检测缺失数据的方法"
description: "了解基本的编程技巧，以识别缺失数据的存在和位置。"
order: 2
plots: ["plots/3960-0.json"]
sourceHash: "3f3d37da8c6bf8b1582a7ad4be89450b4ea4f748bb323a904f2505e3b9aca7cf"
sourceCorrections: []
---

识别缺失数据的位置并量化 (quantization)其程度，是数据清洗和预处理中的基本步骤。仅仅知道数据缺失是不够的；需要系统的方法来精确确定其发生位置。这通常涉及使用编程工具来检查数据结构，例如表格或DataFrame。

假设我们正在处理已加载到pandas DataFrame中的数据，这是Python中用于数据分析的常见结构。Pandas提供了专门设计用于检测缺失值的便捷函数，它通常将缺失值识别为`NaN`（非数字）或`None`。

### 检查缺失值

检查缺失条目最直接的方法是使用`isnull()`方法（或其相同别名`isna()`）。当应用于pandas DataFrame或Series（单列）时，它不会改变数据本身。相反，它会返回一个相同形状的新对象，其中填充了布尔值：`True`表示该位置的原始值缺失，`False`表示它存在。

```python
import pandas as pd
import numpy as np

# 包含缺失值的示例数据
data = {'Product ID': [101, 102, 103, 104, 105],
        'Category': ['Electronics', 'Apparel', np.nan, 'Electronics', 'Home Goods'],
        'Price': [1200, 55, 250, np.nan, 80],
        'Rating': [4.5, np.nan, 3.8, 4.1, np.nan]}
df = pd.DataFrame(data)

# 检测缺失值
missing_mask = df.isnull()

print(missing_mask)
# 输出:
#    产品ID  类别  价格  评分
# 0       False     False  False   False
# 1       False     False  False    True
# 2       False      True  False   False
# 3       False     False   True   False
# 4       False     False  False    True
```

这种布尔掩码对于查看每个缺失值的确切位置很有用，但对于大型数据集而言，查看大量`True`/`False`值对于总结问题不是很实用。

### 计数缺失值

一个更常见的任务是统计每列中存在多少缺失值。我们可以通过在`.isnull()`之后链式调用`.sum()`方法来实现这一点。当你对布尔值求和时，`True`被视为1，`False`被视为0。对`isnull()`生成的布尔DataFrame应用`.sum()`可以有效地统计每列中`True`值（缺失条目）的数量。

```python
# 统计每列的缺失值
missing_counts = df.isnull().sum()

print(missing_counts)
# 输出:
# 产品ID    0
# 类别      1
# 价格      1
# 评分      2
# dtype: int64
```

这个输出是一个pandas Series，其中索引表示列名，值表示这些列中缺失条目的数量。这立即告知我们，“产品ID”没有缺失值，“类别”和“价格”各有一个，“评分”有两个。

要获得表示整个DataFrame中缺失值总数的一个数字，可以再次应用`.sum()`：

```python
# 整个DataFrame中的缺失值总数
total_missing = df.isnull().sum().sum()

print(f"数据集中缺失值的总数: {total_missing}")
# 输出:
# 数据集中缺失值的总数: 4
```

### 计算缺失值的百分比

计数很有用，但理解缺失数据相对于每列总大小的比例通常能提供更好的背景信息。在一万行的数据集中缺失10个值的列，比在只有50行的列中缺失10个值不那么令人担忧。

我们可以这样计算每列缺失值的百分比：


$$
\text{缺失百分比} = \frac{\text{缺失值数量}}{\text{总行数}} \times 100
$$


在pandas中，这表示为：

```python
# 计算每列缺失值的百分比
total_rows = len(df)
missing_percentage = (df.isnull().sum() / total_rows) * 100

print(missing_percentage)
# 输出:
# 产品ID     0.0
# 类别      20.0
# 价格      20.0
# 评分      40.0
# dtype: float64
```

这表明“类别”和“价格”条目有20%缺失，而“评分”条目有显著的40%缺失。百分比有助于确定哪些列在清洗过程中可能需要更多关注。

### 可视化缺失数据计数

尽管数值摘要很重要，但有时可视化表示可以使缺失数据在各列中的分布更直观。一个显示每列缺失值数量的简单条形图通常很有效。



![每列缺失值计数](plots/3960-0.json)



> 这个条形图可视化了从`df.isnull().sum()`得出的计数，使得比较不同特征间缺失数据的程度变得容易。

这些编程方法结合了布尔检查、计数和百分比，构成了了解数据集中缺失数据范围的基础。一旦你确定了数据缺失的*位置*和*程度*，就可以继续选择并应用适当的策略来处理它，我们将在后续章节中介绍。

## 参考资料

- [pandas.DataFrame.isnull - pandas 2.2.0 documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isnull.html) — pandas development team (2024)
  pandas DataFrame 和 Series 中用于检测缺失值的核心方法的官方文档。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media; Pages: 582
  pandas 创建者撰写的权威指南，全面涵盖数据操作，包括识别缺失数据的实用技术。
- [Hands-On Data Preprocessing with Python: Learn how to effectively prepare your data for successful data science projects](https://www.packtpub.com/product/hands-on-data-preprocessing-with-python/9781838645060) — Sinan Ozdemir (2020)
  Publisher: Packt Publishing
  一本专注于 Python 数据预处理技术的实用指南，其中包含检测和处理缺失数据的方法。
