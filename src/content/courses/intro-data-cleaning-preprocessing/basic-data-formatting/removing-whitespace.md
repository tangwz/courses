---
course: "intro-data-cleaning-preprocessing"
chapter: "basic-data-formatting"
lesson: "removing-whitespace"
sourceId: 4030
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-5-basic-data-formatting/removing-whitespace"
title: "去除前导/尾随空格"
description: "学习如何去除文本条目开头或末尾不必要的空格。"
order: 3
plots: []
sourceHash: "a059599dbf316b82df063bc805192662476fa93bbdffc2b73e7cbb73997782b2"
sourceCorrections: []
---

不可见的字符，尤其是文本条目开头或末尾的多余空格，是数据清洗中常见的问题来源。这些前导或尾随空格看似无害，但它们会阻碍准确的匹配、分组和分析。例如，数据库中可能包含“ apple”、“apple ”和“apple”等条目。对于人类读者来说，它们都指代同一事物，但对于执行比较的计算机而言，由于多余空格的存在，它们是不同的字符串。

### 隐藏空格带来的问题

想象一下您正在分析客户数据并希望按城市对销售额进行分组。如果您的数据集中城市列包含“ New York ”、“New York”和“New York ”，简单的分组操作可能会将它们视为三个独立的城市。这会导致结果分散和摘要不准确。同样，如果您尝试将此数据集与另一个基于城市名称的数据集连接，带有额外空格的记录可能无法正确匹配。

这些多余的空格通常在数据录入期间，或在不同系统或文件格式之间导入或导出数据时出现。仅凭肉眼难以察觉它们的存在，这使得它们成为一个隐蔽的错误源。

### 修剪边缘：去除空格

幸运的是，去除这些不必要的前导和尾随空格是一个简单直接的过程。大多数数据处理工具和编程库都提供专门用于“剥离”或“修剪”文本字符串两端空格的功能。此操作会清理条目，同时不影响文本 *内部* 的空格（例如“New York”中的空格）。

我们来看一个使用Python和pandas库（一种常用的数据处理工具）的实际例子。假设我们有一个DataFrame，其中一列包含不一致的城市名称：

```python
import pandas as pd

# 带有空格问题的示例数据
data = {'ID': [1, 2, 3, 4],
        'City': [' London', 'Paris ', ' Berlin ', ' London']}
df = pd.DataFrame(data)

print("原始数据：")
print(df)
```

运行这段代码将显示：

```
Original Data:
   ID       City
0   1    London
1   2     Paris 
2   3    Berlin 
3   4    London 
```

请注意“London”（前导）、“Paris”（尾随）和“Berlin”（两者）周围的多余空格。为了清理“City”列，我们可以对其应用`str.strip()`方法。此方法在对包含字符串的pandas Series调用时，会去除序列中每个字符串开头和末尾的空格。

```python
# 从 'City' 列中去除前导/尾随空格
df['City'] = df['City'].str.strip()

print("\n清理后的数据：")
print(df)
```

输出现在显示清理后的数据：

```
Cleaned Data:
   ID     City
0   1   London
1   2    Paris
2   3   Berlin
3   4   London
```

如您所见，`strip()`方法有效地去除了不必要的空格，使“City”条目保持一致。现在，基于此列进行分组或匹配将按预期工作。

### 变体：左侧和右侧修剪

虽然`str.strip()`去除两端的空格，但pandas也提供`str.lstrip()`来仅去除前导空格（从左侧），以及`str.rstrip()`来仅去除尾随空格（从右侧）。

```python
# 示例：仅去除前导空格
# df['City'] = df['City'].str.lstrip()

# 示例：仅去除尾随空格
# df['City'] = df['City'].str.rstrip()
```

在大多数数据清洗情况下，`str.strip()`是首选方法，因为它同时处理前导和尾随空格。仅当您有特定原因需要保留字符串一侧的空格时，才使用`lstrip`或`rstrip`。

### 何时应用空格去除

此技术通常应用于包含文本或分类数据的列（在pandas中通常存储为“object”或“string”数据类型）。在处理缺失值并确保列具有适当的数据类型之后，但在执行依赖于精确字符串匹配的操作（例如分组、连接或查找唯一值）之前，执行空格修剪是一种良好实践。

通过系统地去除前导和尾随空格，您可以确保间距的变化不会干扰您的分析，有助于获得更整洁、更可靠的数据集。

## 参考资料

- [pandas.Series.str.strip](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.strip.html) — pandas development team (2024)
  官方文档，解释了pandas库中用于数据清洗的字符串去除方法的使用和行为。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于Python数据操作的基础教材，提供了数据预处理的实用指导，包括使用pandas进行字符串操作，对解决数据不一致问题很有帮助。
- [Data Cleaning: A Practical Guide](https://books.google.com/books/about/Data_Cleaning.html?id=z12uDwAAQBAJ) — Ihab F. Ilyas and Xu Chu (2019)
  Publisher: Morgan & Claypool Publishers; DOI: [10.2200/S00910ED1V01Y201903DBS017](https://doi.org/10.2200/S00910ED1V01Y201903DBS017)
  本书系统地概述了数据清洗问题及解决方案，为文本数据标准化提供了背景信息，指出其是数据质量的重要组成部分。
