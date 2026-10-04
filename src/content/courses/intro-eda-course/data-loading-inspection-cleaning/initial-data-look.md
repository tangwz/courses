---
course: "intro-eda-course"
chapter: "data-loading-inspection-cleaning"
lesson: "initial-data-look"
sourceId: 1136
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-2-data-loading-inspection-cleaning/initial-data-look"
title: "数据初步观察：形状、头部、尾部"
description: "使用 Pandas 的 `shape`、`head` 和 `tail` 等功能，快速查看数据集的结构。"
order: 2
plots: []
sourceHash: "cdb48a774044c817d4703e75555a8270709678a4a27cfa04557c577f4b481429"
sourceCorrections: []
---

成功将数据集载入 Pandas DataFrame 后，紧接着要做的就是快速了解其整体结构和内容。这就像打开一本新书；你可能会扫一眼页数，然后读几行字来大致了解。Pandas 提供了方便的属性和方法来完成这项工作。

### 查看数据尺寸：`shape` 属性

在查看实际数据值之前，了解数据集的大小很有用。它包含多少个观测值（行）和多少个特征（列）？DataFrame 的 `shape` 属性返回一个表示尺寸（行、列）的元组。

假设您已将数据载入名为 `df` 的 DataFrame 中：

```python
# 假设 'df' 是您的 Pandas DataFrame
dimensions = df.shape
print(f"数据集有 {dimensions[0]} 行和 {dimensions[1]} 列。")
```

了解形状非常重要。一个拥有数百万行的数据集，可能需要与只有数百行的数据集不同的分析策略或计算资源。同样，列的数量能让您初步了解数据的复杂程度或“宽度”。

### 预览数据：`head()` 和 `tail()`

`shape` 告诉您数据大小，但它不显示实际数据。要快速查看前几行并了解列名及其包含的数据类型，请使用 `head()` 方法。

默认情况下，`head()` 显示前 5 行：

```python
# 显示前 5 行
print("数据集的前 5 行：")
print(df.head())
```

此输出对以下几个方面有帮助：

1. **验证数据载入情况：** 它确认数据载入大致正确，并未完全乱码。
2. **列名：** 您可以看到实际的列名。它们是否具信息量？是否包含以后可能需要清理的空格或特殊字符？
3. **数据值：** 您可以大致看到每列中的值。数字是否在预期范围内？分类列是否有合理条目？您能否发现明显的例如占位符值的问题？

您也可以通过传入一个整数参数 (parameter)来指定要查看的行数：

```python
# 显示前 10 行
print("数据集的前 10 行：")
print(df.head(10))
```

类似地，`tail()` 方法显示 DataFrame 的*最后*几行。这对于检查文件末尾是否附加了任何汇总行，或者数据在整个文件中是否保持一致性很有用。与 `head()` 类似，它默认显示 5 行，但接受一个整数参数。

```python
# 显示最后 5 行
print("数据集的最后 5 行：")
print(df.tail())

# 显示最后 3 行
print("\n数据集的最后 3 行：")
print(df.tail(3))
```

结合使用 `shape`、`head()` 和 `tail()` 可以快速获得数据集尺寸和内容预览的重要概览。这种初步检查是熟悉数据的重要一步，在此之后可以进行更详细的分析，例如检查数据类型或处理缺失值。

## 参考资料

- [User Guide: Data structure intro - Viewing data](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#viewing-data) — The Pandas Development Team (2023)
  Pandas库的官方文档，解释了`head()`、`tail()`和`shape`方法的使用。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Python和Pandas进行数据分析的基础著作，由该库的创建者撰写。其中包含了初始数据检查方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的机器学习指南，包含在项目背景下使用Pandas方法进行数据加载和初步探索的章节。
