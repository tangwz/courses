# Pandas 数据结构：Series

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/pandas-series)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然像 NumPy 的 `ndarray` 这样强大的数组结构常用于数值计算，但许多数据不仅仅是原始数字；它们带有标签和结构。例如，随时间变化的股票价格、来自不同位置的传感器读数，或者调查受访者的人口统计信息。这就是 Pandas 发挥作用的地方，它用于一维数据的基本数据结构是 `Series`。

可以将 `Series` 想象成电子表格中的单个列，或者 Python 列表或 NumPy 数组的更高级版本。它本质上是一个一维的类数组对象，包含一系列值和一组相关的数据标签数组，称为其 **索引**。

### Series 的结构

Pandas `Series` 有两个主要组成部分：

1. **值：** 这是一系列数据点。在底层，这些值通常存储在 NumPy `ndarray` 中，这使得对其进行操作既快速又高效。Series 中的值通常具有相同的数据类型（如整数、浮点数、字符串或 Python 对象）。
2. **索引：** 这是一组与值对应的标签数组。与主要使用基于整数的索引（0, 1, 2, ...）的 NumPy 数组不同，Pandas Series 有一个*显式* 索引。此索引可以由整数组成，但也可以由字符串、日期或其他 Python 对象构成。如果您在创建 Series 时未指定索引，Pandas 会自动创建一个默认的整数索引，范围从 0 到 $N-1$，其中 $N$ 是值的数量。

这里是一个简单的可视化表示：

> Pandas Series 将值数组（通常是 NumPy 数组）与一个用于标记 (token)的显式索引对象结合起来。

### 带标签数据的重要性

显式索引是 Pandas Series 的一个重要特性。相比仅使用普通的 NumPy 数组，它提供了几个优势：

- **直观访问：** 您可以使用有意义的标签（如日期或类别名称）访问数据点，而不是仅仅通过整数位置。例如，获取“周三”的温度通常比记住它在索引位置 2 更直观。
- **数据对齐 (alignment)：** 在多个 Series 之间执行操作时，Pandas 会根据索引标签自动对齐数据。这可以防止在使用无序或不同顺序的数据时出现的许多常见错误。
- **灵活性：** 相比简单的整数索引，索引允许更复杂的选择和处理逻辑。

可以将 Series 看作是通过增加这一层有意义的标签来增强 NumPy 数组。它保留了 NumPy 对底层值的计算效率，同时提供了一种更灵活、内容更丰富的结构，适合数据分析。在下一节中，我们将介绍在 Python 中创建这些 `Series` 对象的实用方法。

## 参考资料

- [Pandas Series (Data structures)](https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html#series) — The Pandas Development Team (2023)
  作为 Pandas `Series` 对象的官方且最新的技术参考，涵盖其组成部分和基本概念。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  第3版。对 Pandas `Series` 数据结构及其实际应用提供了全面的说明，由 Pandas 的创建者撰写。

---

[上一节](01-Pandas%20%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md) · [下一节](03-%E5%88%9B%E5%BB%BA%20Series.md)
