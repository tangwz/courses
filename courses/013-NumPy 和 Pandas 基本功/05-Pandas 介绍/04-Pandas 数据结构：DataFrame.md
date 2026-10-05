# Pandas 数据结构：DataFrame

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-5-introduction-to-pandas/pandas-dataframe)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Pandas `DataFrame` 是处理表格数据的核心数据结构，这些数据通常由行和列组成。虽然 Pandas 也提供了用于一维带标签数据的 `Series`，但 `DataFrame` 是管理多维数据集的重要工具，可说是 Pandas 中最主要的数据结构。它直接受到 R 编程语言中数据帧概念的启发。

可以将 `DataFrame` 看作一个普适的二维表格，类似于你在 Microsoft Excel 中使用的电子表格，或 SQL 数据库中的表。它旨在以结构化的方式存放数据，方便处理和分析。

以下是 DataFrame 的主要特点：

1. **二维：** 它同时具有行和列，形成网格状结构。
2. **带标签的轴：** 行和列都有标签。行标签统称为 `index`，列标签称为 `columns`。这使得根据这些标签而非仅仅整数位置来获取数据变得更直接。
3. **异构数据类型：** 与通常只存储单一数据类型的 NumPy 数组不同，DataFrame 中的列可以包含不同的数据类型（例如：整数、浮点数、字符串、布尔值、Python 对象）。
4. **大小可变：** 通常，在 DataFrame 创建后，你可以添加或移除列。添加或移除行也是可行的。
5. **与 Series 的关系：** 你可以将 `DataFrame` 理解为 `Series` 对象的字典或集合，其中每个 `Series` 代表一列。`DataFrame` 中的所有 `Series`（列）共享相同的索引（行标签）。

> Pandas DataFrame 的视图，显示行索引标签、列标签（可能包含数据类型）和数据网格。

尽管 `DataFrame` 内部使用 NumPy 数组来提升效率，但它提供了更灵活和富有表现力的接口来处理结构化数据。它在操作期间自动处理数据对齐 (alignment)，并提供精巧的方法进行索引、切片、重塑、合并和处理缺失信息。这使其成为数据清理、数据检查和分析任务中不可或缺的工具，这些任务在数据科学和 AI 工作流程中很常见。

接下来的章节将会说明如何从各种数据源创建这些多用途的 `DataFrame` 对象，以及如何开始了解其内容。

## 参考资料

- [Data structures - pandas 2.3.0 documentation](https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe) — pandas development team (2024)
  提供对Pandas DataFrame结构的官方介绍和详细说明。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython, 3rd Edition](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由Pandas的创建者撰写，提供了关于DataFrame及其在数据分析中作用的深入指南。
- [R for Data Science, 2nd Edition](https://r4ds.hadley.nz/) — Hadley Wickham and Garrett Grolemund (2023)
  Publisher: O'Reilly Media
  解释了R中的数据框概念，该概念对Pandas DataFrame结构有影响。

---

[上一节](03-%E5%88%9B%E5%BB%BA%20Series.md) · [下一节](05-%E5%88%9B%E5%BB%BA%20DataFrame.md)
