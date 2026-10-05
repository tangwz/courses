# 第 7 章：Pandas 中的数据选取与索引

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas)

[返回课程目录](../README.md)

在学会如何将数据载入 Pandas DataFrame 后，接下来重要的技能就是从中获取特定信息。通常，您不需要整个数据集；您需要的是特定的列、行，或满足某些条件的子集。

本章着重介绍在 Pandas Series 和 DataFrame 中选取数据的主要方法。您将学会如何：

*   根据名称选取列。
*   使用 `.loc` 访问器，通过标签访问行和列。
*   使用 `.iloc` 访问器，通过整数位置访问行和列。
*   根据条件筛选数据（布尔索引）。
*   设置和重置 DataFrame 索引，以更有效地访问数据。

掌握这些选取方法对于为数据分析和操作任务做准备是必不可少的。我们将讲解每种方法的语法和常见用法。

## 小节

- 1. [选取列](01-%E9%80%89%E5%8F%96%E5%88%97.md)
- 2. [使用标签选择行 (.loc)](02-%E4%BD%BF%E7%94%A8%E6%A0%87%E7%AD%BE%E9%80%89%E6%8B%A9%E8%A1%8C%20%28.loc%29.md)
- 3. [使用整数位置选择行 (.iloc)](03-%E4%BD%BF%E7%94%A8%E6%95%B4%E6%95%B0%E4%BD%8D%E7%BD%AE%E9%80%89%E6%8B%A9%E8%A1%8C%20%28.iloc%29.md)
- 4. [混合标签和基于位置的索引](04-%E6%B7%B7%E5%90%88%E6%A0%87%E7%AD%BE%E5%92%8C%E5%9F%BA%E4%BA%8E%E4%BD%8D%E7%BD%AE%E7%9A%84%E7%B4%A2%E5%BC%95.md)
- 5. [条件选择 (布尔索引)](05-%E6%9D%A1%E4%BB%B6%E9%80%89%E6%8B%A9%20%28%E5%B8%83%E5%B0%94%E7%B4%A2%E5%BC%95%29.md)
- 6. [设置 DataFrame 索引](06-%E8%AE%BE%E7%BD%AE%20DataFrame%20%E7%B4%A2%E5%BC%95.md)
- 7. [重置 DataFrame 索引](07-%E9%87%8D%E7%BD%AE%20DataFrame%20%E7%B4%A2%E5%BC%95.md)
- 8. [动手实践：访问特定数据子集](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BF%E9%97%AE%E7%89%B9%E5%AE%9A%E6%95%B0%E6%8D%AE%E5%AD%90%E9%9B%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas/quiz)
