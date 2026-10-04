# 第 10 章：组合 DataFrames

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas)

[返回课程目录](../README.md)

数据通常存在于多个文件或结构中。为了有效进行分析，你经常需要将这些独立的数据集合并起来。例如，你可能在某个表中存储用户信息，而在另一个表中存储他们的活动日志，这时就需要将它们关联起来以获取完整视图。

本章主要介绍 Pandas 提供的用于组合 `DataFrame` 对象的方法。我们将讲解两种主要方式：

1.  **连接操作 (Concatenation):** 使用 `pd.concat` 函数将 DataFrames 纵向或横向堆叠。
2.  **合并/连接 (Merging/Joining):** 执行数据库风格的连接操作，使用 `pd.merge` 函数和 `.join` 方法根据公共列或索引标签来组合 DataFrames。

你将学习不同类型的连接操作（内连接、外连接、左连接、右连接）如何影响最终组合成的 `DataFrame`，以及如何根据数据结构和分析目标有效运用这些操作。

## 小节

- 1. [数据合并介绍](01-%E6%95%B0%E6%8D%AE%E5%90%88%E5%B9%B6%E4%BB%8B%E7%BB%8D.md)
- 2. [连接 DataFrames (pd.concat)](02-%E8%BF%9E%E6%8E%A5%20DataFrames%20%28pd.concat%29.md)
- 3. [数据库风格的合并 (pd.merge)](03-%E6%95%B0%E6%8D%AE%E5%BA%93%E9%A3%8E%E6%A0%BC%E7%9A%84%E5%90%88%E5%B9%B6%20%28pd.merge%29.md)
- 4. [理解合并类型（连接）](04-%E7%90%86%E8%A7%A3%E5%90%88%E5%B9%B6%E7%B1%BB%E5%9E%8B%EF%BC%88%E8%BF%9E%E6%8E%A5%EF%BC%89.md)
- 5. [基于索引的合并](05-%E5%9F%BA%E4%BA%8E%E7%B4%A2%E5%BC%95%E7%9A%84%E5%90%88%E5%B9%B6.md)
- 6. [基于索引的合并 (.join)](06-%E5%9F%BA%E4%BA%8E%E7%B4%A2%E5%BC%95%E7%9A%84%E5%90%88%E5%B9%B6%20%28.join%29.md)
- 7. [实践操作：组合数据集](07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E7%BB%84%E5%90%88%E6%95%B0%E6%8D%AE%E9%9B%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas/quiz)
