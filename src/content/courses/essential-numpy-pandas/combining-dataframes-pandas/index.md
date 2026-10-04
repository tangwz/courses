---
course: "essential-numpy-pandas"
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-10-combining-dataframes-pandas"
sourceId: 514
chapter: "combining-dataframes-pandas"
title: "组合 DataFrames"
order: 10
description: "学习如何使用连接操作（堆叠）以及合并/连接（数据库风格）来组合多个 Pandas DataFrame。"
hasQuiz: true
---

数据通常存在于多个文件或结构中。为了有效进行分析，你经常需要将这些独立的数据集合并起来。例如，你可能在某个表中存储用户信息，而在另一个表中存储他们的活动日志，这时就需要将它们关联起来以获取完整视图。

本章主要介绍 Pandas 提供的用于组合 `DataFrame` 对象的方法。我们将讲解两种主要方式：

1.  **连接操作 (Concatenation):** 使用 `pd.concat` 函数将 DataFrames 纵向或横向堆叠。
2.  **合并/连接 (Merging/Joining):** 执行数据库风格的连接操作，使用 `pd.merge` 函数和 `.join` 方法根据公共列或索引标签来组合 DataFrames。

你将学习不同类型的连接操作（内连接、外连接、左连接、右连接）如何影响最终组合成的 `DataFrame`，以及如何根据数据结构和分析目标有效运用这些操作。
