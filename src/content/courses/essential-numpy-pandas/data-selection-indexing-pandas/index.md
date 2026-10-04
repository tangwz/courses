---
course: "essential-numpy-pandas"
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-7-data-selection-indexing-pandas"
sourceId: 507
chapter: "data-selection-indexing-pandas"
title: "Pandas 中的数据选取与索引"
order: 7
description: "掌握使用列名、行标签（.loc）、整数位置（.iloc）和布尔条件从 Pandas DataFrame 中选取数据。"
hasQuiz: true
---

在学会如何将数据载入 Pandas DataFrame 后，接下来重要的技能就是从中获取特定信息。通常，您不需要整个数据集；您需要的是特定的列、行，或满足某些条件的子集。

本章着重介绍在 Pandas Series 和 DataFrame 中选取数据的主要方法。您将学会如何：

*   根据名称选取列。
*   使用 `.loc` 访问器，通过标签访问行和列。
*   使用 `.iloc` 访问器，通过整数位置访问行和列。
*   根据条件筛选数据（布尔索引）。
*   设置和重置 DataFrame 索引，以更有效地访问数据。

掌握这些选取方法对于为数据分析和操作任务做准备是必不可少的。我们将讲解每种方法的语法和常见用法。
