---
course: "essential-numpy-pandas"
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas"
sourceId: 505
chapter: "loading-saving-data-pandas"
title: "Pandas 数据加载与保存"
order: 6
description: "学习如何将数据从 CSV 和 Excel 等常见格式读取到 Pandas DataFrame 中，并将 DataFrame 写回文件。"
hasQuiz: true
---

数据分析通常从将数据导入您的分析环境开始。尽管前面章节展示了如何从头创建 Series 和 DataFrame，但实际应用中常需要加载存储在外部文件中的数据。同样地，在处理和分析之后，您通常需要保存您的结果。

本章侧重于 Pandas 的输入/输出（IO）功能。您将学习如何：

*   使用 `pd.read_csv()` 等函数，从常见文件格式，主要是逗号分隔值（CSV）文件中读取数据。
*   使用 `pd.read_excel()`，从 Microsoft Excel 电子表格（.xls 或 .xlsx）中加载数据。
*   简要介绍读取其他可能的数据源。
*   使用 `.to_csv()` 方法，将您的 DataFrame 内容写回 CSV 文件。
*   使用 `.to_excel()` 方法，将 DataFrame 保存到 Excel 文件。

在本章结束时，您将能够处理将数据读取到 Pandas DataFrame 中进行分析以及导出您处理过的数据这些基本任务。
