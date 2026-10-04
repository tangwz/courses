# 第 6 章：Pandas 数据加载与保存

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas)

[返回课程目录](../README.md)

数据分析通常从将数据导入您的分析环境开始。尽管前面章节展示了如何从头创建 Series 和 DataFrame，但实际应用中常需要加载存储在外部文件中的数据。同样地，在处理和分析之后，您通常需要保存您的结果。

本章侧重于 Pandas 的输入/输出（IO）功能。您将学习如何：

*   使用 `pd.read_csv()` 等函数，从常见文件格式，主要是逗号分隔值（CSV）文件中读取数据。
*   使用 `pd.read_excel()`，从 Microsoft Excel 电子表格（.xls 或 .xlsx）中加载数据。
*   简要介绍读取其他可能的数据源。
*   使用 `.to_csv()` 方法，将您的 DataFrame 内容写回 CSV 文件。
*   使用 `.to_excel()` 方法，将 DataFrame 保存到 Excel 文件。

在本章结束时，您将能够处理将数据读取到 Pandas DataFrame 中进行分析以及导出您处理过的数据这些基本任务。

## 小节

- 1. [从CSV文件读取数据](01-%E4%BB%8ECSV%E6%96%87%E4%BB%B6%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md)
- 2. [从Excel文件读取数据](02-%E4%BB%8EExcel%E6%96%87%E4%BB%B6%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md)
- 3. [从其他格式读取数据](03-%E4%BB%8E%E5%85%B6%E4%BB%96%E6%A0%BC%E5%BC%8F%E8%AF%BB%E5%8F%96%E6%95%B0%E6%8D%AE.md)
- 4. [将数据写入 CSV 文件](04-%E5%B0%86%E6%95%B0%E6%8D%AE%E5%86%99%E5%85%A5%20CSV%20%E6%96%87%E4%BB%B6.md)
- 5. [写入数据到Excel文件](05-%E5%86%99%E5%85%A5%E6%95%B0%E6%8D%AE%E5%88%B0Excel%E6%96%87%E4%BB%B6.md)
- 6. [动手实践：数据集的导入与导出](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E5%AF%BC%E5%85%A5%E4%B8%8E%E5%AF%BC%E5%87%BA.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-6-loading-saving-data-pandas/quiz)
