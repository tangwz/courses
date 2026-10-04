# 第 8 章：使用 Pandas DataFrame 绘图

来源：[原章节](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes)

[返回课程目录](../README.md)

迄今为止，你已经学会使用 Matplotlib 和 Seaborn 结合基本的 Python 数据结构或 NumPy 数组来创建各种图表。然而，用于分析的数据通常存在于外部文件，例如电子表格或 CSV 文件中。Pandas 库是 Python 中加载、管理和处理此类结构化数据的标准工具，主要通过其 DataFrame 对象来实现。

本章着重于连接 Pandas 数据处理与 Matplotlib 和 Seaborn 可视化之间的环节。你将学会如何：

*   将常见文件格式的数据加载到 Pandas DataFrame 中。
*   运用直接内置于 Pandas DataFrame 的基本绘图方法。
*   将 Pandas DataFrame 有效地整合到 Matplotlib 和 Seaborn 函数中，直接传递数据列以创建可视化。
*   在绘图前，使用 Pandas 执行基本的数据筛选和准备步骤。

直接使用 DataFrame 简化了绘图过程，尤其是在处理包含多个变量的带标签数据集时。

## 小节

- 1. [Pandas Series 和 DataFrame 简要介绍](01-Pandas%20Series%20%E5%92%8C%20DataFrame%20%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D.md)
- 2. [将数据加载到DataFrame](02-%E5%B0%86%E6%95%B0%E6%8D%AE%E5%8A%A0%E8%BD%BD%E5%88%B0DataFrame.md)
- 3. [直接从DataFrame进行基础绘图](03-%E7%9B%B4%E6%8E%A5%E4%BB%8EDataFrame%E8%BF%9B%E8%A1%8C%E5%9F%BA%E7%A1%80%E7%BB%98%E5%9B%BE.md)
- 4. [使用 Matplotlib 配合 DataFrame 列](04-%E4%BD%BF%E7%94%A8%20Matplotlib%20%E9%85%8D%E5%90%88%20DataFrame%20%E5%88%97.md)
- 5. [使用 Seaborn 和 DataFrames](05-%E4%BD%BF%E7%94%A8%20Seaborn%20%E5%92%8C%20DataFrames.md)
- 6. [数据筛选与绘图准备](06-%E6%95%B0%E6%8D%AE%E7%AD%9B%E9%80%89%E4%B8%8E%E7%BB%98%E5%9B%BE%E5%87%86%E5%A4%87.md)
- 7. [动手实践：从文件中进行数据可视化](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E6%96%87%E4%BB%B6%E4%B8%AD%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/quiz)
