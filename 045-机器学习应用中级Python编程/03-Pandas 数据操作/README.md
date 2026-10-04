# 第 3 章：Pandas 数据操作

来源：[原章节](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas)

[返回课程目录](../README.md)

在学习了NumPy的数值计算之后，我们现在开始关注结构化数据的管理和处理，这是任何机器学习项目中的一项基本任务。实际数据很少是干净的或为分析而完美格式化的。本章介绍Pandas库，它是Python中用于数据规整的标准工具。

您将了解Pandas的核心数据结构：一维的 `Series` 和二维的 `DataFrame`，它们提供了处理表格数据的强大且灵活的方法。我们将介绍以下主要操作：

*   从各种文件格式（例如CSV、Excel和SQL数据库）加载数据。
*   使用索引方法（例如`.loc`和`.iloc`）选择数据子集。
*   识别和处理缺失值的方法。
*   清洗、转换和重塑数据集的方法。
*   使用`groupby`进行分组分析和聚合。
*   通过合并、连接和拼接组合来自多个来源的数据。
*   有效地处理时间序列数据。

学习完本章后，您将能够使用Pandas高效地准备各种数据集，以进行分析和机器学习模型构建。

## 小节

- 1. [Pandas数据结构简介](01-Pandas%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E7%AE%80%E4%BB%8B.md)
- 2. [从多种来源加载数据](02-%E4%BB%8E%E5%A4%9A%E7%A7%8D%E6%9D%A5%E6%BA%90%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE.md)
- 3. [数据索引与选择](03-%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95%E4%B8%8E%E9%80%89%E6%8B%A9.md)
- 4. [处理缺失数据](04-%E5%A4%84%E7%90%86%E7%BC%BA%E5%A4%B1%E6%95%B0%E6%8D%AE.md)
- 5. [数据清洗与数据转换技巧](05-%E6%95%B0%E6%8D%AE%E6%B8%85%E6%B4%97%E4%B8%8E%E6%95%B0%E6%8D%AE%E8%BD%AC%E6%8D%A2%E6%8A%80%E5%B7%A7.md)
- 6. [分组和聚合操作](06-%E5%88%86%E7%BB%84%E5%92%8C%E8%81%9A%E5%90%88%E6%93%8D%E4%BD%9C.md)
- 7. [合并、连接和拼接数据帧](07-%E5%90%88%E5%B9%B6%E3%80%81%E8%BF%9E%E6%8E%A5%E5%92%8C%E6%8B%BC%E6%8E%A5%E6%95%B0%E6%8D%AE%E5%B8%A7.md)
- 8. [Pandas中的时间序列数据处理](08-Pandas%E4%B8%AD%E7%9A%84%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md)
- 9. [实践：使用 Pandas 整理数据](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20Pandas%20%E6%95%B4%E7%90%86%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/quiz)
