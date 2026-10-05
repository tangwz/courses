# 第 9 章：数据分组与聚合

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas)

[返回课程目录](../README.md)

在学习了Pandas的数据加载、清洗和选择之后，我们现在将转向一种数据分析的基本方法：按类别汇总数据。本章主要介绍分组操作。

您将学习使用“分拆-应用-组合”方法，这是数据聚合的一种常见模式。具体来说，您将学习：

*   使用`groupby()`方法根据列值对DataFrame进行分段。
*   对这些分段应用`sum()`、`mean()`、`count()`、`min()`和`max()`等聚合函数。
*   使用`.agg()`方法同时执行多个聚合操作。
*   根据多列对数据进行分组，以创建分层汇总。
*   在应用`groupby()`之后访问单个组及其相关数据。

学完本章后，您将能够有效地分段处理数据，并计算出DataFrame中不同组的有意义的汇总统计量。

## 小节

- 1. [拆分-应用-组合方法](01-%E6%8B%86%E5%88%86-%E5%BA%94%E7%94%A8-%E7%BB%84%E5%90%88%E6%96%B9%E6%B3%95.md)
- 2. [使用 groupby() 方法对数据进行分组](02-%E4%BD%BF%E7%94%A8%20groupby%28%29%20%E6%96%B9%E6%B3%95%E5%AF%B9%E6%95%B0%E6%8D%AE%E8%BF%9B%E8%A1%8C%E5%88%86%E7%BB%84.md)
- 3. [应用聚合函数](03-%E5%BA%94%E7%94%A8%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0.md)
- 4. [应用多种聚合操作](04-%E5%BA%94%E7%94%A8%E5%A4%9A%E7%A7%8D%E8%81%9A%E5%90%88%E6%93%8D%E4%BD%9C.md)
- 5. [按多列分组](05-%E6%8C%89%E5%A4%9A%E5%88%97%E5%88%86%E7%BB%84.md)
- 6. [遍历分组](06-%E9%81%8D%E5%8E%86%E5%88%86%E7%BB%84.md)
- 7. [动手实践：使用 GroupBy 汇总数据](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20GroupBy%20%E6%B1%87%E6%80%BB%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/quiz)
