# 第 4 章：数据聚合

来源：[原章节](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data)

[返回课程目录](../README.md)

通常，单行数据所提供的信息不如从中得出的汇总信息有价值。在获取、筛选和排序数据之后，下一步通常是对分组数据计算汇总统计量。本章将介绍 SQL 聚合函数，它们是执行这些计算的工具。

您将学习如何:

*   使用 `COUNT`、`SUM`、`AVG`、`MIN` 和 `MAX` 等常见聚合函数，计算数据中的总和、平均值并找出边界值。
*   使用 `GROUP BY` 子句对具有共同特征的行进行分组，从而可以为每个不同的组计算聚合值。例如，计算每位客户的平均订单总值 $AVG(order\_total)$。
*   使用 `HAVING` 子句根据聚合值筛选这些分组后的结果，该子句在聚合操作完成后执行。

## 小节

- 1. [聚合函数简介](01-%E8%81%9A%E5%90%88%E5%87%BD%E6%95%B0%E7%AE%80%E4%BB%8B.md)
- 2. [使用 COUNT 统计行数](02-%E4%BD%BF%E7%94%A8%20COUNT%20%E7%BB%9F%E8%AE%A1%E8%A1%8C%E6%95%B0.md)
- 3. [使用 SUM 计算总和](03-%E4%BD%BF%E7%94%A8%20SUM%20%E8%AE%A1%E7%AE%97%E6%80%BB%E5%92%8C.md)
- 4. [使用AVG计算平均值](04-%E4%BD%BF%E7%94%A8AVG%E8%AE%A1%E7%AE%97%E5%B9%B3%E5%9D%87%E5%80%BC.md)
- 5. [使用 MIN/MAX 函数查找最小值和最大值](05-%E4%BD%BF%E7%94%A8%20MIN-MAX%20%E5%87%BD%E6%95%B0%E6%9F%A5%E6%89%BE%E6%9C%80%E5%B0%8F%E5%80%BC%E5%92%8C%E6%9C%80%E5%A4%A7%E5%80%BC.md)
- 6. [用 \`GROUP BY\` 进行数据分组](06-%E7%94%A8%20%60GROUP%20BY%60%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%88%86%E7%BB%84.md)
- 7. [使用 HAVING 过滤分组](07-%E4%BD%BF%E7%94%A8%20HAVING%20%E8%BF%87%E6%BB%A4%E5%88%86%E7%BB%84.md)
- 8. [动手实践：聚合与分组](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%81%9A%E5%90%88%E4%B8%8E%E5%88%86%E7%BB%84.md)

章节测验：[在线测验](https://apxml.com/zh/courses/sql-for-data-science/chapter-4-aggregating-data/quiz)
