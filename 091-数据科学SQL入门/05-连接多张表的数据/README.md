# 第 5 章：连接多张表的数据

来源：[原章节](https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables)

[返回课程目录](../README.md)

用于分析的数据很少存储在一张单一的表中。出于组织性和效率的考虑，信息通常分散在多张关联的表中。因此，一项常见任务是将来自这些不同表的数据汇集起来，以获得完整的信息。

本章将介绍使用SQL连接多张表数据的方法。我们将从了解连接表的结构元素开始：主键和外键。接下来您将学习 `SQL JOIN` 操作，侧重于 `INNER JOIN` 子句，根据匹配条件合并行。我们将介绍连接两张表，并将这种方法推广到连接多张表，以及使用表别名来简化查询语法。完成本章学习后，您将明白如何编写查询，以整合来自各种关联表的数据，从而进行全面分析。

## 小节

- 1. [为何合并数据？](01-%E4%B8%BA%E4%BD%95%E5%90%88%E5%B9%B6%E6%95%B0%E6%8D%AE%EF%BC%9F.md)
- 2. [理解主键和外键](02-%E7%90%86%E8%A7%A3%E4%B8%BB%E9%94%AE%E5%92%8C%E5%A4%96%E9%94%AE.md)
- 3. [SQL JOINs 简介](03-SQL%20JOINs%20%E7%AE%80%E4%BB%8B.md)
- 4. [使用 INNER JOIN](04-%E4%BD%BF%E7%94%A8%20INNER%20JOIN.md)
- 5. [连接多个表](05-%E8%BF%9E%E6%8E%A5%E5%A4%9A%E4%B8%AA%E8%A1%A8.md)
- 6. [使用表的别名](06-%E4%BD%BF%E7%94%A8%E8%A1%A8%E7%9A%84%E5%88%AB%E5%90%8D.md)
- 7. [实践操作：编写 INNER JOIN 查询](07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E7%BC%96%E5%86%99%20INNER%20JOIN%20%E6%9F%A5%E8%AF%A2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/sql-for-data-science/chapter-5-combining-data-multiple-tables/quiz)
