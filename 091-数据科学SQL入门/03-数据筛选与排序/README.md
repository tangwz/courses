# 第 3 章：数据筛选与排序

来源：[原章节](https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data)

[返回课程目录](../README.md)

从表中获取*所有*数据往往只是第一步。为了进行有意义的分析，你通常需要筛选出符合特定条件的数据子集，或者以特定顺序显示结果。本章将侧重讲解这些基本的数据优化技巧。

你会学习如何使用 `WHERE` 子句，这是根据你定义的条件筛选行的标准 SQL 机制。我们会讲解比较运算符（如 `=`、`>`、`<`）、逻辑运算符（`AND`、`OR`、`NOT`）以及专用运算符（`IN`、`BETWEEN`、`LIKE`）的用法，以处理更复杂的筛选任务，包括如何妥善处理 `NULL` 值。最后，你将学习如何使用 `ORDER BY` 子句来控制查询结果的显示顺序。

## 小节

- 1. [WHERE 子句筛选数据简介](01-WHERE%20%E5%AD%90%E5%8F%A5%E7%AD%9B%E9%80%89%E6%95%B0%E6%8D%AE%E7%AE%80%E4%BB%8B.md)
- 2. [比较运算符 (=, <>, <, >, <=, >=)](02-%E6%AF%94%E8%BE%83%E8%BF%90%E7%AE%97%E7%AC%A6%20%28%3D%2C%20--%2C%20-%2C%20-%2C%20-%3D%2C%20-%3D%29.md)
- 3. [使用 AND、OR、NOT 进行筛选](03-%E4%BD%BF%E7%94%A8%20AND%E3%80%81OR%E3%80%81NOT%20%E8%BF%9B%E8%A1%8C%E7%AD%9B%E9%80%89.md)
- 4. [使用 IN 和 BETWEEN 进行筛选](04-%E4%BD%BF%E7%94%A8%20IN%20%E5%92%8C%20BETWEEN%20%E8%BF%9B%E8%A1%8C%E7%AD%9B%E9%80%89.md)
- 5. [处理 NULL 值](05-%E5%A4%84%E7%90%86%20NULL%20%E5%80%BC.md)
- 6. [使用 LIKE 进行模式匹配](06-%E4%BD%BF%E7%94%A8%20LIKE%20%E8%BF%9B%E8%A1%8C%E6%A8%A1%E5%BC%8F%E5%8C%B9%E9%85%8D.md)
- 7. [使用 ORDER BY 排序结果](07-%E4%BD%BF%E7%94%A8%20ORDER%20BY%20%E6%8E%92%E5%BA%8F%E7%BB%93%E6%9E%9C.md)
- 8. [实战练习：筛选和排序查询](08-%E5%AE%9E%E6%88%98%E7%BB%83%E4%B9%A0%EF%BC%9A%E7%AD%9B%E9%80%89%E5%92%8C%E6%8E%92%E5%BA%8F%E6%9F%A5%E8%AF%A2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/sql-for-data-science/chapter-3-filtering-sorting-data/quiz)
