# 第 6 章：使用 Seaborn 绘制分类数据

来源：[原章节](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data)

[返回课程目录](../README.md)

许多数据并非纯粹是数值型的；它们通常包含产品类型、区域或实验组等类别。可视化这类分类数据中的关系和分布需要特定的方法。本章主要介绍使用 Seaborn 专为有效绘制分类信息而设计的函数。

您将学习创建和解读几种适用于分类分析的图表类型，包括：

*   `barplot` 显示每个类别的汇总统计量（如均值或中位数）。
*   `countplot` 显示每个类别中观测值的频率。
*   `boxplot`、`stripplot` 和 `swarmplot` 用于比较不同类别间的分布，显示单个数据点或汇总统计量。
*   `pointplot` 突出显示类别间点估计值的趋势或差异。

通过本章学习，您将能够选择并生成合适的图表，使用 Seaborn 查看和呈现分类数据。

## 小节

- 1. [理解分类数据](01-%E7%90%86%E8%A7%A3%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE.md)
- 2. [汇总统计条形图 (barplot)](02-%E6%B1%87%E6%80%BB%E7%BB%9F%E8%AE%A1%E6%9D%A1%E5%BD%A2%E5%9B%BE%20%28barplot%29.md)
- 3. [频率计数图 (countplot)](03-%E9%A2%91%E7%8E%87%E8%AE%A1%E6%95%B0%E5%9B%BE%20%28countplot%29.md)
- 4. [分类变量的箱线图 (boxplot)](04-%E5%88%86%E7%B1%BB%E5%8F%98%E9%87%8F%E7%9A%84%E7%AE%B1%E7%BA%BF%E5%9B%BE%20%28boxplot%29.md)
- 5. [散点分布图与分群散点图：独立数据点的呈现 (stripplot, swarmplot)](05-%E6%95%A3%E7%82%B9%E5%88%86%E5%B8%83%E5%9B%BE%E4%B8%8E%E5%88%86%E7%BE%A4%E6%95%A3%E7%82%B9%E5%9B%BE%EF%BC%9A%E7%8B%AC%E7%AB%8B%E6%95%B0%E6%8D%AE%E7%82%B9%E7%9A%84%E5%91%88%E7%8E%B0%20%28stripplot%2C%20swarmplot%29.md)
- 6. [趋势点图 (pointplot)](06-%E8%B6%8B%E5%8A%BF%E7%82%B9%E5%9B%BE%20%28pointplot%29.md)
- 7. [实践操作：分类特征的可视化](07-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%88%86%E7%B1%BB%E7%89%B9%E5%BE%81%E7%9A%84%E5%8F%AF%E8%A7%86%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/quiz)
