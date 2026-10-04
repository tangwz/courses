---
course: "intro-eda-course"
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis"
sourceId: 424
chapter: "bivariate-analysis"
title: "双变量分析：考察变量间的关系"
order: 4
description: "考察变量对之间的关系，使用散点图、相关分析和比较可视化图表。"
hasQuiz: true
---

在上一章单独分析了变量之后，我们现在将重心转向理解变量对之间如何关联。这一过程，被称为双变量分析，有助于发现数据集中两个不同特征之间可能存在的关联、依赖性或模式。

在本章中，你将学习以下方法：

*   使用散点图来呈现两个数值变量之间的关系。
*   使用相关系数（如皮尔逊相关系数$r$）来衡量数值变量间的线性关联强度和方向。
*   比较数值变量在分类变量不同类别中的分布，使用分组箱线图或小提琴图等比较图表。
*   使用交叉制表（列联表）来分析两个分类变量之间的关联。
*   使用堆叠条形图或分组条形图来呈现两个分类变量之间的关系。

我们将使用 Pandas 等 Python 库进行计算，并使用 Matplotlib/Seaborn 创建富有信息量的可视化图表，以有效地考察这些成对的关系。理解这些交互作用对于形成对数据的直观认识和指导后续建模步骤非常重要。
