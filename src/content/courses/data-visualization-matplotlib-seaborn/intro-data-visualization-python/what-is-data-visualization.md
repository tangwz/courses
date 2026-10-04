---
course: "data-visualization-matplotlib-seaborn"
chapter: "intro-data-visualization-python"
lesson: "what-is-data-visualization"
sourceId: 1134
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-1-intro-data-visualization-python/what-is-data-visualization"
title: "数据可视化是什么？"
description: "理解数据可视化在数据分析中的定义和作用。"
order: 1
plots: ["plots/1134-0.json"]
sourceHash: "9d171b9c5a3c466be3d75cc36c395e1c2383287a1d6f8bb1e9f9ceed5006f1e0"
sourceCorrections: []
---

其核心是，数据可视化是将原始数据（如存储在表格或数据库中的数字和文本）转化为图形表示的实践。例如图表、图形、地图和仪表盘。它的主要目的是让数据易于理解。

为什么需要费力创建可视化呢？因为原始数据，尤其是在数据量很大时，可能令人难以承受且难以理解。我们的大脑非常擅长处理视觉信息。通过将数据转化为视觉格式，我们能够：

1. **辨别模式：** 识别数字行和列中可能不易察觉的趋势、分组和关联。
2. **理解复杂性：** 更直观地把握复杂数据集及其数据间的关联。
3. **发现异常值：** 快速注意到异常或不寻常的数据点，这些点值得进一步查看。
4. **传达信息：** 高效地与他人分享发现，无论他们的技术背景如何。

设想一个电子表格，它包含某几种产品一年内的月销售数据。找出表现最好的产品或辨别季节性趋势，需要仔细查看和比较。一个精心设计的可视化图表，如显示销售随时间变化的折线图或比较每种产品总销售额的条形图，几乎即刻就能使这些信息显而易见。



![简单产品销售比较](plots/1134-0.json)



> 一个简单的条形图快速显示，产品B的销售额最高，而产品C的销售额最低，这种信息不如直接查看数字[150, 220, 80, 190]那么明显。

数据可视化不仅仅是让数据看起来美观；它是数据分析过程中的一个重要步骤。它既是一种用于了解数据中隐藏信息的工具，也是一种用于清楚且有说服力地传达这些信息的说明工具。从显示股价变化的简单折线图到说明网站用户行为的复杂热力图，可视化将抽象的数字转化为具体的见解。

在本课程中，您将学习如何使用Python以及Matplotlib和Seaborn库，直接从您的数据生成各种有信息且有效的可视化图表。

## 参考资料

- [Information Visualization: Perception for Design, Fourth Edition](https://www.elsevier.com/books/information-visualization/ware/978-0-12-812875-6) — Colin Ware (2020)
  Publisher: Morgan Kaufmann
  详细介绍了与数据可视化设计相关的视觉感知和认知原理。
- [Python for Data Analysis, 3rd Edition](https://wesmckinney.com/book) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用Python（包括Matplotlib和Seaborn）进行数据处理和可视化的标准参考书。
