---
course: "intro-eda-course"
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis"
sourceId: 422
chapter: "univariate-analysis"
title: "单变量分析：理解单个变量"
order: 3
description: "分析数据集中的单个变量，使用描述性统计量和直方图、箱线图等可视化图表。"
hasQuiz: true
---

完成数据集加载并进行了初步检查后，下一步是检查单个变量的特征。本章重点介绍单变量分析，这种方法一次只分析一个变量，以了解其潜在的分布、集中趋势和离散程度。

您将学习如何计算和解释数值变量的描述性统计量，包括集中趋势度量（均值、中位数、众数）和离散程度度量（方差、标准差 $ \sigma $、极差、四分位距或IQR）。我们将使用Matplotlib和Seaborn创建的直方图和箱线图等可视化图表，以图形方式呈现这些分布并识别潜在的异常值。

对于分类变量，重点将转向理解频率计数和比例。您将学习使用Pandas计算这些汇总统计量，并使用条形图有效进行可视化。我们还将介绍基本的统计方法，用于识别数值数据中的潜在异常值，例如使用Z分数，计算公式为：
$$ Z = \frac{x - \mu}{\sigma} $$
或应用IQR规则。

完成本章后，您将能够系统地汇总和可视化单个变量的属性，这是任何数据分析过程中的一个基本步骤。
