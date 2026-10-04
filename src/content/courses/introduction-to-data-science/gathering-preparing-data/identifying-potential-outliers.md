---
course: "introduction-to-data-science"
chapter: "gathering-preparing-data"
lesson: "identifying-potential-outliers"
sourceId: 2133
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-4-gathering-preparing-data/identifying-potential-outliers"
title: "识别潜在异常值"
description: "Learn what outliers are and why identifying them might be important during data preparation."
order: 5
plots: ["plots/2133-0.json"]
sourceHash: "9f55f1ae5f9d4c0a2e0142352ae28bbbe2ce2a57e09d043cab2a1568256a107d"
sourceCorrections: []
---

在检查数据集时，有时会遇到看起来格格不入的值，它们远远超出大多数数据点所在的范围。这些不寻常的值通常被称为**异常值**。识别它们是数据准备中的一个重要步骤，因为它们可能对分析结果产生显著影响，并可能导致得出不正确的结论。

设想一个包含典型小学班级学生年龄的数据集。大多数年龄可能在6到8岁之间。如果你突然发现一个记录为75的年龄，这个值就显得非常突出。这个“75”就会被视为一个异常值。

### 为什么我们需要关注异常值？

异常值值得关注的原因有以下几点：

1. **对统计指标的影响：** 许多常见的统计指标，特别是均值（平均数）和标准差（衡量分散程度的指标），对异常值非常敏感。单个极端值可以显著拉高或拉低平均数，从而错误地代表数据集中的典型值。同样，它会夸大标准差，使数据主体间的变异性看起来比实际更大。
2. **对模型的影响：** 当你日后开始构建预测模型时（我们很快会介绍这个话题），异常值有时会不成比例地影响模型学习规律的方式，可能导致在典型数据上的表现不佳。
3. **潜在错误：** 异常值可能表明数据收集或输入中存在错误。小学班级中“75”岁的年龄几乎肯定是一个错误。识别这些异常值可以让你调查并可能修正这些错误。
4. **真实的极端值：** 并非所有异常值都是错误。有时，异常值代表一个真实但罕见的事件。例如，在交易金额的数据集中，一笔非常大的交易可能是一个异常值，但它可能代表一次合法的重大采购，甚至欺诈活动。这些真实的异常值有时是最有意思的数据点，它们能提供独特的看法。

### 识别潜在异常值的简单方法

对于初学者来说，目视检查通常是识别潜在异常值最直观的方法。

#### 目视检查：箱线图

识别异常值最有效的图表之一是**箱线图**（有时也称为箱须图）。箱线图使用四分位数以视觉方式归纳数值变量的分布情况。

- “箱体”代表四分位距（IQR），包含数据中间的50%。箱体内的线是中位数（中间值）。
- “触须”通常从箱体延伸出来，以包含大部分数据。一个常见的做法是将触须延伸到箱体边缘（Q1和Q3）外1.5倍IQR的距离。
- 落在触须*之外*的数据点会单独绘制，并通常被认为是潜在异常值。



![含异常值的样本数据分布](plots/2133-0.json)



> 一个箱线图显示了大部分值聚集在21到28之间的数据集，但一个值（95）远远超出此范围，并被标示为一个单独的点（潜在异常值）。

在上图中，大多数数据点都在合理范围内，形成了箱体和触须。那个远远高于上方触须的单个点立即被目视标记 (token)为潜在异常值。直方图也可能有用；异常值可能表现为远离数据主体孤立的条形。

#### 统计规则（简要提及）

虽然目视检查功能强大，但你可能也会听说定义异常值的统计规则。这些规则通常涉及计算数据点距离分布中心有多远。

- **IQR 方法：** 这将箱线图方法正式化。如果一个点小于 $Q1 - 1.5 \times IQR$ 或大于 $Q3 + 1.5 \times IQR$，则可能被标记，其中Q1是第一四分位数，Q3是第三四分位数，且 $IQR = Q3 - Q1$。
- **Z-Score 方法：** 这衡量一个数据点距离平均数有多少个标准差。一个常见的阈值是认为Z-score大于3或小于-3的点为异常值。

目前，理解异常值远离典型范围的*观念*就足够了。箱线图等目视方法通常隐含地包含这些统计思想。

### 如何处理异常值？

一旦你识别出潜在异常值，下一步需要仔细斟酌。没有一个通用的规则。

1. **调查：** 务必尝试弄清楚异常值的*原因*。是数据输入时的笔误？传感器故障？还是一个真实、极端的值？数据的情境在这里非常重要。
2. **修正：** 如果异常值明显是一个错误，并且你知道正确的值，就修正它。（例如，将“75”岁改为“7”岁）。
3. **移除：** 如果异常值是一个错误且无法修正，或者它代表了一个与你的分析不相关的情况（例如，设备校准前进行的测试测量），你可能会选择将其移除。请谨慎操作，因为移除数据可能引入偏差。务必记录你移除了哪些异常值以及原因。
4. **保留：** 如果异常值是一个真实但极端的数值，你可能会决定保留它。分析时应要么使用对异常值不那么敏感的方法，要么异常值本身可能成为你调查的重点（例如，单独分析高价值交易）。
5. **转换：** 有时，对数据进行数学转换（如取对数）可以减小异常值对某些分析的影响。这是一种更高级的技术，但值得了解它的存在。

处理异常值是统计检查和专业知识的结合。识别潜在异常值是数据清理和准备过程的标准组成部分，能确保你后续的分析建立在一个更可靠的基础上。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media
  提供数据科学实用的统计方法，包括使用IQR和Z-分数进行异常值识别及其对分析的影响。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013)
  Journal: Springer Texts in Statistics; Publisher: Springer; DOI: [10.1007/978-1-4614-7138-7](https://doi.org/10.1007/978-1-4614-7138-7)
  一本经典教材，涵盖统计学习基本概念，包括数据准备和数据异常处理。
- [Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking](https://www.oreilly.com/library/view/data-science-for/9781449361327/) — Foster Provost, Tom Fawcett (2013)
  Publisher: O'Reilly Media
  从商业角度介绍数据科学，讨论数据质量问题（如异常值）及其与分析结果的相关性。
- [Exploratory Data Analysis](https://archive.org/details/exploratorydataa00tuke) — John W. Tukey (1977)
  Publisher: Addison-Wesley; Pages: 688
  一部奠基性著作，介绍了箱线图等图形技术，这是识别异常值的一个重要可视化方法。
