---
course: "intro-data-cleaning-preprocessing"
chapter: "importance-clean-data"
lesson: "impact-poor-data-quality"
sourceId: 3950
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-1-importance-clean-data/impact-poor-data-quality"
title: "数据质量差的影响"
description: "了解不干净的数据如何影响分析结果、模型性能和决策。"
order: 4
plots: ["plots/3950-0.json"]
sourceHash: "6b7688ff976ea7d6c3e6407ec2c51376590027aa07a77751b6821bada8bb682d"
sourceCorrections: []
---

您已经了解到原始数据通常包含错误、不一致和缺失信息。但是，如果使用这些“脏”数据进行分析或构建机器学习 (machine learning)模型，会发生什么呢？忽视数据质量问题不只是一件小麻烦；它可能带来严重的不良后果，从而破坏您的整个项目。现在，我们来了解一下数据质量差的具体影响。

### 分析不准确，认识有误

脏数据最直接的后果可能是导致计算不正确和解读有偏差。简单的统计指标，如平均值或总和，可能会因离群值或错误而严重偏离。想象一下，您正在计算客户的平均订单值，但由于数据录入错误，有几项记录的数值错误地比实际大了一千倍。您计算出的平均值将会被人为虚高，从而给人一种客户消费高的错误印象。

考虑这个简单场景：您有五笔交易，金额分别为$50、$60、$45、$70、$45。平均值为$54。现在，假设其中一个值被错误地输入为$5000而不是$50。交易金额变为$5000、$60、$45、$70、$45。新的平均值飙升至$1044。这一个错误就完全扭曲了典型交易金额的真实情况。



![离群值对平均交易值的影响](plots/3950-0.json)



> 从五个数据点计算出的平均交易值，对比了引入单个大离群值（$50而不是$5000）之前和之后的情况。

不一致的格式（例如“USA”、“U.S.A.”和“United States”都代表同一个国家）会阻碍准确的分组和聚合。不正确的数据类型，例如将数字存储为文本，会使数学运算无法进行或出错。这些问题会导致分析结果与现实不符，生成误导性图表、报告和结论。

### 机器学习 (machine learning)模型性能不佳

机器学习模型直接从训练数据中学习模式。如果训练数据有缺陷，模型将学习到错误的模式或噪声。这通常可以用“输入垃圾，输出垃圾”（Garbage In, Garbage Out, GIGO）来概括。

例如：

- **缺失值：** 如果缺失值处理不当，许多算法无法处理数据，或者它们可能会做出对您面对的问题不正确的默认假设。
- **离群值：** 极端值可能会不成比例地影响模型的学习过程，可能导致模型在典型数据点上的表现不佳。
- **错误标签：** 在监督学习 (supervised learning)中（模型从带标签的示例中学习，例如将电子邮件分类为“垃圾邮件”或“非垃圾邮件”），错误的标签会教导模型错误的关联。一个用大量错误标记 (token)的电子邮件训练出来的模型，在分类新邮件时会不可靠。
- **特征不一致：** 如果同一内容在您的数据中以不同方式表示（例如，同一列中单位不一致，如千克和磅），模型可能会将它们视为不同的特征，从而阻碍其学习有意义的关联的能力。

在脏数据上训练出的模型可能会有较低的准确性，做出不可靠的预测，并且难以很好地泛化到新的、未见过的数据。

### 浪费时间和资源

处理数据质量差的后果通常耗时且低效。分析师和数据科学家可能花费数小时调试意想不到的结果，结果却发现问题根源在于一个本可以更早修复的数据错误。如果数据质量问题在项目后期才被发现，这可能需要大量返工，包括重新收集数据、重新运行分析或重新训练模型。这会延迟项目时间，并消耗宝贵的计算资源和人员工时。

### 错误的决策

最终，数据分析和机器学习 (machine learning)的目标通常是为决策提供信息。如果由于质量问题导致从数据得出的认识不准确，那么基于这些认识做出的决策很可能不够理想，甚至是有害的。企业可能会错误分配资源、定位错误的客户群体、对市场趋势得出错误的结论，或者无法发现风险，因为他们的理解是基于有缺陷的数据。

### 信任受损

当分析、报告或数据驱动的产品被发现是基于不可靠数据时，这可能会损害相关个人、团队或组织的可信度。利益相关者可能会对所呈现的结果失去信心，客户也可能会因为底层数据问题导致产品或服务表现异常而失去信任。重建这种信任可能是一个困难而漫长的过程。

总而言之，在项目开始时投入时间进行数据清洗和预处理不仅仅是为了整洁。它是确保后续任何分析或模型构建的可靠性、准确性和有效性的基本步骤。干净的数据能为可靠的认识和稳定有效的结果提供坚实保障。

## 参考资料

- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Morgan Kaufmann; Pages: 83-128
  提供数据预处理的基础概述，包括数据质量问题及其对分析的影响。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media
  解释数据准备对机器学习模型的实践重要性，以及数据质量差对模型性能的后果。
- [The impact of data quality on business processes and decision-making: A case study approach](https://www.sciencedirect.com/science/article/pii/S187705092200898X) — Marvin Haffke, Maximilian Kaltenbach, Clemens Stöhr (2022)
  Journal: Procedia Computer Science; Publisher: Elsevier; Volume: 200; Pages: 2039-2048; DOI: [10.1016/j.procs.2022.01.403](https://doi.org/10.1016/j.procs.2022.01.403)
  提供关于数据质量差如何具体影响业务流程和组织决策的最新学术观点。
- [Data quality problems: What happens when they don't get fixed?](https://www.researchgate.net/publication/259160533_Data_quality_problems_What_happens_when_they_don%27t_get_fixed) — Thomas C. Redman Jr. (2013)
  Journal: MIT Sloan Management Review; Publisher: Massachusetts Institute of Technology; Volume: 54; Pages: 35-39; DOI: [10.1177/154165181305400405](https://doi.org/10.1177/154165181305400405)
  一篇备受推崇的文章，讨论组织中数据质量问题被忽视时所产生的后果和成本。
