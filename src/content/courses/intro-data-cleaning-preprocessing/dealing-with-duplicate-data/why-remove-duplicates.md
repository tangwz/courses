---
course: "intro-data-cleaning-preprocessing"
chapter: "dealing-with-duplicate-data"
lesson: "why-remove-duplicates"
sourceId: 3987
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-3-dealing-with-duplicate-data/why-remove-duplicates"
title: "为何移除重复数据？"
description: "了解重复记录在分析和建模中引起的问题。"
order: 2
plots: ["plots/3987-0.json"]
sourceHash: "2c8ea5807e5284247c0e0a42d0567b02550599fbf42d8509a0620e3be2cff1dd"
sourceCorrections: []
---

你已经了解了重复数据是*什么*以及如何发现它。但你可能会问：“它真的有那么糟糕吗？为什么还要费力去删除它呢？” 结果是，在数据集中保留重复记录会导致一些重要问题，影响从简单计算到复杂的机器学习 (machine learning)结果。让我们看看处理重复数据成为数据准备中必要步骤的主要原因。

### 统计分析偏差

重复数据最直接的影响之一就是对基本统计指标。假设你正在分析客户销售数据以计算平均购买金额。

考虑这个小型数据集：

| 客户ID | 购买金额 |
| --- | --- |
| 101 | \$50 |
| 102 | \$75 |
| 103 | \$120 |
| **102** | **\$75** |
| 104 | \$90 |
| **101** | **\$50** |

如果你直接从这张表中计算平均购买金额，你会将所有金额相加（$50 + 75 + 120 + 75 + 90 + 50 = 460$），然后除以条目数（6），得到 460 / 6 \approx \76.67\$。

然而，客户101和102都出现了两次，并且购买金额完全相同。如果这些代表的是被多次记录的*同一*购买事件，那么唯一购买记录只有：

| 客户ID | 购买金额 |
| --- | --- |
| 101 | \$50 |
| 102 | \$75 |
| 103 | \$120 |
| 104 | \$90 |

现在，总和是 $50 + 75 + 120 + 90 = 335$，唯一购买记录数为4。正确的平均购买金额是 335 / 4 = \83.75\$。

重复数据人为地夸大了低价值购买（$50和$75）的数量，从而拉低了平均值。类似的不准确情况也会发生在计数、求和、中位数以及其他汇总统计中。如果你试图了解客户行为或业务表现，重复数据会导致不准确的结论。



![重复数据对数量和平均值的影响](plots/3987-0.json)



> 比较说明了重复记录如何影响总计数和计算出的平均购买金额。

### 有偏差的机器学习 (machine learning)模型

重复数据也可能误导机器学习模型。模型从训练数据中学习模式。如果某些记录被大量重复，模型可能会错误地认为这些实例比实际情况更普遍或更重要。

想象一下训练一个预测客户流失的模型。如果数据录入错误为*未*流失的客户创建了许多重复数据，模型可能会变得过于乐观，预测的流失率低于实际情况。这是因为它在训练过程中看到了“未流失”模式被过度重复。

此外，重复数据会人为地夸大模型性能指标。如果相同的重复记录同时出现在训练集（用于教导模型）和测试集（用于评估模型）中，模型很可能会在这些特定重复数据上表现非常好，仅仅因为它之前见过它们。这会给人一种对模型泛化到新的、未见数据能力的不实信心。移除重复数据可确保模型学习到真实模式，并使性能评估更贴近实际。

### 计算资源浪费

尽管可能不如分析准确性或模型偏差那么严重，重复数据也会消耗不必要的资源。多次存储相同信息会占用额外的磁盘空间。更重要的是，在分析或模型训练期间处理这些冗余行需要额外的计算时间和内存。在大型数据集上，移除重复数据可以明显提升处理速度和效率。

### 数据完整性受损与操作问题

在分析和建模中，重复数据会损害数据整体的完整性，并导致实际问题。考虑以下情景：

- **客户关系管理：** 重复的客户条目可能导致向同一个人发送多封营销邮件，引起反感，并可能损害客户关系。不同的客服人员可能与同一客户的不同重复记录进行交互，导致服务不一致。
- **库存或事件追踪：** 如果追踪收到的物品或发生的事件的记录重复，可能导致库存计数不准确或事件发生频率被高估。这可能导致基于错误数据的糟糕业务决策。

移除重复数据有助于保持每条记录的单一、准确版本，从而确保一致性并防止操作错误。

总之，移除重复数据不仅仅是为了整洁。这是确保以下几点的重要一步：

- 准确的统计分析和报告。
- 公平可靠的机器学习 (machine learning)模型训练和评估。
- 高效使用存储和计算资源。
- 为运营目的提供一致可靠的数据。

通过识别和移除重复数据，你为后续的任何数据工作打下了更清晰、更可信的根基。

## 参考资料

- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, Jian Pei (2011)
  Publisher: Elsevier
  这本书是数据挖掘领域公认的资源，其中专门有一章介绍数据预处理，涵盖数据清洗策略及解决不一致和重复数据的基本原理，以确保分析的数据质量。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125964/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本机器学习实用指南广泛涵盖数据准备技术，强调干净数据（包括重复数据移除）对于构建稳健、准确的机器学习模型的重要性。
- [Data Quality: The Ten Dimensions That Every Data Professional Needs to Know](https://www.technicspublications.com/shop/books/data-quality-the-ten-dimensions-that-every-data-professional-needs-to-know/) — James D. Price (2017)
  Publisher: Technics Publications
  本书提供了一个理解和管理数据质量的实用框架，详细说明了重复记录等问题如何破坏数据完整性，并从多个维度影响业务运营和分析结果。
