---
course: "introduction-to-data-science"
chapter: "understanding-data-fundamentals"
lesson: "structured-vs-unstructured-data"
sourceId: 2094
sourceUrl: "https://apxml.com/zh/courses/introduction-to-data-science/chapter-2-understanding-data-fundamentals/structured-vs-unstructured-data"
title: "结构化数据与非结构化数据"
description: "了解结构化数据（如表格）与非结构化数据（如文本或图像）之间的区别。"
order: 2
plots: []
sourceHash: "ceee78ac3846eb8fe5d6cce4da71e695acddff943c3f66f2a6df2f72d5786631"
sourceCorrections: []
---

数据形式多样。对数据进行分类的一个基本方法是根据其组织程度：数据是整齐排列的，还是更自由的？这引出了结构化数据和非结构化数据之分。理解这种差异很重要，因为数据类型通常决定了我们如何存储、处理和分析它。

## 结构化数据：整齐的行和列

可以将结构化数据看作是能很好地符合预定义模型的信息，比如电子表格或数据库表中的行和列。它具有一致的格式，并遵循特定的模式（一种定义组织方式的蓝图）。每条数据都有其指定位置。

常见特征包括：

- **有组织性：** 数据位于记录或文件中的固定字段内。
- **定义模式：** 每个数据点的含义和格式都预先清晰定义。
- **易于查询：** 其规则的结构使得使用标准工具（如数据库的SQL或电子表格软件中的函数）进行查询和分析相对容易。

**例子：**

- 电子表格中的客户联系人列表，包含 `Name`（姓名）、`Email`（电子邮件）、`Phone Number`（电话号码）和 `City`（城市）等列。
- 数据库表中记录的销售交易，包含 `TransactionID`（交易ID）、`Date`（日期）、`ProductID`（产品ID）、`Quantity`（数量）和 `Price`（价格）等字段。
- 按固定间隔记录的传感器读数，包含 `Timestamp`（时间戳）、`SensorID`（传感器ID）和 `Value`（值）。

请看这个结构化客户数据的简单表示：

| 客户ID | 姓名 | 电子邮件 | 城市 |
| --- | --- | --- | --- |
| 101 | Alice Smith | [alice@example.com](mailto:alice@example.com) | 纽约 |
| 102 | Bob Johnson | [bob.j@example.org](mailto:bob.j@example.org) | 伦敦 |
| 103 | Carol Lee | [c.lee@domain.net](mailto:c.lee@domain.net) | 旧金山 |

这种表格格式是结构化数据的典型例子。每行代表一个客户，每列代表该客户的一个特定属性。

## 非结构化数据：多样和自由形式的信息

非结构化数据基本上是所有其他类型的数据。它没有预定义的数据模型，也没有易于识别的、能整齐放入行和列的结构。它通常包含文本、图像、音频和视频。虽然它含有有价值的信息，但提取它需要更先进的技术。

常见特征包括：

- **无预定义模型：** 数据缺乏一致的、可重复的结构。
- **定性性质：** 通常由文本、图像、声音或视频组成，而非易于量化 (quantization)的数字。
- **传统处理困难：** 标准数据库工具和电子表格分析方法通常不足以直接分析。需要专门技术，如用于文本的自然语言处理（NLP）或用于图像的计算机视觉。

**例子：**

- **文本：** 电子邮件、社交媒体帖子、文章、客服通话记录、书籍、包含开放式问题的调查问卷回复。
- **图像：** 照片、医学图像（X射线、MRI）、卫星图像。
- **音频：** 会议录音、音乐文件、播客、语音消息。
- **视频：** 安防录像、电影、演示文稿、视频通话。

设想一下你电子邮件收件箱的内容。你有发件人信息、收件人信息、时间戳（这些具有一定结构），但主要内容，即邮件正文，是自由形式的文本。同样地，产品评论集合包含星级评分（有点结构化），但也包含解释评分的自由文本评论（非结构化）。

## 两者对比

主要区别在于组织方式。结构化数据高度组织化并遵循严格的格式，而非结构化数据则多样化且缺乏固定模式。

> 该图表展示了结构化数据（表格式组织，有定义模式，更易分析）与非结构化数据（自由形式如文本或图像，无预定义模型，需要复杂分析）之间的区别。

## 这种区别为何重要？

识别数据是结构化还是非结构化，通常是决定如何处理数据科学问题的第一步。

1. **工具：** 需要不同的工具和技术。结构化数据通常可以用关系型数据库、电子表格和标准统计软件处理。非结构化数据则常需要专用数据库（如NoSQL）、数据湖以及高级分析技术（如用于文本或图像分析的机器学习 (machine learning)模型）。
2. **准备：** 准备结构化数据通常涉及清理缺失值或纠正定义字段中的不一致。准备非结构化数据可能涉及复杂步骤，如文本解析、图像特征提取或音频转录，然后才能开始分析。
3. **分析复杂性：** 查询和汇总结构化数据通常是直接的。从非结构化数据中提取有意义的洞察通常更复杂且计算密集。

在实际中，你经常会遇到这两种类型，有时甚至在同一数据集中混合出现（通常称为半结构化数据，如JSON或XML文件，它们有标签但内容灵活）。能够识别和适当处理每种类型是数据科学中的一项基本能力。

## 参考资料

- [Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking](https://www.oreilly.com/library/view/data-science-for/9781449361327/) — Foster Provost, Tom Fawcett (2013)
  Publisher: O'Reilly Media
  数据科学原理的介绍，涵盖各种数据类型及其业务意义。
- [Database System Concepts](http://www.db-book.com/) — Avi Silberschatz, Henry F. Korth, S. Sudarshan (2019)
  Publisher: McGraw-Hill
  一本涵盖关系数据库系统、模式设计和结构化数据管理的教科书。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供关于各种数据存储模型的见解，包括关系型、NoSQL及其对不同数据类型的适用性。
