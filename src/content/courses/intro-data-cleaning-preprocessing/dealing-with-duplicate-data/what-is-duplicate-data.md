---
course: "intro-data-cleaning-preprocessing"
chapter: "dealing-with-duplicate-data"
lesson: "what-is-duplicate-data"
sourceId: 3984
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-3-dealing-with-duplicate-data/what-is-duplicate-data"
title: "如何定义重复数据？"
description: "了解不同类型的重复项：完整行重复与基于特定列的重复。"
order: 1
plots: []
sourceHash: "be792ce0b7eb24830b44fa591a700fb80fc4e10e5f17672ae7b55d73dbdd6bdd"
sourceCorrections: []
---

当我们谈论重复数据时，我们指的是数据集中多次表示同一实体或事件的记录或条目。然而，究竟是什么让一条记录成为另一条记录的“重复”，这并非总是一个简单的“是”或“否”的问题。定义在很大程度上取决于数据结构以及您认为的独特信息是什么。我们来分析一下常见情况。

### 完整行重复

最直接的重复类型是**完整行重复**。当数据集中一行与另一行在*所有*列中都完全相同时，就会出现这种情况。一行中的每个值都与另一行中的对应值匹配。

想象一个简单的客户注册数据集：

重复数据指的是数据集中代表同一实体或事件多次的记录或条目。然而，究竟是什么使一条记录成为另一条记录的“重复”，这并非总是一个简单的“是”或“否”问题。其定义在很大程度上取决于数据结构以及您认为的唯一信息。

在此表中，第三行和第四行是完整的重复项。每条信息（`用户ID`、`姓名`、`电子邮件`、`注册日期`）都完全相同。这些通常是由于数据收集期间的技术故障、表单意外重复提交或合并不同数据文件时的错误引起的。识别这些通常很直接。

### 基于特定列的重复

通常，重复的情况更不明显。即使其他列不同，也可以根据特定列集中的值将记录视为重复项。这些列通常作为独特实体或事件的**标识符**。我们可以称之为**部分重复**或**基于列子集的重复**。

考虑一个更新的客户活动视图，例如跟踪登录情况：

| 日志ID | 用户ID | 电子邮件 | 登录时间 | 操作 |
| --- | --- | --- | --- | --- |
| 5001 | 101 | [alice@example.com](mailto:alice@example.com) | 2023-02-01 09:00:15 | 查看页面 |
| **5002** | **102** | **[bob@example.com](mailto:bob@example.com)** | **2023-02-01 09:05:22** | **登录** |
| **5003** | **102** | **[bob@example.com](mailto:bob@example.com)** | **2023-02-01 10:15:00** | **更新资料** |
| 5004 | 101 | [alice@example.com](mailto:alice@example.com) | 2023-02-01 11:00:05 | 注销 |
| **5005** | **102** | **[bob@example.com](mailto:bob@example.com)** | **2023-02-01 09:05:22** | **登录** |

这里，`日志ID`为5002和5005的行看起来非常相似。

- 如果我们将重复条目简单地定义为*所有*列中的值都相同，那么5002和5005就*不是*完整重复项，因为它们的`日志ID`值不同（5002与5005）。
- 然而，如果我们的目标是根据`用户ID`、`电子邮件`和`登录时间`的组合来查找独特的*登录事件*，那么行5002和5005表示的是*同一事件*。它们在这三列中的值完全相同。`日志ID`可能只是一个内部数据库行计数器，并且`操作`也相同（'登录'）。在这种情况下，其中一行可以被视为重复的登录记录。
- 如果我们只对独特的*用户*感兴趣，我们只会查看`用户ID`（或者可能是`电子邮件`）列。在这种情况下，行5002、5003和5005都指的是同一用户（Bob，用户ID 102），但它们表示不同的操作，或者可能是同一操作的重复日志。

### 语境与标识符的重要性

决定什么构成重复项，需要理解数据以及分析的目的。

- 您对独特的客户感兴趣吗？那么重复项可能由`客户ID`或`电子邮件`定义。
- 您对独特的交易感兴趣吗？那么重复项可能由`交易ID`定义，或者可能是`客户ID` + `时间戳` + `产品`的组合。
- 您在寻找数据录入错误吗？那么完整行重复是主要目标。

确定应唯一标识记录的正确列集是基础步骤。这些列通常被称为**唯一标识符**或**主键**（尽管主键一词在数据库中有特定含义，但这里的思想类似）。有时单列（如`用户ID`）就足够了；其他时候，需要列的组合。这通常需要一定的**专业知识**，即熟悉数据所代表的主题，才能做出正确的选择。

理解这些不同类型的重复是您能有效识别和处理数据集中重复项的第一步。接下来的部分将说明为什么通常需要删除重复项以及用于查找和删除它们的技术。

## 参考资料

- [Data Cleaning: Problems and Current Approaches](https://www.researchgate.net/publication/2372795_Data_Cleaning_Problems_and_Current_Approaches) — Erhard Rahm, Hong Hai Do (2000)
  Journal: IEEE Bulletin of the Technical Committee on Data Engineering; Publisher: IEEE Computer Society; Volume: 23; Pages: 3-13
  阐述了数据清洗问题及其处理方法，包括如何识别和解决重复记录，建立了概念框架。
- [Data Science for Business: What You Need to Know about Data Mining and Data-Analytic Thinking](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGFXxmCpA9YBZZxX5hs5-vzWe5MUG44LRTHSfXKQ_Qratwrgvz3qlVbTWiq0my0LU5pfdsOVGDvwVyqIHM928YL82EfzhYt36fQmUW9tVkjSdO0UTjO14A0cG67gVx8hl_OIRHhgATLoXqLNSLhy_qdYvrn3NZxpSvN9i3uX5E=) — Foster Provost, Tom Fawcett (2013)
  Publisher: O'Reilly Media; Pages: 408 pages
  涵盖了数据分析中的数据准备和数据质量理解，对于正确定义和管理数据重复至关重要。
- [Data Quality and Entity Resolution](https://link.springer.com/referenceworkentry/10.1007/978-3-319-76840-5_37-1) — Larisa Gheorghe and Erhard Rahm (2019)
  Journal: Encyclopedia of Big Data Technologies; Publisher: Springer, Cham; DOI: [10.1007/978-3-319-76840-5_37-1](https://doi.org/10.1007/978-3-319-76840-5_37-1)
  提供了关于数据质量和实体解析的最新学术观点，侧重于在大数据集中检测重复项的方法。
