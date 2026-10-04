---
course: "intro-etl-pipelines"
chapter: "the-transformation-stage"
lesson: "practice-applying-transformations"
sourceId: 5548
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-3-the-transformation-stage/practice-applying-transformations"
title: "实践：应用简单转换"
description: "进行动手活动以应用基本数据清洗和格式转换。"
order: 9
plots: []
sourceHash: "6446adfb5e5e791b9861a84db66b4ff9e08de5a2c625173839dc0322c831cb8b"
sourceCorrections: []
---

既然我们已经讨论了数据转换、清洗、结构化、丰富化和标准化的*原因*和*内容*，现在就将这些想法付诸实践。本节提供使用示例数据集的动手练习，以巩固您对常见转换技术的理解。我们在这里不会编写代码；重点在于应用您学到的逻辑。

### 场景：准备客户数据

设想您已从旧系统中提取了以下客户数据。目标是在将其加载到新的营销数据库之前清洗并标准化这些数据。

这是我们的原始数据：

| CustID | FullName | RegistrationDate | Email | State | Orders |
| --- | --- | --- | --- | --- | --- |
| 101 | Alice Smith | 2022-05-15 | [alice.s@mail.com](mailto:alice.s@mail.com) | CA | 5 |
| 102 | Bob Johnson | 11/20/21 | bob | ny | 3 |
| 103 |  | 2023-01-10 | [charlie@mail.com](mailto:charlie@mail.com) | Texas | -2 |
| 104 | Diana Lee | March 3, 2022 | diana.lee@mail | ca | 10 |
| 105 | Evan Green | 2022-08-01 |  | FL | 7 |
| 106 | Fiona Adams | 07/10/23 | [fiona.a@mail.com](mailto:fiona.a@mail.com) | Nevada | 0 |
| 107 | George Miller | 2021-12-25 | george@mail | CA | none |

这些数据有几个原始提取中常见的问题：缺失值、日期格式不一致、州属表示不一致、无效的电子邮件格式以及可能不正确的订单数量。我们的任务是应用转换来解决这些问题。

### 活动1：数据清洗

首先，让我们处理最明显的错误和缺失信息。

1. **缺失值：** 找出 `FullName` 和 `Email` 缺失的行。您会如何处理它们？一种常见方法是根据目标系统的要求，将它们替换为“Unknown”或 `NULL` 等默认值。我们决定对 `FullName` 使用“Unknown”，如果 `Email` 确实缺失，则将其保留为空（但我们稍后会验证格式）。
2. **不正确的值：** 查看 `Orders` 列。客户ID 103 有 -2 个订单，这很可能是一个错误。客户ID 107 有 'none'。这些应该如何处理？我们可能认为负数订单是无效的，应该设置为 0 或标记 (token)以供调查。像 'none' 这样的文本值肯定应该转换为数字表示，很可能是 0。

**应用清洗：**

- 客户ID 103：将 `FullName` 设置为“Unknown”。将 `Orders` 设置为 0。
- 客户ID 105：将 `Email` 设置为空字符串或 `NULL`（暂时表示为空）。
- 客户ID 107：将 `Orders` 设置为 0。

我们的数据现在看起来是这样：

| CustID | FullName | RegistrationDate | Email | State | Orders |
| --- | --- | --- | --- | --- | --- |
| 101 | Alice Smith | 2022-05-15 | [alice.s@mail.com](mailto:alice.s@mail.com) | CA | 5 |
| 102 | Bob Johnson | 11/20/21 | bob | ny | 3 |
| 103 | Unknown | 2023-01-10 | [charlie@mail.com](mailto:charlie@mail.com) | Texas | 0 |
| 104 | Diana Lee | March 3, 2022 | diana.lee@mail | ca | 10 |
| 105 | Evan Green | 2022-08-01 |  | FL | 7 |
| 106 | Fiona Adams | 07/10/23 | [fiona.a@mail.com](mailto:fiona.a@mail.com) | Nevada | 0 |
| 107 | George Miller | 2021-12-25 | george@mail | CA | 0 |

### 活动2：数据验证和标准化

接下来，让我们强制格式一致并验证某些字段。

1. **电子邮件验证：** 检查 `Email` 列。客户ID 102（'bob'）和客户ID 104/107（'diana.lee@mail'，'george@mail'）似乎不完整或无效。一个简单的验证规则可能会检查是否存在 '@' 符号以及 '@' 符号后的一个点 '.'。未通过此检查的电子邮件可以被标记 (token)或设置为默认的无效标记。假设如果它们未能通过基本验证，我们将其标记为待审查或设置为空/NULL。
2. **日期标准化：** `RegistrationDate` 列使用多种格式（'YYYY-MM-DD'，'MM/DD/YY'，'Month Day, YYYY'）。我们需要将其标准化，例如转换为“YYYY-MM-DD”格式。
   - '11/20/21' 变为 '2021-11-20'
   - 'March 3, 2022' 变为 '2022-03-03'
   - '07/10/23' 变为 '2023-07-10'
3. **州属标准化：** `State` 列有 'CA'、'ny'、'Texas'、'ca'、'FL'、'Nevada'。我们需要将其标准化为一致的格式，例如两个字母的大写缩写。我们需要一个针对全名或小写版本的映射（查找表）。
   - 'ny' 变为 'NY'
   - 'Texas' 变为 'TX'
   - 'ca' 变为 'CA'
   - 'Nevada' 变为 'NV'

**应用验证和标准化：**

- 客户ID 102：电子邮件 'bob' 未通过验证。设置为空。日期变为 '2021-11-20'。州属 'ny' 变为 'NY'。
- 客户ID 104：电子邮件 'diana.lee@mail' 未通过验证（@ 后没有点）。设置为空。日期变为 '2022-03-03'。州属 'ca' 变为 'CA'。
- 客户ID 103：州属 'Texas' 变为 'TX'。
- 客户ID 106：日期 '07/10/23' 变为 '2023-07-10'。州属 'Nevada' 变为 'NV'。
- 客户ID 107：电子邮件 'george@mail' 未通过验证。设置为空。

此阶段后的数据：

| CustID | FullName | RegistrationDate | Email | State | Orders |
| --- | --- | --- | --- | --- | --- |
| 101 | Alice Smith | 2022-05-15 | [alice.s@mail.com](mailto:alice.s@mail.com) | CA | 5 |
| 102 | Bob Johnson | 2021-11-20 |  | NY | 3 |
| 103 | Unknown | 2023-01-10 | [charlie@mail.com](mailto:charlie@mail.com) | TX | 0 |
| 104 | Diana Lee | 2022-03-03 |  | CA | 10 |
| 105 | Evan Green | 2022-08-01 |  | FL | 7 |
| 106 | Fiona Adams | 2023-07-10 | [fiona.a@mail.com](mailto:fiona.a@mail.com) | NV | 0 |
| 107 | George Miller | 2021-12-25 |  | CA | 0 |

### 活动3：数据结构化和丰富化

最后，让我们对数据进行一些重组，以更好地适应目标系统的模式。假设目标数据库需要单独的 `FirstName` 和 `LastName` 字段，而不是 `FullName`。

1. **字段拆分：** 我们需要将 `FullName` 列拆分为 `FirstName` 和 `LastName`。我们可以根据第一个空格拆分字符串。对于“Alice Smith”，`FirstName` 是“Alice”，`LastName` 是“Smith”。对于“Unknown”怎么办？我们可以将 `FirstName` 和 `LastName` 都设置为“Unknown”。
2. **添加字段（丰富化）：** 让我们添加一个简单的派生字段。假设我们想根据客户的注册年份对他们进行分类。我们可以从标准化的 `RegistrationDate` 中提取年份，并创建一个新的 `RegistrationYear` 列。

**应用结构化和丰富化：**

- 将 `FullName` 拆分为 `FirstName` 和 `LastName`。
- 从 `RegistrationDate` 创建 `RegistrationYear`。

最终转换后的数据，准备加载：

| CustID | FirstName | LastName | RegistrationDate | RegistrationYear | Email | State | Orders |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 101 | Alice | Smith | 2022-05-15 | 2022 | [alice.s@mail.com](mailto:alice.s@mail.com) | CA | 5 |
| 102 | Bob | Johnson | 2021-11-20 | 2021 |  | NY | 3 |
| 103 | Unknown | Unknown | 2023-01-10 | 2023 | [charlie@mail.com](mailto:charlie@mail.com) | TX | 0 |
| 104 | Diana | Lee | 2022-03-03 | 2022 |  | CA | 10 |
| 105 | Evan | Green | 2022-08-01 | 2022 |  | FL | 7 |
| 106 | Fiona | Adams | 2023-07-10 | 2023 | [fiona.a@mail.com](mailto:fiona.a@mail.com) | NV | 0 |
| 107 | George | Miller | 2021-12-25 | 2021 |  | CA | 0 |

### 总结

通过这些活动，我们模拟了几个基本的数据转换步骤：

- **清洗：** 处理了缺失值并纠正了明显不正确的数据。
- **验证：** 根据简单规则检查了电子邮件格式。
- **标准化：** 将日期和州属名称转换为一致的格式。
- **结构化：** 将单个字段拆分为多个字段。
- **丰富化：** 从现有数据派生了一个新字段（`RegistrationYear`）。

经过转换的这个数据集现在更加一致、可靠，并且结构适合加载到目标营销数据库中。尽管本例使用了简单的规则和小型数据集，但其原理适用于ETL过程中遇到的更大、更复杂的转换任务。重要的是要理解目标要求，并系统地应用必要的清洗、验证、标准化、结构化和丰富化步骤。

## 参考资料

- [Executing Data Quality Projects: Ten Steps to Quality Data and Trusted Information](https://www.elsevier.com/books/executing-data-quality-projects/mcgilvray/978-0-12-818015-0) — Danette McGilvray (2021)
  Publisher: Elsevier/Academic Press; Pages: 376
  提供实施数据质量改进的系统方法，包括数据清洗、验证和标准化技术。
