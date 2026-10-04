---
course: "intro-etl-pipelines"
chapter: "the-loading-stage"
lesson: "data-validation-post-load"
sourceId: 5571
sourceUrl: "https://apxml.com/zh/courses/intro-etl-pipelines/chapter-4-the-loading-stage/data-validation-post-load"
title: "加载后数据验证"
description: "了解加载完成后校验数据完整性的必要性。"
order: 7
plots: ["plots/5571-0.json"]
sourceHash: "294bda8ffc7c63ad081356e8be778adbc6597ed05d31b62c82521fe186732f15"
sourceCorrections: []
---

将数据加载到目标系统可能感觉像是终点，但还有一个必要步骤：加载后验证。仅仅因为加载过程没有报错完成，并不意味着数据自动就是正确、完整或可用的。可以将其看作是组装家具后再次检查您的工作；您需要确保所有螺丝都拧紧了，并且抽屉能顺畅滑动，然后才能开始使用。这最终的核实能建立信心，确保目前驻留在目标系统中的数据准确反映了ETL过程的预期结果。

### 为什么要加载后验证？

您可能想知道为什么在加载*后*仍然需要验证，特别是如果您之前已经执行了数据转换和清洗。这个最终检查如此必要的原因如下：

1. **捕获加载特有的错误：** 加载过程本身有时会引入问题。数据可能被截断，数据类型可能被隐式错误转换，或者记录可能被意外删除或重复，即使加载任务报告成功。
2. **校验端到端完整性：** 加载后验证作为最终的端到端检查，确认数据从源头经过转换到达目标系统的整个过程没有损坏。
3. **确保目标系统兼容性：** 它确认加载的数据符合目标系统（例如数据仓库）的特定规则、约束和模式要求。
4. **建立信任：** 下游用户，如数据分析师或报表工具，依赖于这些数据。验证这些数据可确保他们使用的是准确可靠的信息，从而促进对数据管道的信任。

### 常见的加载后验证方法

验证检查从简单的计数到更具体的数据质量测试不等。以下是一些适合初学者的常见方法：

#### 记录计数

这通常是第一个也是最简单的检查。将加载到目标表中的记录数量与转换阶段后处理的记录数量进行比较。

- **完全加载：** 对于完全加载（目标表被完全替换的情况），目标中的记录数量应与预期加载的记录数量完全匹配。设 $N_{transformed}$ 为准备加载的记录数量， $N_{target}$ 为加载后目标表中的记录数量。您预期会得到 $N_{target} = N_{transformed}$
- **增量加载：** 对于增量加载（追加新数据或更新现有记录），计算略有不同。您可以将新增记录的数量或总计数增量与源数据的预期更改数量进行比较。

您通常可以使用简单的SQL查询获取这些计数：

```sql
-- 从目标表中获取计数
SELECT COUNT(*) FROM your_target_table;
```



![记录计数验证：源数据与目标数据对比](plots/5571-0.json)



> 示例对比，显示预期的源数据/转换后的计数与目标系统中实际加载的计数相匹配。

#### 模式校验

确保加载后目标表的结构符合您的预期。检查以下内容：

- **正确的列名：** 所有预期列是否存在？
- **正确的数据类型：** 列是否以预期的数据类型（例如 `VARCHAR`、`INTEGER`、`TIMESTAMP`）加载？有时加载期间的隐式转换可能导致意想不到的问题。
- **约束：** `NOT NULL` 或 `UNIQUE` 等约束是否按预期执行？

大多数SQL环境都提供检查表结构的命令：

```sql
-- MySQL/PostgreSQL 检查表结构的示例
DESCRIBE your_target_table;
-- or
\d your_target_table
```

#### 数据范围和值检查

校验实际数据值在您的应用程序上下文 (context)中是否合理。

- **数值范围：** 检查数值列中的值是否落在合理的最小和最大限制内（例如，百分比在0到100之间，价格为正）。
- **日期范围：** 确保日期在预期的时间范围内（例如，订单日期不在未来，出生日期不过于不合理）。
- **分类值：** 检查预期包含特定类别（例如“活跃”、“非活跃”、“待处理”）的列是否只包含这些值。

使用 `MIN()`、`MAX()`、`DISTINCT` 或 `WHERE` 子句的SQL查询可以提供帮助：

```sql
-- 检查价格列的最小值/最大值
SELECT MIN(price), MAX(price) FROM your_target_table;

-- 检查状态列中的不同值
SELECT DISTINCT status FROM your_target_table;

-- 查找具有意外值的行
SELECT COUNT(*) FROM your_target_table WHERE percentage < 0 OR percentage > 100;
```

#### 空值检查

确认本应*始终*有值的列是否包含意外的 `NULL`。在主键或必要标识符等重要字段中的 `NULL` 通常表示源数据、转换逻辑或加载过程中存在问题。

```sql
-- 统计重要列中的空值
SELECT COUNT(*) FROM your_target_table WHERE essential_identifier IS NULL;
```

#### 唯一性检查

如果某列（或多列组合）应是唯一的（如主键），请验证加载后此属性是否成立。

```sql
-- 检查不同ID的数量是否与总行数匹配
SELECT COUNT(DISTINCT user_id), COUNT(*) FROM your_target_table;
-- 如果这两个数字不同，则表示存在重复的user_id！
```

#### 抽样检查

有时，自动化检查不足以发现所有问题。手动检查少量记录可以帮助您发现细微问题或验证难以自动化的数据上下文。选择少量记录并将其与源数据或预期转换后的值进行比较。

### 实现验证检查

这些检查通常通过以下方式实现：

- **SQL查询：** 直接对目标数据库或数据仓库执行。
- **脚本：** 简单的脚本（例如，使用Python和数据库连接器）可以自动化多项检查并提供更复杂的验证逻辑。
- **ETL工具功能：** 许多ETL工具都内置了专门用于数据验证的组件或功能，可以作为管道中的步骤添加。

### 应对验证失败

在加载后验证期间发现差异非常重要。根据问题的严重性和性质，应对措施可能包括：

- **告警：** 通知数据工程团队有关失败的信息。
- **调查：** 诊断根本原因（源数据问题、转换错误、加载错误）。
- **回滚：** 如果数据存在严重缺陷，可能会回滚加载操作（如前一节关于处理失败所述）。
- **修正：** 修复潜在问题并重新运行ETL过程，或在适当时直接对目标数据采取纠正措施。

> 图示加载后验证在ETL工作流中的位置，它发生在数据进入目标系统之后，下游使用之前。

执行加载后验证是数据管道管理良好的标志。它确保最终输出不仅被加载，而且被*正确*加载，为数据驱动的决策和应用程序提供了可靠的支撑。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108304/) — Joe Reis, Matt Housley (2022)
  Publisher: O'Reilly Media
  一本关于现代数据工程实践的综合指南，涵盖了数据管道生命周期中数据质量和验证的最佳实践。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.oreilly.com/library/view/the-data-warehouse/9781118530801/) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons, Inc.
  一本详细介绍数据仓库和ETL流程方法的入门级资源，强调数据完整性和质量控制是其重要组成部分。
- [Data Driven: Profiting from Your Most Important Business Asset](https://www.barnesandnoble.com/w/data-driven-thomas-c-redman/1100375990) — Thomas C. Redman (2008)
  Publisher: Harvard Business Review Press
  提供实现高数据质量的原则和策略，强调其对业务决策的重要性以及验证的作用。
