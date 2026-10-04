---
course: "data-modeling-schema-design-analytics"
chapter: "advanced-dimension-handling"
lesson: "date-and-time-dimensions"
sourceId: 7982
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-3-advanced-dimension-handling/date-and-time-dimensions"
title: "日期和时间维度"
description: "创建完善的日期和时间维度表。"
order: 4
plots: ["plots/7982-0.json"]
sourceHash: "5301a6cf7fd1bb2e8cf041ad33edcf51b77973f0a4cbf4e78443f640041adbcf"
sourceCorrections: []
---

数据仓库中一个普遍存在的设计问题是，依赖原始SQL日期函数来回答业务问题。尽管操作型系统存储时间戳以标记 (token)事务发生的时间，但分析型系统需要该时间戳周围的丰富相关信息。一个简单的日期字符串，如 `2023-12-25`，本身不包含它是联邦假日、财年最后一星期或星期一的信息。

为了连接原始时间戳和业务逻辑，我们建立了一个**日期维度**（通常称为 `dim_date`）。这是一个专门的维度表，其中包含日历中每一天的一行数据，覆盖数据仓库的整个生命周期，通常为10到20年。与其他随着新实体创建而增长的维度不同，日期维度是静态且预先填充的。

### 日期维度结构

此表的主要作用是作为日历属性的查询机制。通过预先计算“星期几”、“财季”和“是否假日”等属性，您可以在分析查询中避免使用复杂的`CASE`语句和函数转换。

日期维度的主键需要认真考量。虽然我们在讨论SCD时提到了系统生成的代理键（如 $1, 2, 3 \dots$ 等整数），但日期维度通常更适合采用“智能键”或整数日期格式。将日期格式化为整数 $YYYYMMDD$（例如，$20231025$）有助于事实表中的高效分区和便于理解的外键，同时在技术上仍作为与原生`DATE`数据类型不同的代理键。

下面是一个图表，展示了日期维度的模式及其与事实表的关系。

> 模式设计，将事务事实表与包含日历和财年属性的日期维度连接起来。

### 处理财年日历

多数组织采用与标准公历不同的财年日历。例如，一家零售公司可能在二月份开始其财年，以考虑到节后退货，而政府机构可能在十月份开始。

在查询内部计算财年期间既低效又容易出错。一个典型的SQL尝试来调整财年可能如下所示：

```sql
SELECT 
    CASE 
        WHEN MONTH(order_date) >= 2 THEN YEAR(order_date) 
        ELSE YEAR(order_date) - 1 
    END as fiscal_year,
    SUM(revenue)
FROM orders
GROUP BY 1
```

这种做法强制数据库在查询执行期间对事实表中的每一行进行计算。通过将此逻辑移入 `dim_date` 表中的 `fiscal_year` 和 `fiscal_quarter` 等列，查询就简化为直接连接和筛选。这种预计算确保了所有报告的一致性；两位分析师将不会无意中以不同方式定义第一季度的开始。

### 粒度：分离日期和时间

维度建模中一个常见的设计错误是试图将日期和时间合并到单个维度中。这在低容量系统中可能有效，但在大规模情况下会导致严重的性能降低。

考虑数据的基数。一个涵盖20年的日期维度大约包含 $7,300$ 行（$365 \times 20$）。这是一个小表，数据库引擎可以轻松地将其缓存在内存中。

然而，如果您精确到秒，表的粒度会显著变化。一天有 $86,400$ 秒。
$7,300 \text{ 天} \times 86,400 \text{ 秒} = 630,720,000 \text{ 行}$

一个包含6.3亿行的维度表不再是一个快速查询表；它会成为性能瓶颈。解决方案是将它们分成两个不同的维度：

1. **日期维度 (DimDate):** 粒度为一天。处理日历逻辑。
2. **时间维度 (DimTime):** 粒度通常为一分钟或一秒。处理一天中的时间分析（例如，“上午与下午销售”或“午餐高峰期”）。

事实表将包含两个独立的外键：`date_key` 和 `time_key`。这允许数据库独立优化连接。如果分析师只需要报告每日销售额，庞大的`DimTime`表将不会被访问。

下方的图表直观显示了仅日期维度与组合日期-时间维度在行数（基数）上的显著差异。



![基数影响：日期维度对比日期-时间维度](plots/7982-0.json)



> 行数对比，说明为什么将日期和时间合并到单个维度会导致表大小难以控制。请注意对数刻度。

### 生成维度

与每日摄取数据的事务表不同，日期维度通常通过脚本或电子表格一次性填充。您应该填充未来几年的日期。这使得业务能够处理未来日期的订单、预测或订阅续订，而不会产生完整性错误。

高级实现通常包含以下列：

- **相对偏移量:** `current_day_offset` (0 = 今天, -1 = 昨天)。这需要每日更新过程，但简化了“最近30天”的报告。
- **公司特有属性:** `is_pay_period_end`（是否发薪期结束）、`is_board_meeting_date`（是否董事会会议日期），或与您行业相关的特定季节性标志。

通过将时间视为结构化维度而非原始数据类型，您为分析引擎提供了必要的背景信息，以按照人类对时间、季度、节假日和财年期间的理解来聚合数据，而不仅仅是机器时间戳。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  涵盖日期和时间维度、代理键、财政日历的设计与实现，以及在维度模型中预计算属性的优势。这是数据仓库领域的经典著作。
- [Star schema and the importance of data warehouse design for analytics](https://learn.microsoft.com/en-us/azure/architecture/data-guide/relational-data/star-schema) — Microsoft Learn contributors (2023)
  Journal: Microsoft Learn
  概述了星型模式设计原则，其中包括日期维度等维度表在高效分析查询和报告中的作用。
