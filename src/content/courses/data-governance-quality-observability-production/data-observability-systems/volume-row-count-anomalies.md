---
course: "data-governance-quality-observability-production"
chapter: "data-observability-systems"
lesson: "volume-row-count-anomalies"
sourceId: 8096
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-3-data-observability-systems/volume-row-count-anomalies"
title: "数据量与行数异常"
description: "检测数据量和行数的意外变化。"
order: 3
plots: ["plots/8096-0.json"]
sourceHash: "6974d84b2d0de00f313a439b233e326ef009f967c64398bb3fef493a6a690c92"
sourceCorrections: []
---

数据管道常常静默失败。转换逻辑错误通常会抛出异常并停止执行，这会立即通知值班工程师。然而，当管道成功运行但产生错误输出时，就会发生静默失败。静默失败最常见的迹象是数据量的意外变化。

数据量监控是数据基础设施的生命线。它回答了一个基本问题：“我们收到了预期的数据量吗？” 如果一个表通常每小时摄入一百万行，但突然只摄入一万行，那么管道在技术上是成功的，但数据在功能上却毫无用处。反之，行数飙升到两百万可能表示重复处理或重试循环出错。

### 数据量检测机制

数据量异常通常分为三类。

1. **数据完全丢失：** 行数降至零。这最容易检测，但通常表示灾难性的上游故障，例如连接断开或凭证过期。
2. **部分数据丢失：** 行数显著下降但保持在零以上。这很危险，因为它经常会被基本检查忽略。这通常表明某个特定分区、分片或区域未能报告，而其他部分成功了。
3. **数据量激增：** 行数显著增加。这通常是由于重试期间幂等性处理不当造成的重复，或是用户行为突然改变（如DDoS攻击或病毒式营销活动）。

为了可视化这些模式，可以考虑记录计数的时间序列视图。一个有效的监控器必须区分正常波动和实际异常。



![24小时内检测到的数据量异常](plots/8096-0.json)



> 数据量监控可检测到与预期基线偏离的突然下降（部分数据丢失）和激增（重复）。

### 固定阈值与自适应监控

数据量监控最简单的方法是应用固定阈值。你可以设定一个表必须至少有500行。这适用于大小静态且已知的参考表或字典。

```sql
-- 简单的固定阈值检查
SELECT 
  CASE 
    WHEN count(*) < 500 THEN 'FAIL' 
    ELSE 'PASS' 
  END as status
FROM target_table
WHERE ingestion_date = CURRENT_DATE;
```

然而，固定阈值在生产环境中会失效，因为数据量很少是静态的。它会根据一天中的时间、一周中的某天和业务季节性而波动。在周二早上有效的固定阈值很可能会在周日晚上触发误报。

为了处理这种情况，我们采用基于统计基线的自适应监控。我们不定义一个固定数值，而是根据历史表现定义一个可接受范围。

### Z分数异常检测的实现

一种检测时间序列数据量异常的方法是Z分数（标准分数）。此指标量化 (quantization)了数据点偏离均值的标准差数量。

检测时间 $t$ 异常的公式是：

$| V_t - \mu | > k \cdot \sigma$

其中：

- $V_t$ 是在时间 $t$ 观察到的数据量。
- $\mu$ (mu) 是在回溯窗口期（例如，过去30天）内数据量的移动平均值。
- $\sigma$ (sigma) 是同一窗口期内的标准差。
- $k$ 是敏感度阈值（通常为2或3）。

如果当前数据量与平均值之间的绝对差值大于 $k$ 个标准差，系统将标记 (token)为异常。$k$ 值为3大约捕获了正态分布中99.7%的预期值，这意味着超出此范围的任何数据点极有可能是一个离群值。

以下SQL模式演示了如何使用窗口函数计算这些指标。这使你能够随着数据到达动态计算异常分数。

```sql
WITH daily_counts AS (
    SELECT 
        ingestion_date,
        COUNT(*) as row_count
    FROM raw_events
    GROUP BY 1
),
stats AS (
    SELECT 
        ingestion_date,
        row_count,
        -- 计算过去7天的移动平均值
        AVG(row_count) OVER (
            ORDER BY ingestion_date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) as moving_avg,
        -- 计算过去7天的标准差
        STDDEV(row_count) OVER (
            ORDER BY ingestion_date 
            ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
        ) as moving_stddev
    FROM daily_counts
)
SELECT 
    ingestion_date,
    row_count,
    moving_avg,
    moving_stddev,
    (row_count - moving_avg) / NULLIF(moving_stddev, 0) as z_score
FROM stats
WHERE ingestion_date = CURRENT_DATE;
```

如果计算出的 `z_score` 大于3或小于-3，监控器将触发警报。

### 处理季节性

Z分数模型假设分布相对一致，但数据通常表现出强烈的季节性。例如，B2B SaaS应用通常在工作日数据量大，周末数据量小。将周一的数据量与过去7天的平均值（包括周末）进行比较，可能仅仅因为周末拉低了平均值，就将周一标记 (token)为高数据量异常。

为了纠正这种情况，高级数据量监控器使用**季节性感知基线**。你不是将“今天”与“过去7天”进行比较，而是进行以下比较：

1. **同期比较：** 将本周一的数据量与上周一的数据量进行比较。
2. **同一小时比较：** 将今天早上9点与昨天早上9点进行比较。

通过调整基线比较以匹配预期周期，可以减少误报。生产级监控器的逻辑流程按顺序纳入这些检查。

> 此工作流程演示了一种季节性感知检查，其中基线源自可比较的历史时期，而非简单的移动平均值。

### 元数据与计算

在实现数据量检查时，性能是一个考量因素。在PB级规模的表上运行 `COUNT(*)` 既昂贵又缓慢。

对于Snowflake、BigQuery或Redshift等数据仓库，应尽可能使用系统元数据表（Information Schema）。这些表维护着缓存的行数，并随数据加载而更新。

- **精确计数：** `SELECT COUNT(*) FROM table` （计算成本高，100%准确）。
- **元数据计数：** `SELECT row_count FROM information_schema.tables` （零计算成本，最终一致）。

对于异常检测，元数据计数通常足够。目标是检测显著偏差（例如20%的下降），而不是精确核对账本到每一分钱。使用元数据可以让你频繁运行数据量监控器，每15分钟或每小时一次，而不会产生显著的仓库成本。

### 数据量警报的根因分析

当数据量警报触发时，调查应遵循标准路径。如果数据量低，首先检查上游提取日志。API是否返回了200 OK但载荷为空？如果数据量高，检查主键约束。你是否看到唯一ID多次出现？

数据量监控是第一道防线。它们不会告诉你数据是否准确，但会告诉你系统是否正常运行。在下一节中，我们将介绍如何检测数据存在但结构已改变的情况，这是一种称为模式漂移的现象。

## 参考资料

- [Fundamentals of Data Engineering: Planning and Building Robust Data Systems](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108702/) — Joe Reis and Matt Housley (2022)
  Publisher: O'Reilly Media
  这本书涵盖了现代数据工程的关键方面，包括专门讨论数据质量、可靠性和可观测性的内容，这些都是数据管道监控的核心。
- [A Survey of Outlier Detection Methodologies](https://dl.acm.org/doi/10.1145/1541880.1541882) — Varun Chandola, Arindam Banerjee, and Vipin Kumar (2009)
  Journal: ACM Computing Surveys (CSUR); Publisher: Association for Computing Machinery; Volume: 41; Pages: 1-58; DOI: [10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882)
  这篇综述文章概述了各种异常值检测技术，包括直接适用于识别数据集中数据量和行数异常的统计方法。
