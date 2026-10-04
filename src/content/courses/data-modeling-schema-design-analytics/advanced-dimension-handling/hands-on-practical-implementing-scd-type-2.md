---
course: "data-modeling-schema-design-analytics"
chapter: "advanced-dimension-handling"
lesson: "hands-on-practical-implementing-scd-type-2"
sourceId: 7984
sourceUrl: "https://apxml.com/zh/courses/data-modeling-schema-design-analytics/chapter-3-advanced-dimension-handling/hands-on-practical-implementing-scd-type-2"
title: "动手实践: 实现SCD 类型 2"
description: "编写SQL以实现SCD 类型 2 逻辑。"
order: 6
plots: ["plots/7984-0.json"]
sourceHash: "19a3505f78f4fe16caae4bf3ac2e8ba858e39f4baa4c4bac3739d447fc4217c5"
sourceCorrections: []
---

保留历史数据的理论需要，在许多分析场景中都很明显。例如，当分析师要求查询上一年的销售区域销售业绩时，查询结果必须反映去年存在的区域划分，而不是现在的划分。这项要求需要具体的SQL实现。

本实践练习将引导你构建一个慢变维度（SCD）类型2加载器。我们将重点关注 `dim_customer` 表，在该表中我们追踪 `address` 字段的变更。

### 设计目标架构

在编写转换逻辑之前，物理表结构必须支持历史版本管理。标准的类型1维度表只包含自然键和属性。对于类型2，我们添加元数据列来控制每条记录的有效期间。

你的目标维度表需要以下结构：

1. **代理键 ($k$):** 一个系统生成的整数，唯一标识记录的特定版本。
2. **自然键:** 来自源系统的持久标识符（例如，`customer_id`）。
3. **属性:** 我们追踪的描述性数据（例如，`address`，`city`）。
4. **生效日期 (`start_date`):** 该版本开始生效的时间点 $t$。
5. **失效日期 (`end_date`):** 该版本停止生效的时间点。
6. **当前标志 (`is_current`):** 一个布尔指示器，用于高效筛选最新记录。

以下图表说明了处理传入变更时所需的数据流和状态转换。

> 根据状态比较，用于确定插入、更新或忽略传入记录的逻辑流程。

### 步骤 1: 初始化维度

首先，我们创建表结构。在云数据仓库中，你可能会使用 `IDENTITY` 或 `AUTOINCREMENT` 作为代理键。我们使用一个最大日期（例如 '9999-12-31'）来表示活动记录的失效日期，确保它在范围查询中充当“无限期”。

```sql
CREATE TABLE dim_customer (
    customer_sk INT IDENTITY(1,1) PRIMARY KEY, -- 代理键
    customer_id VARCHAR(50),                   -- 自然键
    address VARCHAR(255),                      -- 在此追踪变更
    city VARCHAR(100),
    start_date DATE,
    end_date DATE,
    is_current BOOLEAN
);
```

### 步骤 2: 暂存源数据

在生产流程中，数据会到达暂存区。我们假设有一个 `stg_customer_updates` 表，其中包含来自操作系统的客户当前状态。

我们需要识别三类记录：

1. **新记录:** `customer_id` 存在于暂存区但不在维度表中。
2. **变更记录:** `customer_id` 在两者中都存在，但 `address` 不同。
3. **未变更记录:** `customer_id` 在两者中都存在，且 `address` 相同。

### 步骤 3: 插入更新逻辑

实现 SCD 类型 2 通常涉及事务中的两个不同操作：识别要“失效”的记录，并插入新版本。

#### 使旧记录失效

首先，我们识别已变更的记录。我们查找自然键匹配、维度记录当前活跃且属性值不同的行。我们更新这些行，将 `end_date` 设置为当前处理日期（或业务生效日期），并将 `is_current` 切换为 `false`。

```sql
UPDATE dim_customer d
SET 
    end_date = CURRENT_DATE(),
    is_current = FALSE
FROM stg_customer_updates s
WHERE d.customer_id = s.customer_id
  AND d.is_current = TRUE
  AND d.address <> s.address; -- 检测变更
```

#### 插入新版本

接下来，我们插入新的活跃记录。这包含两种情况：全新的客户，以及我们在上一步中使其失效的客户的新版本。

```sql
INSERT INTO dim_customer (
    customer_id, 
    address, 
    city, 
    start_date, 
    end_date, 
    is_current
)
SELECT 
    s.customer_id,
    s.address,
    s.city,
    CURRENT_DATE(), -- 新版本从现在开始
    '9999-12-31',   -- 未来无限日期
    TRUE
FROM stg_customer_updates s
LEFT JOIN dim_customer d 
    ON s.customer_id = d.customer_id 
    AND d.is_current = TRUE
WHERE 
    d.customer_id IS NULL           -- 这是一个全新的客户
    OR s.address <> d.address;      -- 或者这是一个已变更的客户
```

### 记录历史可视化

随着时间推移应用此逻辑后，单个客户实体将跨越多行。每行代表一个独立的时间段。查询特定客户的表会显示其历史记录。

考虑一位客户，他曾居住在纽约，六月搬到芝加哥，十一月搬到西雅图。



![客户位置历史 (SCD 类型 2)](plots/7984-0.json)



> 单个客户活动记录的时间线视图。不同的颜色代表特定时间段内的不同活动地址。

### 查询时点数据

SCD 类型 2 的优点在于能够重构任何特定时刻的状态。如果你需要根据客户在**9月15日**的居住地生成发票报告，你可以使用日期范围进行筛选。

SQL条件 $start\_date \leq t < end\_date$ 是这些查询的标准模式。

```sql
SELECT 
    k.customer_sk,
    k.city,
    f.order_amount
FROM fact_orders f
JOIN dim_customer k 
    ON f.customer_id = k.customer_id
WHERE f.order_date = '2023-09-15'
  AND f.order_date >= k.start_date 
  AND f.order_date < k.end_date;
```

如果你在 `customer_id` 上使用标准连接而没有日期逻辑，你可能会导致结果集扇出（将一行事实表记录连接到三行维度表记录），或者连接到错误的地址，从而导致不正确的税费或运费计算。

### 实现

在高容量环境中实现此功能时，请留意以下技术细节：

- **空值处理:** 标准 SQL 比较运算符（如 `<>`）通常在比较的一侧为 `NULL` 时返回 `NULL`。如果你的源系统允许空属性，请使用空值安全比较（例如，`COALESCE(d.address, '') <> COALESCE(s.address, '')`）来确保准确检测变更。
- **哈希比较:** 如果维度有 50 个属性，单独比较每个列可能会冗长且慢。一种常见的优化方法是计算源和目标中所有类型 2 列的哈希值（如 MD5 或 SHA256）。然后你只需比较 `source_hash <> target_hash` 来检测变更。
- **日期粒度:** 确保你的 `start_date` 和 `end_date` 与事实数据的粒度保持一致。如果你的事实数据包含时间戳，你的维度生效日期也可能需要是时间戳，以避免在变更边界处发生的事件产生歧义。

## 参考资料

- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: John Wiley & Sons, Inc.; Pages: 600
  这本书是维度建模的主要资源，为SCD（包括Type 2）的模式设计、实现考量和查询策略提供了详细指导。
- [Implementing Slowly Changing Dimensions (SCDs)](https://medium.com/snowflake/slowly-changing-dimensions-with-dynamic-tables-f7a9359e9a4) — Sasha Lionheart (2023)
  Publisher: Snowflake Builders Blog
  使用SQL在现代云数据仓库环境中，为SCD（包括Type 2）的实现提供了实用示例和常见模式。
- [Fundamentals of Database Systems](https://www.pearson.com/us/higher-education/program/Elmasri-Fundamentals-of-Database-Systems-7th-edition/PGM175344.html) — Ramez Elmasri, Shamkant B. Navathe (2021)
  Publisher: Pearson
  一本全面的教科书，涵盖数据库系统概念、关系数据建模、SQL和模式设计原则。
