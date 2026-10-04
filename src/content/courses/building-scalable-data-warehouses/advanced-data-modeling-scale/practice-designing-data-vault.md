---
course: "building-scalable-data-warehouses"
chapter: "advanced-data-modeling-scale"
lesson: "practice-designing-data-vault"
sourceId: 8030
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-2-advanced-data-modeling-scale/practice-designing-data-vault"
title: "动手实践：设计数据保险库"
description: "从原始数据源设计并实现数据保险库架构的实际练习。"
order: 5
plots: ["plots/8030-0.json"]
sourceHash: "42a0a6368f467511846b8141120f7e38c58f2aac1fa737583b226dda8c1b8411"
sourceCorrections: []
---

数据保险库模型要求数据仓库对关系和数据存储的看法发生根本转变。与为读取优化而预连接数据的星型模式不同，数据保险库优化了写入的灵活性和可审计性。正在为电子商务平台构建一个原始数据保险库。该系统必须从客户关系管理系统（客户数据）、企业资源规划系统（订单处理）和产品信息管理系统摄取数据。

目标是构建一个支持高速数据摄取并允许源结构演变而不破坏下游数据仓库的架构。我们将着重介绍三个核心实体：中心表（业务键）、链接表（关系）和卫星表（上下文 (context)/属性）。

### 定义架构

在大规模并行处理（MPP）环境（如Snowflake或BigQuery）中，我们避免为代理键使用序列或自增整数，因为它们需要一个中央协调点，这在并行加载期间会成为瓶颈。相反，数据保险库2.0依赖于业务键的加密哈希（MD5或SHA-256）。这使得在分布式节点上可以进行确定性键生成，无需查找现有记录。

以下图表概述了我们将实现的关系结构。请注意业务键如何隔离在中心表中，而时间上下文 (context)则存在于卫星表中。

> 该图将结构组件（中心表和链接表）与描述性组件（卫星表）分开。中心表连接到链接表以构成骨架，而卫星表则附加到中心表或链接表以提供详细信息。

### 实现中心表

中心表是业务对象的入口点。它仅包含哈希键、加载日期、记录来源和原始业务键。此表很少更改。

对于我们的客户实体，业务键是 `Customer_ID`。我们通过对修剪后、大写的业务键进行哈希处理来生成 `Hub_Customer_HK`（哈希键）。

$H_{键} = SHA256(UPPER(TRIM(业务键)))$

我们选择SHA-256是因为其低冲突概率，这在处理TB级数据时是必要的。

**HUB\_CUSTOMER 的 SQL 实现：**

```sql
CREATE TABLE HUB_CUSTOMER (
    HUB_CUSTOMER_HK   BINARY(32)     NOT NULL, -- 主键
    CUSTOMER_ID       VARCHAR(100)   NOT NULL, -- 业务标识符
    LOAD_DT           TIMESTAMP_NTZ  NOT NULL,
    RECORD_SOURCE     VARCHAR(50)    NOT NULL,
    CONSTRAINT PK_HUB_CUSTOMER PRIMARY KEY (HUB_CUSTOMER_HK)
);
```

加载此表时，您需要使用暂存区域中的 `DISTINCT` 查询。因为哈希是确定性的，您可以执行反连接（`LEFT JOIN ... WHERE target.HK IS NULL`）来只插入新的业务键。

### 实现链接表

链接表代表关联或事务。在我们的场景中，一个订单将客户与一个事务关联起来。`LINK_ORDER_CUSTOMER` 表解决了 `HUB_ORDER` 和 `HUB_CUSTOMER` 之间的多对多或一对多关系。

链接表包含其自身的哈希键、所引用中心表的哈希键以及审计元数据。

**LINK\_ORDER\_CUSTOMER 的 SQL 实现：**

```sql
CREATE TABLE LINK_ORDER_CUSTOMER (
    LNK_ORD_CUST_HK   BINARY(32)     NOT NULL,
    HUB_ORDER_HK      BINARY(32)     NOT NULL,
    HUB_CUSTOMER_HK   BINARY(32)     NOT NULL,
    LOAD_DT           TIMESTAMP_NTZ  NOT NULL,
    RECORD_SOURCE     VARCHAR(50)    NOT NULL,
    CONSTRAINT PK_LNK_ORD_CUST PRIMARY KEY (LNK_ORD_CUST_HK),
    CONSTRAINT FK_LNK_ORD FOREIGN KEY (HUB_ORDER_HK) REFERENCES HUB_ORDER (HUB_ORDER_HK),
    CONSTRAINT FK_LNK_CUST FOREIGN KEY (HUB_CUSTOMER_HK) REFERENCES HUB_CUSTOMER (HUB_CUSTOMER_HK)
);
```

要生成 `LNK_ORD_CUST_HK`，您需要连接参与中心表的业务键（以排序顺序确保一致性），并对结果进行哈希处理。

$H_{链接} = SHA256(Sort(键_{订单}, 键_{客户}))$

### 实现卫星表

卫星表存储数据随时间变化的状态。它们是唯一存储可变属性（如客户地址或订单状态）的地方。卫星表必须有一个父级（通常是中心表），并使用父级的哈希键作为其主标识符，与加载日期结合使用。

为了在大规模并行处理系统（MPP）中高效地检测变化，我们计算一个 `HashDiff`。这是行中所有描述性列的哈希值。在加载过程中，我们将传入的 `HashDiff` 与目标卫星表中最新的 `HashDiff` 进行比较。如果它们不同，则插入新行。这消除了昂贵的逐列比较。

**SAT\_CUSTOMER\_CRM 的 SQL 实现：**

```sql
CREATE TABLE SAT_CUSTOMER_CRM (
    HUB_CUSTOMER_HK   BINARY(32)     NOT NULL,
    LOAD_DT           TIMESTAMP_NTZ  NOT NULL,
    HASH_DIFF         BINARY(32)     NOT NULL, -- 用于变更检测
    FIRST_NAME        VARCHAR(100),
    LAST_NAME         VARCHAR(100),
    EMAIL             VARCHAR(255),
    ADDRESS_JSON      VARIANT,       -- 存储半结构化地址数据
    RECORD_SOURCE     VARCHAR(50)    NOT NULL,
    CONSTRAINT PK_SAT_CUST PRIMARY KEY (HUB_CUSTOMER_HK, LOAD_DT),
    CONSTRAINT FK_SAT_HUB FOREIGN KEY (HUB_CUSTOMER_HK) REFERENCES HUB_CUSTOMER (HUB_CUSTOMER_HK)
);
```

#### HashDiff 的计算

`HASH_DIFF` 确保我们只存储实际的变化。此公式适用于所有描述性列：

$H_{差异} = SHA256(COALESCE(列_1, '') || '|' || COALESCE(列_2, '') || ...)$

我们使用分隔符（如 `|`）并处理 NULL 值，以确保 `NULL` + `A` 与 `A` + `NULL` 的处理方式不同。

### 加载模式与并行处理

我们刚刚实现的这个设计实现了高度并行。

1. **暂存：** 原始数据进入暂存表。
2. **哈希生成：** 我们在暂存表之上的视图中计算所有哈希键和 Hash Diffs。
3. **中心表/链接表加载：** 我们并行加载中心表和链接表。由于链接表仅依赖于中心表键（我们在暂存阶段生成），技术上我们无需等待中心表加载完成即可准备链接数据，尽管数据库中的引用完整性约束可能要求中心表首先提交。
4. **卫星表加载：** 卫星表通过比较暂存 `HashDiff` 与当前卫星 `HashDiff` 来加载。

这种“仅插入”架构创建了完整的审计追踪。我们从不更新行。要获取客户的当前状态，您需要查询卫星表以获取特定 `HUB_CUSTOMER_HK` 的最大 `LOAD_DT` 所在的行。

### 可视化存储影响

在列式存储系统中，与宽表相比，这种分解（范式化）可能显得过多。然而，大规模并行处理系统会单独压缩列。卫星表中哈希键的重复压缩效果非常好（游程编码），并且分离避免了锁定。

以下图表显示了成熟数据保险库中常见的存储分布。卫星表由于历史追踪而占用大部分存储空间，而中心表和链接表则保持为轻量级索引结构。



![数据保险库中常见的存储分布](plots/8030-0.json)



> 卫星表不可避免地会变得最大，因为它们保留了属性变化的完整历史记录。中心表和链接表一旦核心业务实体和关系建立，就会稳定下来。

通过实施这种结构，您可以确保，如果客户关系管理系统向客户资料添加新字段，您只需向卫星表添加一列（或创建一个新的卫星表），而无需重建中心表或链接表。这种适应性是数据保险库在PB级数据仓库中的主要工程优势。

## 参考资料

- [Building a Scalable Data Warehouse with Data Vault 2.0](https://www.elsevier.com/books/building-a-scalable-data-warehouse-with-data-vault-2-0/linstedt/978-0-12-802510-9) — Daniel Linstedt, Michael Olschimke (2015)
  Publisher: Morgan Kaufmann
  Data Vault 2.0 方法论的权威指南，涵盖其原则、架构（中心、链接、卫星）以及可扩展和可审计数据仓库的实现。
- [The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling](https://www.wiley.com/en-us/The+Data+Warehouse+Toolkit%3A+The+Definitive+Guide+to+Dimensional+Modeling%2C+3rd+Edition-p-9781118530801) — Ralph Kimball, Margy Ross (2013)
  Publisher: Wiley
  维度建模和星型模式的奠基性文本，为理解数据仓库建模的设计选择和权衡提供了有价值的背景。
