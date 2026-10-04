---
course: "building-scalable-data-warehouses"
chapter: "governance-security-observability"
lesson: "dynamic-masking-tokenization"
sourceId: 8042
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-5-governance-security-observability/dynamic-masking-tokenization"
title: "动态数据屏蔽与令牌化"
description: "通过基于策略的动态屏蔽和令牌化保护个人身份信息。"
order: 2
plots: ["plots/8042-0.json"]
sourceHash: "f43e46ac7831997001477cacbf4219758d0e973e86a6b8218c88980af3ee815a"
sourceCorrections: []
---

在多 TB 级别的数据仓库中保护敏感数据，是一项具体的工程难题：即在存储效率与精细访问控制之间取得平衡。传统系统常要求工程师创建多个视图或物理复制表，以便对特定用户群体隐藏个人身份信息 (PII)。这种方法会增加存储成本并提高管道维护的难度。现代大规模并行处理 (MPP) 平台通过动态数据屏蔽 (DDM) 和令牌化来应对此情况，使您能够在查询时模糊敏感数据，而无需更改底层物理存储。

### 基于策略的架构

动态数据屏蔽作为逻辑层，位于存储引擎和查询执行计划之间。当用户提交查询时，数据库引擎会拦截请求并评估与目标列关联的活动屏蔽策略。如果存在策略，引擎会重写查询执行计划，根据用户当前的角色层次结构应用转换函数。

我们可以将屏蔽策略在数学上定义为应用于数据值 $v$ 和请求角色 $r$ 的函数 $M$。设 $T$ 为转换逻辑：


$$
M(v, r) = \begin{cases} v & \text{if } r \in R_{\text{特权}} \\ T(v) & \text{if } r \notin R_{\text{特权}} \end{cases}
$$


这里，$R_{\text{特权}}$ 表示被授权查看原始数据的角色集合。转换 $T(v)$ 可能返回一个静态字符串、一个值的局部减少形式或一个加密哈希。

下图以可视化方式呈现了访问被屏蔽列时的查询生命周期。策略引擎在优化器生成最终分布式执行计划之前注入条件逻辑。

> 此拦截流程说明了策略引擎如何在基于成本的优化器 (CBO) 确定最终执行计划之前注入条件逻辑。

### 实现列级屏蔽

在实际应用中，您使用 SQL 数据定义语言 (DDL) 实现这些策略。策略被定义一次，然后应用于不同表中的多个列。这推动了“定义一次，多处应用”的模式，确保数据仓库中的一致性。

考虑一个必须保护电子邮件地址的场景。分析师应看到一个屏蔽的域名，而 `PII_ADMIN` 角色则看到完整的地址。

```sql
-- 定义屏蔽策略
CREATE OR REPLACE MASKING POLICY email_mask AS (val string) 
RETURNS string ->
  CASE
    WHEN current_role() IN ('PII_ADMIN', 'ACCOUNTADMIN') THEN val
    -- 正则表达式替换，仅为分析师显示域名
    ELSE REGEXP_REPLACE(val, '^.*@', '*****@')
  END;

-- 将策略应用于表列
ALTER TABLE customers MODIFY COLUMN email 
SET MASKING POLICY email_mask;
```

当具有 `ANALYST` 角色的用户执行 `SELECT email FROM customers` 时，数据库不会返回微分区中的原始字节。相反，它会即时计算 `REGEXP_REPLACE` 函数。此操作需要 CPU 周期。尽管对于简单的标量函数，开销通常可以忽略不计，但涉及正则表达式或大量类型转换的复杂屏蔽逻辑可能会带来延迟，尤其是在扫描数十亿行时。

### 令牌化策略

屏蔽模糊了数据以供显示，而令牌化则用非敏感的替代值（或“令牌”）替换敏感数据，该令牌保留了原始数据的特定属性。当数据需要保持可连接或有分析价值，同时不泄露底层身份时，令牌化是必要的。

与 MPP 系统相关的令牌化主要有两种类型：

1. **不可逆令牌化（哈希）：** 我们使用加密哈希函数（如 SHA-256）来转换输入。这是确定性的；相同的输入总是产生相同的输出。这允许分析师对哈希值执行 `GROUP BY` 或 `JOIN` 操作，而无需看到明文。
2. **可逆令牌化：** 这涉及从安全存储库中用令牌替换值，或使用格式保留加密 (FPE)。

#### 用于分析的确定性加密

为了在不泄露 PII 的情况下维护数据仓库中的引用完整性，确定性加密是首选。如果 `user_id` 12345 在订单表中变为 `abcde`，那么它在支持工单表中也必须变为 `abcde`。

当通过用户定义函数 (UDF) 或外部函数实现此功能时，您必须考虑对优化器的性能影响。


$$
E_{\text{确定性}}(x) = E_{\text{确定性}}(y) \iff x = y
$$


此属性允许数据库引擎执行相等性检查。然而，这会造成频率分析攻击的弱点。如果攻击者知道“加利福尼亚”是数据集中最常见的州，并且令牌 `XYZ` 是州列中最常见的令牌，他们就可以推断 `XYZ` = “加利福尼亚”。为了减轻这种情况，我们通常引入“盐值”或使用高基数列进行令牌化，而不是性别或州等低基数属性。

### 外部令牌化与性能

对于高安全要求（例如 PCI-DSS），令牌化逻辑通常位于数据仓库外部的专用服务中（例如 Protegrity、Voltage）。数据仓库通过外部函数（API 调用）与这些服务通信。

对每行数据使用外部函数进行令牌化是一个显著的瓶颈。在批次摄取 1 亿行数据时，向令牌化服务器发出 1 亿次 HTTP 请求将导致管道超时或运行数天。

为了优化此过程，请在外部函数中使用**向量 (vector)化**或**批处理**。数据仓库将一批行（例如 5 MB 数据）以单个有效载荷发送到令牌化服务，该服务返回一批令牌。

下图以可视化方式呈现了本地屏蔽（内部）和外部令牌化之间的延迟权衡。



![屏蔽策略的相对性能开销](plots/8042-0.json)



> 相较于标量外部调用，原生函数的开销极小。批处理外部调用大幅减少了网络延迟，但仍比原生执行慢。

### 对查询优化和剪枝的影响

动态数据屏蔽的一个显著后果是可能中断分区剪枝。现代数据仓库使用元数据（微分区的最大/最小值）来跳过与查询谓词不匹配的数据。

如果用户在被屏蔽的列上进行筛选：

```sql
SELECT * FROM customers WHERE email = 'john.doe@example.com';
```

如果 `email` 列对此用户进行了屏蔽，数据库无法将字面值 `'john.doe@example.com'` 与存储的数据进行比较，因为用户无权查看存储数据。引擎必须扫描所有分区，应用屏蔽，然后比较结果（由于屏蔽输出是 `*****@example.com`，这很可能无论如何都无法匹配）。

为了在过滤列上强制执行安全性的同时保持性能，您有两种架构选择：

1. **策略感知筛选：** 某些平台允许您定义策略，允许对原始值进行查找，但在 `SELECT` 列表中返回屏蔽值。
2. **辅助列：** 为敏感列（例如 `email_hash`）创建确定性哈希，并将其提供给分析师。分析师根据哈希进行筛选，而原始 `email` 列则保持严格屏蔽。

### 层级与继承

在复杂的组织中，用户可能拥有多个角色。数据工程师也可能是一个普通员工。大多数 MPP 系统根据会话的*当前活动角色*来确定活动的屏蔽策略，而不是所有分配角色的并集。

您必须设计您的基于角色的访问控制 (RBAC)，使角色层级与数据敏感性保持一致。如果用户切换到较低权限的角色上下文 (context)，屏蔽策略会立即生效。

当嵌套策略时（例如，基于带有策略的表构建的带有策略的视图），最接近数据的策略（表策略）通常优先。这可以防止视图创建者意外暴露在存储级别受保护的数据。

理解这些机制可确保您能够实施严密的合规措施，而不会降低数据仓库的高吞吐量 (throughput)能力。分布式系统中的安全性不仅仅是限制。它关乎在数学上保证隐私的同时，实现对数据效用的充分利用。

## 参考资料

- [Dynamic Data Masking](https://docs.snowflake.com/en/user-guide/data-masking-overview) — Snowflake Documentation (2024)
  Publisher: Snowflake
  提供在MPP数据仓库中实现DDM的官方指南和实践示例。
- [NIST Special Publication 800-38G: Recommendation for Block Cipher Modes of Operation: Methods for Format-Preserving Encryption](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-38G.pdf) — Morris Dworkin and Melvin Young (2016)
  Journal: NIST Special Publication; Publisher: National Institute of Standards and Technology (NIST); DOI: [10.6028/NIST.SP.800-38G](https://doi.org/10.6028/NIST.SP.800-38G)
  定义了格式保持加密（FPE）的密码学方法，这是本节讨论的一种可逆令牌化技术。
- [Dynamic data masking for privacy-preserving data processing](https://www.sciencedirect.com/science/article/pii/S074373152030198X/fulltext) — Bin Li, Mengyu Zhang, Jiying Wang, and Peng Liu (2021)
  Journal: Journal of Parallel and Distributed Computing; Publisher: Elsevier; Volume: 149; Pages: 140-150; DOI: [10.1016/j.jpdc.2020.12.001](https://doi.org/10.1016/j.jpdc.2020.12.001)
  讨论了动态数据屏蔽在隐私保护数据处理中的原理、架构和挑战，涵盖与本节内容相关的方面。
