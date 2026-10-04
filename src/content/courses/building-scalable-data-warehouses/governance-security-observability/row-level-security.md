---
course: "building-scalable-data-warehouses"
chapter: "governance-security-observability"
lesson: "row-level-security"
sourceId: 8043
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-5-governance-security-observability/row-level-security"
title: "行级安全实现"
description: "编写安全谓词策略以在行级别限制数据访问。"
order: 3
plots: ["plots/8043-0.json"]
sourceHash: "548100db331a10674af20230c23c7f9b65960a0abbb53cd45fd30a1e57fe1243"
sourceCorrections: []
---

对象级别的安全，即控制谁可以查看哪个表，是必需但不足以应对现代多租户架构的。当数TB来自不同地区、部门或客户的数据共存于一个表中时，为每种权限组合创建单独视图将难以管理。行级安全 (RLS) 解决了这个问题，它将访问控制从容器（表）转移到内容（行）。

从根本上说，RLS 会在针对受保护表的每个 SQL 查询执行计划中注入一个谓词过滤器。这个过程发生在数据库引擎层，使最终用户和应用程序无感知。如果用户运行 `SELECT * FROM sales`，引擎实际执行的是：

$\sigma_{P(u, r)}(R)$

其中 $R$ 是关系（表），$u$ 是当前用户上下文 (context)，$P$ 是必须评估为 `TRUE` 的安全谓词。如果 $P$ 评估为 `FALSE` 或 `NULL`，该行将从结果集中排除。

### 设计安全谓词

RLS 的核心是策略逻辑。在 Snowflake 或 BigQuery 等大规模并行处理 (MPP) 系统中，您通常使用集中式策略对象或行访问策略函数来定义此逻辑。此函数根据当前会话的上下文 (context)返回一个布尔值。

最简单的实现方式是将数据中的某一列与会话属性进行比较。假设有一个全球销售表，经理只能查看其所属区域的交易。

$\text{能否访问}(r_{region}) = \begin{cases} \text{真} & \text{如果 } r_{region} = \text{当前区域}() \\ \text{假} & \text{否则} \end{cases}$

然而，将映射逻辑硬编码到策略函数中是脆弱的。一种可扩展的办法是使用一个映射表（通常称为权限表），将角色或用户与特定维度键关联。这使得安全管理员可以通过修改标准表中的行来更新访问权限，而不是更改 DDL（数据定义语言）对象。

> 数据库引擎在优化器生成最终执行计划之前，拦截传入的 SQL 请求并注入安全谓词。

### 映射表模式与连接性能

使用映射表时，RLS 策略会为每个查询实际在事实表和映射表之间执行一个 `SEMI-JOIN`。

```sql
-- 逻辑透明应用
SELECT f.*
FROM fact_sales f
WHERE EXISTS (
  SELECT 1 
  FROM security_mapping m
  WHERE m.user_id = CURRENT_USER()
    AND m.region_id = f.region_id
);
```

在 MPP 环境中，如果设计不当，此连接会成为一个严重的瓶颈。如果映射表很大，并且优化器决定广播事实表而不是映射表（或反之亦然），查询延迟会急剧增加。

为了缓解性能下降，请确保映射表高度压缩并有效缓存。一些平台允许您将策略函数定义为 `MEMOIZABLE`，这意味着当前用户的查找结果在查询或会话期间会被缓存。这避免了对每个扫描到的微分区重复查找映射表。

### 管理分层访问

在实现分层访问时会遇到一个常见问题，即经理需要访问自己的数据 *以及* 所有下属的数据。

行访问策略中的标准递归公共表表达式 (CTE) 可能会导致较高的计算成本。一种更高性能的策略是将层次结构“展平”到访问桥接表中。此表包含每个祖先-后代关系的一行。如果用户 A 管理用户 B，用户 B 管理用户 C，则桥接表会明确地将 A 与 C 关联。

这使得 RLS 策略保持为一个简单的等值连接，避免了查询运行时的递归。

### 对剪枝和缓存的影响

应用 RLS 改变了数据库引擎优化数据检索的方式。因为过滤器是根据用户上下文 (context)在运行时应用的，所以全局结果缓存（存储所有用户的查询结果）通常会失效。如果用户 A 和用户 B 运行完全相同的 SQL 语句，但应用了不同的 RLS 策略，系统无法将用户 A 的缓存结果提供给用户 B。

然而，分区剪枝仍然有效。如果 RLS 策略对聚簇列（例如 `date` 或 `region_id`）进行过滤，查询优化器仍然可以跳过不包含该特定用户相关数据的分区。



![RLS 实现对分区扫描的影响](plots/8043-0.json)



> 存储扫描效率对比。对非聚簇列应用 RLS 会强制进行全表扫描，而将 RLS 谓词与聚簇键对齐 (alignment)可恢复剪枝能力。

### 可观察性与审计

实现 RLS 引入了一层不透明性；用户可能会声称数据“丢失”，而实际上它只是被过滤掉了。为了保持可观察性，管理员必须具备模拟策略的能力。

大多数高级平台提供 `EXECUTE AS` 功能或策略模拟函数。这允许特权管理员验证特定用户将看到什么，而无需重置密码或以该用户身份登录。

此外，应监控审计日志以检测绕过 RLS 的企图。RLS 阻止了数据访问，但复杂的攻击可能会尝试通过侧信道攻击来推断数据值，例如，通过分析精心构造的 `WHERE` 子句中除零操作的错误消息。确保您的 RLS 策略在用户定义的过滤器 *之前* 进行评估，以防止这种信息泄露。安全数据仓库中的标准执行顺序保证了这一点，但这是架构设计中必需的验证步骤。

## 参考资料

- [Row Access Policies](https://docs.snowflake.com/en/user-guide/data-governance/understanding-row-access-policies) — Snowflake (2024)
  Publisher: Snowflake
  雪花数据仓库官方文档，详细介绍了在MPP数据仓库中实现行访问策略的语法和操作注意事项。
- [Efficient Fine-Grained Access Control for Data-Intensive Applications](https://dl.acm.org/doi/10.1145/3183713.3190669) — Kevin Nitsche, Wolfgang Lehner (2018)
  Journal: Proceedings of the 2018 ACM SIGMOD International Conference on Management of Data; Publisher: ACM; Pages: 1891-1894; DOI: [10.1145/3183713.3190669](https://doi.org/10.1145/3183713.3190669)
  一篇研究论文，讨论了数据密集型应用程序中高效细粒度访问控制的技术，与可扩展数据仓库中行级安全性的性能影响相关。
- [Data Governance: The Definitive Guide](https://www.oreilly.com/library/view/data-governance-the/9781492063490/) — Evren Eryurek, Uri Gilad, Valliappa Lakshmanan, Anita Kibunguchy-Grant, Jessi Ashdown (2021)
  Publisher: O'Reilly Media
  提供了数据治理策略和实践的背景知识，其中一些部分与在全面的数据治理框架内实施行级安全性等安全控制措施相关。
