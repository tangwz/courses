---
course: "building-scalable-data-warehouses"
chapter: "governance-security-observability"
lesson: "rbac-hierarchies"
sourceId: 8041
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-5-governance-security-observability/rbac-hierarchies"
title: "基于角色的访问控制层级结构"
description: "构建细粒度权限模型和角色继承。"
order: 1
plots: ["plots/8041-0.json"]
sourceHash: "dec6ce2c678c93c48afd56562a4558a1d8e11ead919e6d90c161b20af06ba25a"
sourceCorrections: []
---

通过向单个用户分配明确的授权来管理权限，会造成维护瓶颈，阻碍系统扩展。分布式数据仓库环境中数据库对象数量庞大，通常包含数千张表和视图，使得手动访问管理几乎不可能。可扩展的架构并非将权限视为用户与数据之间的静态连接，而是依赖基于角色的访问控制（RBAC）层级结构。此方法将权限抽象为逻辑容器，使安全策略得以演进，而无需更新单个用户账户。

### 角色继承的机制

可扩展RBAC模型的主要构成是继承原则。我们将角色视为有向图中的节点，而非扁平的权限集合。一个角色可以拥有特定权限，同时也能继承其他角色的权限。

从数学角度看，我们可以使用集合论来定义这种关系。令 $P$ 代表仓库中所有可用权限的集合。角色 $R_x$ 是权限的一个子集，即 $R_x \subseteq P$。如果建立角色A继承角色B的继承关系，则角色A的有效权限，表示为 $E(A)$，定义为：

$E(A) = P(A) \cup E(B)$

这种传递性使得工程师能够从简单的构成单元构建复杂的权限结构。如果角色A继承角色B，且角色B继承角色C，则角色A自动拥有授予角色C的所有权限。此结构最大限度地减少了冗余；您只需定义一次特定的权限集，便可多次引用。

下图显示了一个标准的继承层级结构。请注意权限流如何从数据对象向上移动，而角色分配则向下流向用户。

> 权限继承流从原始数据对象经过功能层和访问层向上延伸至最终用户。

### 功能角色与访问角色

为在大规模环境中保持秩序，将“个人身份”与“存在的数据”分离很有必要。这引出了一种双层策略：访问角色和功能角色。

**访问角色**是纯粹技术性的。它们与数据库对象的逻辑组合一一对应，通常在模式（schema）层面。访问角色不应关注其使用者；其唯一目的是封装一组特定权限（例如，对 `SALES` 模式的 `READ` 访问）。此处的命名约定有明确规定，通常遵循 `AR_<SCHEMA>_<PRIVILEGE>` 这样的模式。

**功能角色**对应于工作职能或业务单元。这些角色与用户的身份相符（例如，`初级分析师`、`财务经理`、`ETL服务账户`）。功能角色很少包含直接授予数据库对象的权限。相反，它们由访问角色构成。

这种分离简化了审计和入职流程。当引入新的数据集 `MARKETING` 时，您创建一个访问角色 `AR_MARKETING_READ`。然后，将此单一访问角色授予相应的业务功能角色（`FR_MARKETING_ANALYST`、`FR_CMO`）。您无需更改单个用户账户。反之，当新的数据工程师入职时，您只需授予他们单一的 `FR_DATA_ENGINEER` 角色，他们便会立即继承所有必要的底层访问权限。

### 实现细粒度继承

在Snowflake或PostgreSQL等平台上使用SQL实现时，其语法在于将角色授予其他角色。`SECURITYADMIN` 或同等高级管理员通常执行这些命令。

设想一个场景，我们需要为财务团队建立一个层级结构，该团队需要生产数据的读取权限和特定建模模式的写入权限。

```sql
-- 1. 创建访问角色 (技术层)
CREATE ROLE ar_finance_prod_read;
CREATE ROLE ar_finance_modeling_write;

-- 2. 授予对象权限给访问角色
-- 注意: 在生产环境中，这通常通过Terraform或dbt实现自动化
GRANT USAGE ON DATABASE finance_db TO ROLE ar_finance_prod_read;
GRANT USAGE ON SCHEMA finance_db.prod TO ROLE ar_finance_prod_read;
GRANT SELECT ON ALL TABLES IN SCHEMA finance_db.prod TO ROLE ar_finance_prod_read;

GRANT USAGE ON DATABASE finance_db TO ROLE ar_finance_modeling_write;
GRANT USAGE ON SCHEMA finance_db.modeling TO ROLE ar_finance_modeling_write;
GRANT ALL PRIVILEGES ON SCHEMA finance_db.modeling TO ROLE ar_finance_modeling_write;

-- 3. 创建功能角色 (人员层)
CREATE ROLE fr_finance_analyst;
CREATE ROLE fr_finance_manager;

-- 4. 构建层级结构
-- 分析师获得生产环境的读取权限和建模环境的写入权限
GRANT ROLE ar_finance_prod_read TO ROLE fr_finance_analyst;
GRANT ROLE ar_finance_modeling_write TO ROLE fr_finance_analyst;

-- 经理继承分析师的所有权限，并可能拥有更多
GRANT ROLE fr_finance_analyst TO ROLE fr_finance_manager;

-- 5. 分配给用户
GRANT ROLE fr_finance_analyst TO USER john_doe;
```

在此配置下，如果我们判定 `ar_finance_modeling_write` 角色的权限过高，并撤销 `ALL PRIVILEGES` 授权，则该限制会立即传播到 `fr_finance_analyst`、`fr_finance_manager` 和用户 `john_doe`。

### 计算与存储权限的处理

在将计算与存储分离的MPP架构中，权限管理必须兼顾两者。用户可能拥有从表（存储）中选择数据的权限，但若无权使用虚拟仓库（计算），他们便无法执行查询。

将计算权限与数据访问角色捆绑是一种常见的不良模式。这会造成资源争用，例如报告角色意外使用了一个专为高优先级加载任务预留的仓库。

RBAC层级结构将计算仓库视为拥有自己访问角色的独立对象。您可以定义 `AR_WH_REPORTING_USAGE` 和 `AR_WH_LOADING_USAGE`。然后，这些角色被组合到功能角色中。

$\text{分析师角色} = \{ \text{数据访问角色} \} \cup \{ \text{计算使用角色} \}$

这种模块化设计确保了，即使数据科学家和BI工具都访问相同的底层数据表，它们也被强制使用不同的计算资源，从而避免了繁重的临时查询降低管理层仪表盘的性能。

### 最小权限原则与角色爆炸

细粒度RBAC的一个潜在副作用是“角色爆炸”，即角色数量变得难以追踪。为减轻此问题，请在访问角色层面而非功能角色层面应用最小权限原则。

除非行级安全策略确有必要，否则应避免为单个表创建访问角色。相反，应以模式（schema）级别为目标。如果一个模式包含具有不同敏感级别的数据，这表明数据模型本身需要重构，将敏感数据（如PII）分离到安全模式中，而非通过使RBAC模型复杂化来处理异常情况。

下图显示了角色数量与管理开销之间的关系。分层结构即使在用户基数增长时也能保持开销恒定，而直接分配则导致管理繁复程度呈指数级增长。



![管理开销: 直接分配 vs. RBAC层级结构](plots/8041-0.json)



> 直接用户授权与分层角色分配之间复杂程度的增长的比较。

通过遵循严密的层级结构，您可以有效地将招聘速度（增加用户）与数据工程速度（增加表）解耦，使两者能够独立扩展，而不会降低安全性。

## 参考资料

- [Security Engineering: A Guide to Building Dependable Distributed Systems](https://www.cl.cam.ac.uk/~rja14/book.html) — Ross J. Anderson (2020)
  Publisher: Wiley
  第3版。一本关于安全工程的全面书籍，涵盖了RBAC等访问控制模型、其设计以及复杂系统中的挑战。
