---
course: "data-governance-quality-observability-production"
chapter: "foundations-data-governance-code"
lesson: "practice-implementing-access-policies"
sourceId: 8080
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-1-foundations-data-governance-code/practice-implementing-access-policies"
title: "实践：实施访问策略"
description: "在Python中动手实践定义和执行数据访问策略。"
order: 5
plots: []
sourceHash: "bc5cd7c98eb88aed80bd2cf599dcec58585f359cf3067df23ca3da56af7bb422"
sourceCorrections: []
---

通过手动数据库命令或用户界面点击管理访问控制，不可避免地会造成配置漂移。可靠的数据工程环境将访问策略定义为代码对象，并通过程序化方式应用它们。这让权限得以版本控制、可测试，并能在不同环境中保持一致。

在本实践章节中，我们将构建一个基于Python的策略引擎。该引擎将用户的角色和数据集的元数据作为输入，然后动态生成一个安全的SQL查询。这种方法常被称为“动态视图”或“查询重写”，是实现可扩容治理的一个基本模式。

### 定义策略对象

第一步是将策略定义从数据库移出，放入配置结构中。这里我们使用字典结构，但在生产环境中，这可能是一个在git中管理的YAML文件。我们定义角色并将其映射到允许的数据标签。

这个模型基于用户属性（角色）和数据属性（标签）的交集。如果用户的角色允许访问特定标签，他们就可以查询相关的列。

```python
from typing import Dict, List, Optional
from dataclasses import dataclass

# 定义特定标签允许的访问级别
# 具有“analyst”角色的用户只能查看“public”和“internal”数据
ACCESS_POLICIES = {
    "analyst": {"allowed_tags": {"public", "internal"}},
    "engineer": {"allowed_tags": {"public", "internal", "confidential"}},
    "admin": {"allowed_tags": {"public", "internal", "confidential", "pii"}}
}

@dataclass
class ColumnMetadata:
    name: str
    tag: str
    dtype: str

# 定义带有治理标签的表 schema
users_table_schema = [
    ColumnMetadata(name="user_id", tag="public", dtype="int"),
    ColumnMetadata(name="email", tag="pii", dtype="string"),
    ColumnMetadata(name="country", tag="public", dtype="string"),
    ColumnMetadata(name="salary_band", tag="confidential", dtype="string"),
    ColumnMetadata(name="created_at", tag="internal", dtype="timestamp"),
]
```

### 策略执行引擎

一旦定义确定，我们就需要一种执行它们的方式。我们将实现一个充当转译器的函数：它读取schema和请求者的角色，然后构建一个SQL语句。

此逻辑遵循一个简单的集合操作。设 $R$ 为角色允许的标签集合，设 $t_c$ 为与列 $c$ 关联的标签。如果满足以下条件，则按原样选择该列：

$t_c \in R$

如果条件不满足，我们有两种架构选择：完全排除该列，或者对数据进行掩码处理（返回`NULL`或哈希值）。在生产分析中，掩码处理通常更受青睐，以避免当某一列突然从视图schema中“消失”时造成代码中断。

> 策略引擎结合角色定义和schema元数据生成合规SQL的流程。

以下代码实现了这一逻辑。它遍历schema，并判断是暴露原始列还是掩码字面量。

```python
def generate_secure_query(table_name: str, schema: List[ColumnMetadata], role: str) -> str:
    """
    生成一个基于RBAC的动态掩码SELECT语句。
    """
    if role not in ACCESS_POLICIES:
        raise ValueError(f"Role '{role}' is not defined in policies.")

    allowed_tags = ACCESS_POLICIES[role]["allowed_tags"]
    select_clauses = []

    for col in schema:
        if col.tag in allowed_tags:
            # 用户有权访问，选择原始列
            select_clauses.append(col.name)
        else:
            # 用户无权访问，掩码数据
            # CAST NULL确保视图schema保持一致
            select_clauses.append(f"CAST(NULL AS {col.dtype}) AS {col.name}")

    query_cols = ", ".join(select_clauses)
    return f"SELECT {query_cols} FROM {table_name};"

# 示例用法：
print("--- 分析师视图 ---")
print(generate_secure_query("users", users_table_schema, "analyst"))

print("\n--- 管理员视图 ---")
print(generate_secure_query("users", users_table_schema, "admin"))
```

### 分析输出

当你运行上述脚本时，你会看到治理逻辑如何改变生成的SQL。

对于只有`public`和`internal`访问权限的**分析师**，输出会掩盖PII和机密列：

```sql
SELECT user_id, 
       CAST(NULL AS string) AS email, 
       country, 
       CAST(NULL AS string) AS salary_band, 
       created_at 
FROM users;
```

对于拥有所有权限的**管理员**，查询会透明地获取所有数据：

```sql
SELECT user_id, email, country, salary_band, created_at FROM users;
```

### 在基础设施中实现策略即代码

虽然生成字符串对于了解逻辑很有用，但在实际环境中，你使用基础设施即代码工具来应用此方法。

1. **视图生成：** 在迁移脚本（例如，Python的Alembic或dbt宏）中使用上述函数来创建名为`users_analyst`或`users_admin`的受限视图。
2. **行级安全性 (RLS)：** 大多数现代数据仓库（Snowflake、PostgreSQL、BigQuery）都支持RLS。你的Python策略引擎可以生成供应商所需的特定`CREATE ROW POLICY`语句，而不是使用`CAST(NULL...)`。
3. **CI/CD 集成：** 通过在Python中定义策略，你可以为你的治理编写单元测试。在代码部署到生产环境之前，你可以断言特定角色*绝不会*看到标记 (token)为`pii`的列。

这种程序化方法将安全性的*定义*（字典）与安全性的*实现*（SQL生成）分离。这种分离使你能够统一更新策略，而无需手动重写数百个SQL查询。

## 参考资料

- [PostgreSQL: Documentation: 16: 5.8. Row Security Policies](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) — PostgreSQL Global Development Group (2025)
  PostgreSQL官方文档，解释如何在数据库中实现行级安全（RLS），这是一种基于查询重写直接在数据库中强制执行访问策略的实用机制。
- [Data Governance: The Definitive Guide](https://www.oreilly.com/library/view/data-governance/9781492063544/) — Evren Eryurek, Uri Gilad, Valliappa Lakshmanan, Anita Kibunguchy-Grant, Jessi Ashdown (2021)
  Publisher: O'Reilly Media
  关于数据治理原则和实践的综合指南，涵盖了现代数据环境中数据访问、安全和策略实施的策略。
