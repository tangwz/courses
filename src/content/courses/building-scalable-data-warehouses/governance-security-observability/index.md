---
course: "building-scalable-data-warehouses"
sourceUrl: "https://apxml.com/zh/courses/building-scalable-data-warehouses/chapter-5-governance-security-observability"
sourceId: 1403
chapter: "governance-security-observability"
title: "治理、安全与可观测性"
order: 5
description: "实施企业级安全、RBAC、行级安全以及成本监控。"
hasQuiz: false
---

数据仓库的扩展不只是管理存储容量或优化查询延迟。随着数据量的增长，管理访问和监控资源消耗的难度也随之增加。如果没有严密的管控措施，高性能系统很快会成为安全隐患或财务负担。

本章侧重于维持安全和可观测环境所需的运营框架。我们将审视如何构建基于角色的访问控制（RBAC）以高效管理权限，避免通常与手动用户管理相关的管理开销。您将学会实施精细化的安全措施，特别是用于保护敏感字段的动态数据屏蔽和行级安全（RLS）。例如，我们将定义策略，即查询仅当用户的属性 $A$ 与数据的安全标签 $T$ 匹配时才返回结果集 $R$，从而确保用户只能访问与其角色相关的数据。

我们还将处理运行大规模并行处理（MPP）系统的财务方面。本内容涉及可观测性的方法，包括跟踪信用额度使用情况和配置自动化资源监控器。您将实施配额和自动暂停策略，以防止基于消费的云模型中常见的预算超支。到本节结束时，您将能够应用伴随数据基础设施扩展的自动化治理规则。
