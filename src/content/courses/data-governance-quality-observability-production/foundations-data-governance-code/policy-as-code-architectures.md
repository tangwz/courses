---
course: "data-governance-quality-observability-production"
chapter: "foundations-data-governance-code"
lesson: "policy-as-code-architectures"
sourceId: 8079
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-1-foundations-data-governance-code/policy-as-code-architectures"
title: "策略即代码架构"
description: "用于将数据策略实现为代码的架构模式。"
order: 4
plots: ["plots/8079-0.json"]
sourceHash: "06fb4c35b43e0ab770106e4c52295a396f0f6e2887cd942b07cb968a43539ad5"
sourceCorrections: []
---

传统治理依赖静态文档和人工审查来执行标准。现代数据工程环境中数据变化的快速速度超出了人工审计的能力，使得这种方法无法扩展。策略即代码（PaC）将治理从一个繁琐的障碍转变为一个架构组成部分。通过在软件中定义策略，我们可以像处理应用程序逻辑一样，对合规性检查进行版本控制、测试和自动化。

### 解耦架构

策略即代码的基本原则是将策略逻辑与执行它的底层系统分离。在耦合架构中，数据管道可能包含硬编码逻辑，例如 `if user_role == 'analyst': drop_column('ssn')`。这会使治理规则分散在整个代码库中，难以审计或更新。

在PaC架构中，数据系统（执行者）查询一个中央决策点（策略引擎）以确定某个操作是否被允许。管道将当前上下文 (context)发送给引擎，引擎根据定义的策略评估此输入，并返回一个决策。

这种关系可以用数学方式建模。令 $I$ 表示输入上下文（例如，用户元数据、表结构），$P$ 表示策略逻辑。策略引擎 $E$ 作为确定性评估器：

$D = E(I, P)$

其中 $D$ 是结果决策，通常是一个布尔值（允许/拒绝）或一个包含义务的结构化对象（例如，“允许，但遮蔽列X”）。

> 解耦策略架构中的信息流。引擎作为纯函数，根据静态规则处理输入以生成决策。

### 策略执行的类型

在为数据平台设计治理时，通常在两个不同阶段实施策略：静态分析和运行时执行。

#### 静态分析（设计阶段）

静态分析在代码或配置文件部署之前对其进行评估。这主要发生在持续集成（CI）管道中。例如，如果数据工程师通过Terraform或dbt提交了一个添加新表的拉取请求，策略引擎会扫描定义文件。它会检查以下要求：

- 该表是否具有分类标签？
- 保留期限是否已定义？
- 表结构是否包含禁用列名？

如果检查失败，管道将停止。这可以防止不合规的基础设施出现在生产环境中。

#### 运行时执行

运行时执行发生在系统运行期间。这对于在代码审查阶段无法预见的动态条件是必需的，例如用户访问模式或数据文件的具体内容。

例如，对象存储桶策略可能仅在传入文件大小低于特定限制且文件格式为Parquet时才允许写入访问。存储服务在上传请求到达时向策略引擎请求决策。

### Python中的实现模式

尽管Rego（被Open Policy Agent使用）等专用语言在PaC中很常见，但您可以使用标准Python类实现有效的策略架构。这通常使数据团队更容易采纳并整合到现有的Airflow或Prefect工作流中。

A模式是定义一个 `Policy` 基类，并为不同的治理方面创建具体的实现。

```python
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class PolicyResult:
    passed: bool
    reason: str

class DataPolicy:
    """治理策略的基类。"""
    def evaluate(self, context: Dict[str, Any]) -> PolicyResult:
        raise NotImplementedError

class EncryptionPolicy(DataPolicy):
    """强制特定列必须加密。"""

    def __init__(self, sensitive_fields: List[str]):
        self.sensitive_fields = sensitive_fields

    def evaluate(self, context: Dict[str, Any]) -> PolicyResult:
        schema = context.get('schema', {})
        encryption_metadata = context.get('encryption', {})

        for field in self.sensitive_fields:
            if field in schema:
                # 检查字段是否在元数据中标记为已加密
                if not encryption_metadata.get(field, False):
                    return PolicyResult(
                        passed=False, 
                        reason=f"字段 '{field}' 包含敏感数据但未加密。"
                    )

        return PolicyResult(passed=True, reason="加密标准符合要求。")

# 管道中的使用示例
current_dataset_context = {
    "schema": ["user_id", "email", "transaction_total"],
    "encryption": {"user_id": True} # email 缺少加密
}

policy = EncryptionPolicy(sensitive_fields=["email", "ssn"])
result = policy.evaluate(current_dataset_context)

if not result.passed:
    # 在实际管道中，这会引发异常或触发警报
    print(f"Compliance Block: {result.reason}")
```

在这种基于Python的方法中，策略逻辑位于 `EncryptionPolicy` 中。管道代码只需实例化策略并运行 `evaluate`。如果组织决定更改加密要求，工程师会更新策略类，所有使用该策略的管道在下次运行时会自动继承新规则。

### 分层策略管理

随着数据平台的发展，扁平的策略列表变得难以管理。一个有效的架构会分层组织策略。您可能拥有适用于每个数据集的全局策略（例如，“所有表都必须有一个所有者”）以及适用于业务单元的特定策略（例如，“财务表必须保留7年”）。

该架构必须支持策略聚合。当收到请求时，引擎会聚合所有相关策略。如果任何一个“拒绝”策略被触发，则整个操作将被阻止。这种“默认拒绝”或“仅在所有通过时允许”的策略是安全数据系统的标准做法。



![策略评估逻辑](plots/8079-0.json)



> 单个请求必须通过多个独立策略层的逻辑流。任何层（红色路径）的失败都会导致拒绝。

通过将治理构建为代码，您可以创建一个合规性确定且可观察的系统。审计变成了检查策略文件的版本控制历史，而不是询问员工其手动流程。这种架构转变对于在分布式数据环境中保持可靠性是必需的。
