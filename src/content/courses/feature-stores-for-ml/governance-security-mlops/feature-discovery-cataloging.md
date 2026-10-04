---
course: "feature-stores-for-ml"
chapter: "governance-security-mlops"
lesson: "feature-discovery-cataloging"
sourceId: 3430
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-5-governance-security-mlops/feature-discovery-cataloging"
title: "特征发现与编目"
description: "构建系统，帮助工程师和数据科学家轻松查找和理解可用特征。"
order: 6
plots: []
sourceHash: "d62372fb7fc26ccaac910c00fff833cafcdc82a7c865f4425b7efb9c28185cee"
sourceCorrections: []
---

随着特征存储扩展到支持数百甚至数千个由多个团队开发的特征，仅仅存储特征是不够的。数据科学家和机器学习 (machine learning)工程师需要有效的机制来查找、理解并信任他们可用的特征。如果没有强大的特征发现能力，团队就有可能重复工作，例如重新创建已有特征、使用不一致的定义，或依赖质量或来源不明确的特征。本节讨论特征发现与编目系统的实现，这些系统对于最大化高级特征存储的价值和可用性必不可少。

一个精心设计的特征发现系统，通常表现为特征目录或注册表的用户界面，作为与存储中特征交互和理解的核心枢纽。它将特征存储从一个被动的存储库转变为一个活跃的、可搜索的清单。

## 可发现性的重要性

在复杂的机器学习 (machine learning)环境中，无法轻松查找相关特征会导致显著的效率低下：

1. **重复特征工程：** 不了解现有特征的团队可能会花费宝贵时间开发相似或相同的特征，浪费计算资源和工程精力。
2. **特征逻辑不一致：** 可能会出现多个版本的相似特征，实现时存在细微差别，这可能导致训练-服务偏差或难以诊断的模型行为变化。
3. **使用次优特征：** 用户可能会因为特征更容易查找而选择不那么合适的特征，或者在不知情的情况下使用过时、已弃用或存在已知质量问题的特征。
4. **新成员入职挑战：** 新团队成员在尝试理解可用特征时面临陡峭的学习曲线，如果缺乏集中式、可搜索的目录。

有效的发现机制可以缓解这些问题，促进特征重用，提高一致性，并提升机器学习团队的整体生产力。

## 特征目录的组成

特征目录提供了一个以用户为中心的特征存储内容视图。它以有组织、可搜索的方式聚合并呈现与特征、特征视图或特征组相关的元数据。通常包含的基本信息有：

### 核心元数据

- **唯一名称：** 特征的清晰、唯一标识符（例如，`user_7_day_transaction_count`）。
- **描述：** 关于特征代表什么、其目的以及如何计算的人类可读解释。这对可用性非常重要。
- **数据类型：** 物理数据类型（例如，`FLOAT`、`BIGINT`、`STRING`、`ARRAY<DOUBLE>`）。
- **所有者/团队：** 负责维护特征定义和质量的个人或团队。
- **创建/更新时间戳：** 特征定义创建和上次修改的时间。
- **状态：** 生命周期状态（例如，`EXPERIMENTAL` 实验性、`PRODUCTION` 生产、`DEPRECATED` 已弃用、`ARCHIVED` 已存档）。

### 语义和上下文 (context)信息

- **标签/关键词：** 可搜索的标签，表明所属方面（例如，`fraud` 欺诈、`recommendations` 推荐、`user_behavior` 用户行为）、数据源或项目。
- **特征组/视图：** 相关特征的逻辑分组，通常一起计算或服务于特定模型类型。
- **业务词汇表 (vocabulary)链接：** 连接到企业数据字典或业务术语定义。

### 运行和来源元数据

- **源数据：** 指向用于计算特征的上游数据源。
- **转换逻辑：** 定义特征计算管道的代码摘要或链接（在第2章中介绍）。
- **来源信息：** 追溯特征从原始数据衍生而来的可视化或链接（本章前面已讨论）。
- **在线/离线可用性：** 表明特征是在线存储、离线存储还是两者都可用。
- **新鲜度/更新频率：** 特征数据更新的频率（例如，每小时、每天、流式）。

### 质量和使用指标

- **数据质量分数：** 验证检查的汇总统计数据（在第3章中介绍），例如完成率或约束违反计数。
- **分布统计：** 根据最新数据计算的基本统计量（最小值、最大值、均值、中位数、分位数），以帮助理解。
- **陈旧信息：** 自上次更新以来的时间，特别是对于在线特征。
- **使用统计：** 关于哪些模型或下游流程使用该特征的信息（如果跟踪）。这有助于评估影响并识别流行或可能已被废弃的特征。

这是一个简化示例，说明单个特征的元数据如何以 YAML 格式组织：

```yaml
feature_name: user_7_day_transaction_count
version: 2
description: "计算用户过去 7 天内成功交易的数量，不包括保留和冲销。每日更新。"
owner_team: risk_analytics
status: PRODUCTION
data_type: INT64
tags: [欺诈, 用户行为, 交易]
feature_group: user_daily_aggregates
created_at: 2023-01-15T10:00:00Z
last_updated_at: 2023-05-20T14:30:00Z
sources:
  - db: transaction_logs
    table: completed_transactions
transformation_code: "git@github.com:org/feature-repo.git#transforms/user_aggs.py:L55"
lineage_id: "lineage:graph:node:feature:user_7_day_tx_count_v2"
availability: [在线, 离线]
update_frequency: 每日
quality_checks:
  - check: 非空
    status: 通过
  - check: 范围(0, 1000)
    status: 通过 (99.8% 符合)
```

## 启用特征发现

仅仅收集元数据是不够的；它必须可访问。有效的发现依赖于直观的界面和程序化访问点。

### 用户界面 (UI)

基于网络的 UI 是大多数用户主要的发现工具。重要的功能包括：

- **搜索：** 搜索功能必不可少。这应支持按名称、描述、标签、所有者以及其他元数据字段进行搜索。高级实现可能包含语义搜索功能，以根据相似性而非仅仅关键词匹配来查找特征。
- **浏览：** 允许用户按组、标签、所有者或其他相关方面浏览特征。分层视图有助于浏览大型特征集。
- **筛选和排序：** 使用户能够根据状态、数据类型、可用性、新鲜度或质量指标优化搜索结果。
- **详细视图：** 为每个特征提供一个专用页面，以清晰、有组织的方式显示其所有相关元数据。该页面通常包括来源图、分布图和使用信息。

考虑设计 UI 以满足不同角色的需求。数据科学家可能优先关注描述、分布和来源，而机器学习 (machine learning)工程师可能更关注更新频率、转换代码链接和可用性等操作细节。

> 用户、特征目录与其他特征存储组件之间的高层交互。该目录聚合了来自各种来源的元数据，以提供统一的发现界面。

### 程序化访问 (API)

虽然 UI 对于浏览很有用，但通过 API（例如 REST 或 gRPC）和相关客户端库（例如 Python SDK）进行的程序化访问对于自动化和集成非常重要：

- **与笔记本集成：** 数据科学家可以直接在开发环境中查找和检查特征。
- **CI/CD 集成：** 自动化管道可以查询目录以获取模型训练的特征列表或在部署前验证特征是否存在。
- **自定义工具：** 团队可以在目录 API 之上构建专门的工具或仪表板。

典型的 API 交互可能涉及查找符合特定条件的特征：

```python
# 使用 Python 客户端的示例
from feature_store_client import CatalogClient

client = CatalogClient(api_endpoint="http://feature-catalog.internal:8080")

# 查找风险团队拥有的与欺诈相关的生产就绪特征
features = client.search_features(
    query="transaction count",
    tags=["fraud"],
    owner_team="risk_analytics",
    status="PRODUCTION",
    min_quality_score=0.95
)

for feature in features:
    print(f"找到: {feature.name} (所有者: {feature.owner_team})")
    # 访问详细元数据
    print(f"  描述: {feature.description}")
    print(f"  数据类型: {feature.data_type}")
    print(f"  最后更新: {feature.last_updated_at}")
```

## 实现考量

构建或集成特征目录涉及几个技术决策：

- **元数据聚合：** 目录需要管道来收集和整合来自特征注册表、监控系统、来源跟踪器以及可能来自存储层本身的元数据。这通常涉及异步作业。
- **搜索后端：** 为了高效的自由文本搜索和筛选，通常使用专门的搜索引擎（如 Elasticsearch、OpenSearch 或 Apache Solr）来索引聚合的元数据。
- **构建与集成：**
  - **构建：** 开发一个根据特定组织需求定制的 UI 和 API。这提供了最大的灵活性，但需要大量的开发和维护工作。
  - **集成：** 使用开源数据发现工具，如 Amundsen、DataHub 或 OpenMetadata。这些通常为特征存储实体提供可扩展模型和预构建的用户界面，但需要集成工作，并且可能涉及适应其特定的元数据模型。
  - **托管服务：** 云提供商的特征存储（SageMaker、Vertex AI、Azure ML）通常包含内置的注册表和发现功能，尽管可能不如专用工具可定制。

选择取决于特征存储的规模、所需元数据的复杂性、现有基础设施和可用的工程资源。

## 连接发现与治理

特征发现与治理（本章前面已介绍）紧密关联：

- **可见性和访问：** 目录可以强制执行访问控制，根据角色或项目隶属关系仅向用户显示他们被允许查看的特征。
- **透明度：** 它明确了所有权，促进了沟通和问责。
- **生命周期管理：** 显示特征状态（例如，`DEPRECATED` 已弃用）可引导用户避免使用过时的特征。
- **审计：** 目录提供了一个中心点，以便理解存在哪些特征、谁拥有它们以及如何定义它们，支持合规性工作。

通过使特征易于理解和访问，一个良好实施的发现和编目系统增强协作，推广最佳实践，并确保高级特征存储真正加速组织内的机器学习 (machine learning)开发和部署。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  全面介绍了特征存储，包括元数据、发现和治理在更广泛的机器学习系统设计中的重要性。
- [Amundsen: A Data Discovery Platform for the AI Era](http://www.vldb.org/pvldb/vol12/p1679-song.pdf) — V. Song, H. Hacigundogdu, S. Kulkarni, B. Hu, V. Kunjir, X. Deng, K. C. Lee, R. Lin (2019)
  Journal: Proceedings of the VLDB Endowment (PVLDB); Publisher: VLDB Endowment; Volume: 12; Pages: 1679-1692; DOI: [10.14778/3352063.3352077](https://doi.org/10.14778/3352063.3352077)
  介绍了 Lyft 的开源数据发现平台，详细阐述了其架构以及如何促进元数据管理和特征发现能力。
- [DataHub: A Metadata Platform for the Modern Data Stack](http://www.vldb.org/pvldb/vol13/p3105-deng.pdf) — X. Deng, C. P. Binnig, S. Gupta, K. C. Lee, Z. Luo, S. Kulkarni, V. Song, V. Kunjir (2020)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 13; Pages: 3105-3118; DOI: [10.14778/3415494.3415555](https://doi.org/10.14778/3415494.3415555)
  展示了 LinkedIn 的开源元数据平台，提供了构建可扩展的数据和特征编目、血缘和治理系统的见解。
- [Vertex AI Feature Store documentation](https://cloud.google.com/vertex-ai/docs/featurestore) — Google Cloud (2024)
  主要云提供商特征存储的官方文档，展示了托管服务中特征管理（包括发现）的实践方面。
