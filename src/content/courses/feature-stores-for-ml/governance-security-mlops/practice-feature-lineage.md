---
course: "feature-stores-for-ml"
chapter: "governance-security-mlops"
lesson: "practice-feature-lineage"
sourceId: 3437
sourceUrl: "https://apxml.com/zh/courses/feature-stores-for-ml/chapter-5-governance-security-mlops/practice-feature-lineage"
title: "实践：设置特征血缘"
description: "在特征存储环境中，为一组特征实现基本的血缘追踪。"
order: 8
plots: []
sourceHash: "cded53c70f703493fde5340c496e947b3d275a6c8d76e5ba1e9e2311364ae01c"
sourceCorrections: []
---

本节提供了一个动手练习，用于为一小组特征实现基本的血缘追踪。基于本章前面讨论的关于治理和可复现性的理念，我们将手动为简化的特征工程过程添加追踪，以捕获元数据，从而追踪特征的来源和转换。目标是理解血缘追踪的基本机制，然后可以在生产环境中利用更复杂的工具和框架进行扩展和自动化。

我们将模拟一个场景，其中我们根据原始曝光和点击日志以及用户画像数据计算用户的7日点击率 (CTR)。

### 场景设置

假设我们有以下数据源：

1. `impression_logs`：包含 `user_id`、`timestamp`、`ad_id` 的记录。
2. `click_logs`：包含 `user_id`、`timestamp`、`ad_id` 的记录。
3. `user_profiles`：包含 `user_id`、`country`、`registration_date` 的静态数据。

我们的目标是创建特征 `user_ctr_7d`。

### 前提条件

- 基本的 Python 编程环境。
- 熟悉 Python 中的字典操作。
- （可选）如果您想在本地渲染可视化图，需要 Graphviz 库（`pip install graphviz`）。

### 步骤 1：定义数据源和初始血缘

首先，让我们表示我们的原始数据源并关联一些初始血缘元数据。在实际系统中，这可能指向特定的数据库表、文件路径或流主题。

```python
# 血缘元数据存储（本例中为一个简单字典）
lineage_registry = {}

# 表示原始数据源
impression_log_source = {
    "id": "raw_impressions_kafka_topic",
    "type": "kafka_topic",
    "schema": ["user_id", "timestamp", "ad_id"],
    "description": "原始广告曝光事件流",
    "owner": "ad_platform_team"
}
lineage_registry[impression_log_source["id"]] = {
    "type": "DataSource",
    "details": impression_log_source
}

click_log_source = {
    "id": "raw_clicks_kafka_topic",
    "type": "kafka_topic",
    "schema": ["user_id", "timestamp", "ad_id"],
    "description": "原始广告点击事件流",
    "owner": "ad_platform_team"
}
lineage_registry[click_log_source["id"]] = {
    "type": "DataSource",
    "details": click_log_source
}

user_profile_source = {
    "id": "user_profiles_postgres_table",
    "type": "database_table",
    "schema": ["user_id", "country", "registration_date"],
    "description": "用户人口统计信息",
    "owner": "user_service_team"
}
lineage_registry[user_profile_source["id"]] = {
    "type": "DataSource",
    "details": user_profile_source
}

print("初始血缘注册表：")
import json
print(json.dumps(lineage_registry, indent=2))
```

这建立了我们血缘图的根节点。我们存储了每个数据源的基本元数据。

### 步骤 2：实现并追踪特征转换

现在，让我们模拟特征工程步骤并随之记录血缘信息。

**转换 1：计算每个用户的 7 日曝光量和点击量**

设想一个处理日志的函数或任务。我们需要捕获它读取了哪些源以及生成了哪些特征。

```python
# 模拟处理函数（伪代码）
def calculate_user_counts_7d(impressions_data, clicks_data):
    # ... 使用 Spark、Flink 或 Pandas 的复杂逻辑 ...
    # 输出：user_id, impressions_7d, clicks_7d
    user_counts = {
        "user1": {"impressions_7d": 100, "clicks_7d": 5},
        "user2": {"impressions_7d": 200, "clicks_7d": 8},
        # ... 其他用户
    }
    return user_counts

# 定义输出特征及其血缘
feature_impressions_7d = {
    "id": "user_impressions_7d",
    "description": "过去 7 天每个用户的总曝光量",
    "value_type": "integer",
    "entity": "user" # 关联实体
}

feature_clicks_7d = {
    "id": "user_clicks_7d",
    "description": "过去 7 天每个用户的总点击量",
    "value_type": "integer",
    "entity": "user"
}

# 记录这些特征的血缘
transformation_id_counts = "calculate_user_counts_7d_job_v1.2"
lineage_registry[feature_impressions_7d["id"]] = {
    "type": "Feature",
    "details": feature_impressions_7d,
    "generated_by": {
        "transformation_id": transformation_id_counts,
        "inputs": [impression_log_source["id"]] # 主要来源于曝光数据
    }
}
lineage_registry[feature_clicks_7d["id"]] = {
    "type": "Feature",
    "details": feature_clicks_7d,
    "generated_by": {
        "transformation_id": transformation_id_counts,
        "inputs": [click_log_source["id"]] # 主要来源于点击数据
    }
}

# 添加关于转换本身的详细信息
lineage_registry[transformation_id_counts] = {
    "type": "Transformation",
    "details": {
        "description": "聚合用户在 7 天窗口内的曝光量和点击量。",
        "code_reference": "git://repo/jobs/user_aggregation.py#tag=v1.2",
        "compute_framework": "Spark"
    }
}

print("\n转换 1 后的血缘注册表：")
print(json.dumps(lineage_registry, indent=2))
```

我们添加了两个新特征 (`user_impressions_7d`、`user_clicks_7d`) 并将它们链接回特定的转换 (`calculate_user_counts_7d_job_v1.2`) 和它们各自的原始输入源。我们还添加了关于转换本身的元数据。

**转换 2：计算 7 日 CTR**

此步骤使用前一个转换的输出并计算最终的 CTR 特征。

```python
# 模拟 CTR 计算函数（伪代码）
def calculate_ctr(user_counts_data):
    # ... 计算点击/曝光的逻辑，处理除零情况 ...
    user_ctr = {
        "user1": {"user_ctr_7d": 0.05}, # 5 / 100
        "user2": {"user_ctr_7d": 0.04}, # 8 / 200
        # ... 其他用户
    }
    return user_ctr

# 定义最终的 CTR 特征
feature_ctr_7d = {
    "id": "user_ctr_7d",
    "description": "用户过去 7 天的点击率",
    "value_type": "float",
    "entity": "user"
}

# 记录 CTR 特征的血缘
transformation_id_ctr = "calculate_user_ctr_job_v1.0"
lineage_registry[feature_ctr_7d["id"]] = {
    "type": "Feature",
    "details": feature_ctr_7d,
    "generated_by": {
        "transformation_id": transformation_id_ctr,
        "inputs": [feature_impressions_7d["id"], feature_clicks_7d["id"]] # 输入特征
    }
}

# 添加关于 CTR 转换的详细信息
lineage_registry[transformation_id_ctr] = {
    "type": "Transformation",
    "details": {
        "description": "根据 7 日曝光和点击量计算 CTR。",
        "code_reference": "git://repo/jobs/ctr_calculation.py#tag=v1.0",
        "compute_framework": "Python/Pandas"
    }
}

print("\n最终血缘注册表：")
print(json.dumps(lineage_registry, indent=2))
```

现在，`user_ctr_7d` 的血缘正确指向其转换，而该转换又依赖于中间特征 `user_impressions_7d` 和 `user_clicks_7d`。

### 步骤 3：血缘可视化

我们可以使用收集到的血缘信息来可视化依赖关系。让我们生成一个 Graphviz DOT 表示。

```python
def generate_lineage_graph(registry):
    dot_lines = ['digraph G {', 'rankdir=LR;']
    node_shapes = {
        "DataSource": "folder",
        "Feature": "ellipse",
        "Transformation": "box"
    }
    node_colors = {
         "DataSource": "#a5d8ff", # blue
         "Feature": "#b2f2bb",    # green
         "Transformation": "#ffd8a8" # orange
    }

    nodes = set()

    for item_id, item_data in registry.items():
        node_type = item_data.get("type")
        if not node_type: continue

        shape = node_shapes.get(node_type, "plaintext")
        color = node_colors.get(node_type, "#e9ecef") # default gray
        label = item_id.replace("_", "\\n") # Basic formatting

        nodes.add(f'"{item_id}" [label="{label}", shape={shape}, style=filled, fillcolor="{color}"];')

        if node_type == "Feature":
            generator = item_data.get("generated_by")
            if generator:
                transformation_id = generator.get("transformation_id")
                if transformation_id in registry:
                     dot_lines.append(f'"{transformation_id}" -> "{item_id}";')
                input_ids = generator.get("inputs", [])
                for input_id in input_ids:
                    if input_id in registry:
                         dot_lines.append(f'"{input_id}" -> "{transformation_id}";')

    dot_lines.extend(list(nodes))
    dot_lines.append('}')
    return "\n".join(dot_lines)

dot_graph = generate_lineage_graph(lineage_registry)
print("\nGraphviz DOT 格式：")
print(dot_graph)

# 保存到文件以便用 Graphviz 渲染（可选）
# with open("feature_lineage.dot", "w") as f:
#     f.write(dot_graph)
# 您可以使用以下命令渲染此图：dot -Tpng feature_lineage.dot -o feature_lineage.png
```

这是生成的 Graphviz 定义：

> 生成的图表展示了从原始数据源（文件夹）经过转换步骤（方框）到最终特征（椭圆形）的流程。请注意，`user_profiles_postgres_table` 当前是断开的，因为它未用于这些特定转换中。

### 讨论和后续步骤

此练习演示了血缘追踪的核心理念：将元数据与数据源、特征以及连接它们的转换关联起来。

- **手动操作：** 像这样手动捕获血缘只适用于非常简单的流水线。实际操作中，这需要自动化。特征工程框架或库通常提供装饰器或钩子（例如，`log_lineage(inputs=[...], outputs=[...])`），以便在流水线执行期间自动捕获此信息。
- **粒度：** 我们在作业/特征层面追踪血缘。更细粒度的血缘可能追踪特定列甚至数据分区，这需要更复杂的工具。
- **集成：** `lineage_registry` 字典是一个最小表示。系统使用专门的元数据数据库或与 OpenLineage、Amundsen 或 DataHub 等平台集成，这些平台提供 API 以存储、查询和可视化血缘以及其他元数据（模式、所有权、描述）。
- **使用：** 主要价值来自于使用这些血缘信息。它有助于调试（追溯有问题的特征值）、影响分析（理解哪些模型或特征受到上游变化的影响）、合规性报告和可复现性。

完成此练习后，您已实现了特征血缘的基本组成部分。在实际系统中，下一步是将此捕获机制集成到您的特征工程代码中，并借助专门的元数据平台来有效地管理和呈现血缘信息。

## 参考资料

- [OpenLineage Specification](https://openlineage.io/) — LF AI & Data Foundation (2023)
  Publisher: LF AI & Data Foundation
  为数据和机器学习管道中的血缘元数据收集和交换定义了一个开放标准，为自动化血缘追踪提供了框架。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供了构建机器学习系统的实践见解，包括特征存储、数据管理以及可复现性和治理重要性的讨论，其中血缘发挥作用。
- [Provenance for Data Science: Challenges and Opportunities](https://dl.acm.org/doi/10.1145/3095311.3095325) — Lucian Smith, Juliana Freire, Ben Liblit, Ben H. Lee, and Yannis Velegrakis (2017)
  Journal: SIGMOD Record; Publisher: ACM; Volume: 46; Pages: 37-46; DOI: [10.1145/3095311.3095325](https://doi.org/10.1145/3095311.3095325)
  一篇基础的学术论文，探讨了数据科学工作流中数据溯源的挑战和益处，直接支持了血缘追踪在可复现性和信任方面的重要作用。
