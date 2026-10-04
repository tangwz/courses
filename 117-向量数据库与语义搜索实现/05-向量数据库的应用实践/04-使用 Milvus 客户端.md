# 使用 Milvus 客户端

来源：[原文](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice/working-with-milvus)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Milvus 是一个广受欢迎的开源向量 (vector)数据库，专门为大规模向量数据集上的高效相似性搜索和分析而设计。它提供了一个分布式、云原生架构，旨在提供高性能和弹性，使其成为生产级语义搜索系统的有力选择。使用 Milvus 的 Python 客户端库 `pymilvus` 进行交互的方法，以及构建搜索应用所需的基本操作，在此介绍。

### Milvus 核心概念

在查看代码之前，让我们先熟悉一些 Milvus 术语：

- **集合**: 类似于传统关系数据库中的表。它是向量 (vector)数据及相关元数据的主要存储容器。
- **Schema（模式）**: 定义集合的结构，指定其包含的字段（列）。每个集合必须至少有一个主键字段和一个向量字段。
- **字段**: 表示集合模式中的一列。重要的字段类型包括：
  - `DataType.INT64` 或 `DataType.VARCHAR`: 常用于主键。
  - `DataType.FLOAT_VECTOR`: 存储向量嵌入 (embedding)。必须指定向量的 `dim`（维度）。
  - 标量类型（例如：`DataType.VARCHAR`、`DataType.INT32`、`DataType.BOOL`、`DataType.FLOAT`）：用于存储与每个向量相关的元数据。
- **实体**: 表示集合中的单个记录或行，包含所有已定义字段的数据，包括向量及其元数据。
- **分区**: 一种可选方法，用于根据特定字段将集合数据划分为更小、更易管理的部分。搜索可以限定在特定分区内以提高效率。
- **索引**: 建立在向量字段上的数据结构，用于加速相似性搜索。Milvus 支持第3章中已有论述的多种 ANN 索引类型，例如 HNSW、IVF\_FLAT 等。您在创建索引时指定索引类型和相关参数 (parameter)（例如 HNSW 的 `M` 和 `efConstruction`）。
- **度量类型**: 用于衡量向量之间相似性的距离度量方式（例如：欧氏距离的 `L2`，内积的 `IP`）。通常在创建索引和执行搜索时指定此项。

### 连接 Milvus

首先，请确保您有一个 Milvus 实例正在运行（例如，使用 Docker 进行本地开发）并且 `pymilvus` 库已安装（`pip install pymilvus`）。连接很简单：

```python
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# 连接到 Milvus
# 假设 Milvus 运行在 localhost:19530
connections.connect(alias="default", host="localhost", port="19530")

print("已连接到 Milvus。")

# 检查连接是否成功
print(f"可用集合: {utility.list_collections()}")
```

`alias` 参数 (parameter)允许您在需要时管理多个连接；“default” 是常用值。

### 定义 Schema 并创建集合

让我们定义一个 Schema，用于存储技术文章信息，包括标题（元数据）、出版年份（元数据）和内容嵌入 (embedding)（向量 (vector)）。

```python
# 定义 Schema 字段
article_id = FieldSchema(
    name="article_id",
    dtype=DataType.INT64,
    is_primary=True,
    auto_id=True # 让 Milvus 生成唯一 ID
)
title = FieldSchema(
    name="title",
    dtype=DataType.VARCHAR,
    max_length=256 # VARCHAR 字段的最大长度
)
publish_year = FieldSchema(
    name="publish_year",
    dtype=DataType.INT32
)
embedding = FieldSchema(
    name="embedding",
    dtype=DataType.FLOAT_VECTOR,
    dim=768 # 示例维度，与您的嵌入模型匹配
)

# 定义集合 Schema
schema = CollectionSchema(
    fields=[article_id, title, publish_year, embedding],
    description="Technical article collection",
    enable_dynamic_field=False # 设置为 True 允许添加不在 Schema 中的字段
)

# 创建集合
collection_name = "tech_articles"

# 如果集合已存在则删除（用于演示）
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"已删除现有集合: {collection_name}")

collection = Collection(name=collection_name, schema=schema)

print(f"集合 '{collection.name}' 创建成功。")
```

这里，`auto_id=True` 通过让 Milvus 处理主键生成来简化 ID 管理。我们明确设置了向量维度（`dim=768`），这*必须*与您使用的嵌入模型的输出维度相匹配。

### 创建索引

为了高效搜索，特别是在大型数据集上，在向量 (vector)字段上创建索引是必不可少的。这通常应在创建集合*之后*但在插入大量数据*之前*完成，尽管 Milvus 也支持在插入后进行索引。

```python
# 定义 'embedding' 字段的索引参数
# 使用 HNSW 索引类型，L2 距离度量
index_params = {
    "metric_type": "L2",        # 许多嵌入模型的常见选择
    "index_type": "HNSW",       # 分层可导航小世界
    "params": {
        "M": 16,                # 每个节点层的邻居数量
        "efConstruction": 200   # 索引构建过程中动态列表的大小
    }
}

# 创建索引
collection.create_index(
    field_name="embedding",
    index_params=index_params
)

print(f"已在字段 'embedding' 上成功创建索引。")

# 等待索引构建完成（对一致性很重要）
utility.wait_for_index_building_complete(collection_name)
print("索引构建完成。")
```

选择合适的 `index_type` 并调整 `M` 和 `efConstruction` 等 `params` 需要在索引时间、搜索速度和召回率之间进行权衡，这在第3章中已有论述。`L2`（欧氏距离）是一种常用度量，但对于归一化 (normalization)嵌入 (embedding)，通常首选 `IP`（内积），因为它等同于余弦相似度。

### 将数据加载到内存

在 Milvus 中搜索集合之前，您通常需要将其（或特定分区）加载到内存中。这可以预热数据，以实现更快的查询响应。

```python
# 将集合加载到内存中进行搜索
collection.load()
print(f"集合 '{collection.name}' 已加载到内存中。")
```

### 插入数据

现在，让我们插入一些示例数据。数据应结构化为列表的列表或字典的列表，其中顺序或键与 Schema 定义匹配（不包括 `auto_id` 主键）。

```python
# 示例数据（确保向量维度与 Schema 匹配，例如 768）
# 替换为您模型的实际嵌入
data_to_insert = [
    {"title": "Intro to Vector Databases", "publish_year": 2023, "embedding": [0.1] * 768},
    {"title": "Understanding HNSW Indexes", "publish_year": 2022, "embedding": [0.2] * 768},
    {"title": "Semantic Search Pipelines", "publish_year": 2023, "embedding": [0.3] * 768},
    {"title": "Deep Learning Fundamentals", "publish_year": 2021, "embedding": [0.4] * 768}
]

# 插入数据
insert_result = collection.insert(data_to_insert)

print(f"数据已插入。主键: {insert_result.primary_keys}")

# Milvus 操作通常是异步的。Flush 确保数据持久化。
collection.flush()
print(f"刷新后实体数量: {collection.num_entities}")
```

`collection.flush()` 很重要，它能确保插入的数据段被密封并变得可搜索。Milvus 会自动处理 `article_id` 的生成，因为我们设置了 `auto_id=True`。

### 执行相似性搜索

核心操作是搜索与给定查询向量 (vector)相似的向量。您提供查询向量、结果数量（`limit`）和搜索参数 (parameter)。

```python
# 生成查询向量（例如，通过嵌入搜索查询）
# 替换为实际的查询嵌入
query_vector = [[0.15] * 768] # 示例查询向量

# 定义搜索参数
search_params = {
    "metric_type": "L2",        # 许多嵌入模型的常见选择
    "params": {
        "ef": 128  # 搜索范围，值越高意味着召回率越高但速度越慢
    }
}

# 执行搜索
results = collection.search(
    data=query_vector,                 # 查询向量列表
    anns_field="embedding",            # 要搜索的字段
    param=search_params,               # 索引和搜索参数
    limit=3,                           # 返回结果的数量
    output_fields=["title", "publish_year"] # 指定要检索的元数据字段
)

# 处理结果
print("\n搜索结果（前3名）:")
for hit in results[0]: # results[0] 对应于第一个查询向量
    print(f"  ID: {hit.id}, 距离: {hit.distance:.4f}, 标题: {hit.entity.get('title')}, 年份: {hit.entity.get('publish_year')}")
```

HNSW 的 `search_params` 中的 `ef` 参数控制查询时搜索的动态列表大小。更高的值通常会以延迟为代价提高准确性（召回率）。`output_fields` 允许您检索特定元数据以及匹配的向量 ID 和距离。

### 带元数据过滤的搜索

Milvus 支持将向量 (vector)相似性搜索与基于标量元数据字段的布尔表达式过滤结合起来。

```python
# 使用过滤表达式执行搜索
# 查找与 query_vector 相似但发表于 2023 年的文章
filtered_results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_params,
    limit=3,
    expr="publish_year == 2023", # 用于过滤的布尔表达式
    output_fields=["title", "publish_year"]
)

print("\n过滤后的搜索结果（2023 年发布，前3名）:")
if not filtered_results[0]:
    print("  未找到符合过滤条件的匹配结果。")
else:
    for hit in filtered_results[0]:
        print(f"  ID: {hit.id}, 距离: {hit.distance:.4f}, 标题: {hit.entity.get('title')}, 年份: {hit.entity.get('publish_year')}")
```

`expr` 参数 (parameter)使用类似 SQL 的语法来定义元数据字段上的过滤条件。这种预过滤（或后过滤，取决于 Milvus 采用的策略）效率很高。

### 其他操作

- **获取实体**: 通过主键检索完整的实体数据（包括向量 (vector)）：`collection.get(ids=[...])`。
- **查询**: 根据布尔表达式检索实体，而不执行向量搜索：`collection.query(expr="publish_year > 2022", output_fields=["title"])`。
- **删除实体**: 通过主键删除实体：`collection.delete(expr="article_id in [...]")`。
- **分区**: 您可以创建分区（`collection.create_partition()`），并在插入和搜索时指定分区名称（`partition_name=...`）以管理数据子集。

### 释放和删除集合

当一个集合不再需要进行活跃搜索时，将其从内存中释放。要永久删除它，请将其删除。

```python
# 从内存中释放集合
collection.release()
print(f"\n集合 '{collection.name}' 已从内存中释放。")

# 永久删除集合
# utility.drop_collection(collection_name)
# print(f"集合 '{collection.name}' 已删除。")

# 断开与 Milvus 的连接
connections.disconnect(alias="default")
print("已断开与 Milvus 的连接。")
```

使用 Milvus 客户端需要理解其特定概念，例如集合、Schema，以及索引和加载的重要性。它的 Python 客户端提供了一个全面的接口，用于定义数据结构、摄取向量 (vector)和元数据，并执行带过滤条件的相似性搜索，使其成为构建实用语义搜索应用的合适选择。请记住，性能通常需要根据您的具体数据和应用需求仔细调整索引和搜索参数 (parameter)。

## 参考资料

- [Milvus Documentation](https://milvus.io/docs) — Zilliz (2024)
  Milvus官方文档，涵盖客户端使用、API和核心概念。
- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://ieeexplore.ieee.org/document/8447814) — Yu. A. Malkov, D. A. Yashunin (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 42; Pages: 824-836; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  介绍HNSW算法的奠基性论文，与理解本节中提到的索引类型和参数相关。
- [Vector Databases: A Guide to the New Age of AI Applications](https://books.google.com/books/about/Vector_Databases.html?id=z1X7EAAAQBAJ) — Connor, Josh and Zilliz (2023)
  Publisher: O'Reilly Media
  全面概述了向量数据库的架构、用例和基本原理。
- [Building Intelligent Search Applications](https://www.oreilly.com/library/view/building-intelligent-search/9781492077536/) — Turnbull, Doug and Berryman, John (2021)
  Publisher: O'Reilly Media
  涵盖了构建搜索应用程序的原理和实践，包括向量嵌入和相似性搜索的作用。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20Weaviate%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20ChromaDB%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
