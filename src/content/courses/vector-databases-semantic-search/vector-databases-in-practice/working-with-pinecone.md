---
course: "vector-databases-semantic-search"
chapter: "vector-databases-in-practice"
lesson: "working-with-pinecone"
sourceId: 3585
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice/working-with-pinecone"
title: "使用 Pinecone 客户端"
description: "使用 Pinecone Python 客户端进行连接、索引和查询。"
order: 2
plots: []
sourceHash: "15f54c37689e7c00616aed91931cb5562139557c0d63f4bd36a8086c2f76c41b"
sourceCorrections: []
---

Pinecone 是一种流行的托管式向量 (vector)数据库服务，专门为高性能相似性搜索应用设计。它简化了大部分底层基础设施管理，让开发者能够专注于将向量搜索功能集成到自己的系统中。与 Pinecone 的交互主要是通过其客户端库进行，Python 客户端是机器学习 (machine learning)和数据科学工作流程的常用选择。

要开始使用 Pinecone，你需要 API 密钥和环境信息，这些可以在注册 Pinecone 账户后获取。

### 安装与初始化

首先，安装官方的 Pinecone Python 客户端库：

```bash
pip install pinecone-client
```

安装完成后，你可以初始化与 Pinecone 项目的连接。最佳实践是安全地存储你的 API 密钥和环境信息，例如作为环境变量，而不是直接将它们硬编码到脚本中。

```python
import os
from pinecone import Pinecone, ServerlessSpec, PodSpec

# 初始化与 Pinecone 的连接
# 假设 PINECONE_API_KEY 和 PINECONE_ENVIRONMENT 已设置为环境变量
api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENVIRONMENT") # 例如，'gcp-starter' 或像 'us-west1-gcp' 这样的区域

if not api_key or not environment:
    raise ValueError("请设置 PINECONE_API_KEY 和 PINECONE_ENVIRONMENT 环境变量。")

# 初始化 Pinecone 客户端
pc = Pinecone(api_key=api_key)

print("Pinecone 客户端已初始化。")
# 你可以验证连接，例如通过列出索引（见下文）
```

### 管理索引

在 Pinecone 中，你的向量 (vector)存储在一个“索引”中。索引的特点是其名称、将要存储向量的维度，以及用于相似性计算的距离度量（例如，`cosine`、`euclidean`、`dotproduct`）。创建索引时，你还需要指定基础设施类型（pods 或 serverless）。

#### 创建索引

我们来创建一个名为 `semantic-search-demo` 的新索引，用于存储 768 维向量，并使用余弦相似性度量。本例中我们将使用一个基本的无服务器配置。

```python
import time

index_name = "semantic-search-demo"
vector_dimension = 768
similarity_metric = "cosine"

# 检查索引是否已存在
if index_name not in pc.list_indexes().names:
    print(f"正在创建索引 '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=vector_dimension,
        metric=similarity_metric,
        spec=ServerlessSpec(
            cloud='aws',  # 或 'gcp', 'azure'
            region='us-west-2' # 选择一个受支持的区域
        )
        # 或者对于基于 Pod 的索引：
        # spec=PodSpec(
        #     environment=environment, # 你的项目环境
        #     pod_type="p1.x1",      # 示例 Pod 类型
        #     pods=1
        # )
    )
    # 等待索引准备就绪
    while not pc.describe_index(index_name).status['ready']:
        print("正在等待索引初始化...")
        time.sleep(5)
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")

# 连接到索引
index = pc.Index(index_name)
print(index.describe_index_stats())
```

#### 列出和删除索引

你可以方便地管理你的索引：

```python
# 列出项目中的所有索引
index_list = pc.list_indexes()
print("可用索引:", index_list.names)

# 删除一个索引（请谨慎使用！）
# index_to_delete = "some-old-index"
# if index_to_delete in index_list.names:
#     print(f"正在删除索引 '{index_to_delete}'...")
#     pc.delete_index(index_to_delete)
#     print("已启动删除。这可能需要几分钟。")
```

### 向量 (vector)操作

使用索引对象 (`index = pc.Index(index_name)`)，你可以在该索引中对向量执行操作。

#### 插入/更新数据 (创建/更新)

`upsert` 操作在索引中添加新向量或更新现有向量（通过其 `id` 识别）。你通常提供一个元组列表，其中每个元组包含 `(id, vector_values, metadata)`。元数据是一个可选字典，包含与向量关联的附加信息，可用于过滤。

为了效率，强烈建议批量插入/更新数据，而不是一次处理一个向量。Pinecone 对每个 `upsert` 请求的大小（例如，向量数量或总请求大小）有限制。

```python
import random

# 示例数据（请替换为你的实际嵌入）
vectors_to_upsert = []
num_vectors = 100
for i in range(num_vectors):
    vector_id = f"vec_{i}"
    vector_values = [random.random() for _ in range(vector_dimension)] # 768 维
    metadata = {
        "genre": random.choice(["fiction", "non-fiction"]),
        "year": 2020 + random.randint(0, 3),
        "source_doc": f"doc_{i // 10}" # 按源文档对向量进行分组
    }
    vectors_to_upsert.append((vector_id, vector_values, metadata))

# 批量插入/更新（示例批量大小 50）
batch_size = 50
print(f"正在以 {batch_size} 的批量大小插入/更新 {len(vectors_to_upsert)} 个向量...")
for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    upsert_response = index.upsert(vectors=batch)
    print(f"已插入/更新批量 {i // batch_size + 1}，响应：{upsert_response}")

print("插入/更新操作完成。")
# 验证索引大小
print(index.describe_index_stats())
```

#### 查询 (相似性搜索)

主要操作是 `query`，它根据索引的距离度量，查找索引中与给定 `vector`（查询向量）最相似的 `top_k` 个向量。

```python
# 生成一个示例查询向量（应与索引维度匹配）
query_vector = [random.random() for _ in range(vector_dimension)]

# 执行相似性搜索
k = 5 # 要检索的最近邻居数量
print(f"正在查询最接近的 {k} 个邻居...")
query_response = index.query(
    vector=query_vector,
    top_k=k,
    include_metadata=True,  # 与 ID 和分数一起检索元数据
    include_values=False    # 可选地检索向量值（通常为 False 以提升性能）
)

print("查询结果：")
if query_response.matches:
    for match in query_response.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}, Metadata: {match.metadata}")
else:
    print("  未找到匹配项。")
```

返回的 `score` 取决于所使用的度量（例如，对于余弦相似性，分数越高越好；对于欧几里得距离，分数越低越好）。

#### 带元数据过滤的查询

你可以通过根据向量关联的元数据应用过滤器来细化搜索。Pinecone 支持各种过滤操作（等值、不等值、范围查询、集合成员关系）。

```python
# 查询与 query_vector 相似的向量，但仅限于 2022 年出版的“fiction”体裁的向量
print(f"\n正在查询 2022 年来自“fiction”体裁的 {k} 个最接近邻居...")

filter_criteria = {
    "genre": {"$eq": "fiction"},
    "year": {"$eq": 2022}
}

filtered_query_response = index.query(
    vector=query_vector,
    top_k=k,
    filter=filter_criteria,
    include_metadata=True
)

print("过滤后的查询结果：")
if filtered_query_response.matches:
    for match in filtered_query_response.matches:
        print(f"  ID: {match.id}, Score: {match.score:.4f}, Metadata: {match.metadata}")
else:
    print("  未找到符合过滤条件的匹配项。")
```

语义相似性搜索和结构化元数据过滤的这种组合是向量数据库的一个强大特点。

#### 获取和删除向量

你也可以使用其唯一 ID 检索或删除特定向量。

```python
# 按 ID 获取向量
ids_to_fetch = ["vec_10", "vec_25"]
print(f"\n正在按 ID 获取向量: {ids_to_fetch}")
fetch_response = index.fetch(ids=ids_to_fetch)
# print(fetch_response) # 取消注释以查看完整的响应结构
for vec_id, vec_data in fetch_response.vectors.items():
     print(f"  Fetched ID: {vec_id}, Metadata: {vec_data.metadata}") # 向量值也可在 vec_data.values 中获取

# 按 ID 删除向量
ids_to_delete = ["vec_5", "vec_15"]
print(f"\n正在按 ID 删除向量: {ids_to_delete}")
delete_response = index.delete(ids=ids_to_delete)
print(f"删除响应: {delete_response}") # 成功时响应通常为空 {}

# 通过尝试再次获取或检查统计信息来验证删除
time.sleep(2) # 删除可能需要一点时间才能反映在统计信息中
print(index.describe_index_stats())
try:
    fetch_deleted = index.fetch(ids=ids_to_delete)
    print("已删除的向量已找到（意外）:", fetch_deleted.vectors.keys())
except Exception as e: # Pinecone 可能会以不同方式处理对不存在 ID 的获取
    print(f"尝试获取已删除 ID ({ids_to_delete}) 很可能导致空状态或错误状态，符合预期。")
```

使用 Pinecone 客户端涉及以下主要步骤：初始化连接、选择或创建索引，然后使用 `upsert`、`query`、`fetch` 和 `delete` 等方法来管理和搜索你的向量数据。作为一个托管服务，Pinecone 简化了部署和扩展，当你倾向于不自行管理数据库基础设施时，它是一个有力的选择。本章稍后的动手练习将指导你将此客户端集成到一个更完整的应用程序中。

## 参考资料

- [Pinecone Python Client Documentation](https://www.pinecone.io/docs/references/client/python/) — Pinecone (2024)
  官方指南，介绍如何使用Pinecone Python库进行交互，涵盖安装、API参考、索引管理和向量操作。
- [Pinecone Documentation: Core Concepts](https://www.pinecone.io/docs/overview/concepts/) — Pinecone (2024)
  介绍Pinecone的基本概念，包括索引、向量、元数据和距离度量，为实际使用提供背景信息。
- [An Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) — Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (2008)
  Publisher: Cambridge University Press
  信息检索基础教材，涵盖向量空间模型和相似性度量等原则，对于理解语义搜索至关重要。
