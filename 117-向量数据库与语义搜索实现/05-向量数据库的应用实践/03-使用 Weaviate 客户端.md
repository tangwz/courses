# 使用 Weaviate 客户端

来源：[原文](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice/working-with-weaviate)

[返回章节目录](README.md) · [返回课程目录](../README.md)

Weaviate 在向量 (vector)数据库中提供了一种独特的方式，其模块化架构和类似 GraphQL 的 API 将向量搜索与结构化数据筛选结合在一起。它的灵活性允许您接入不同的向量索引算法和嵌入 (embedding)模型，甚至可以使用您自己的预计算向量。学习使用 Weaviate Python 客户端与 Weaviate 实例交互，执行主要操作，并构建搜索功能。

### 连接到 Weaviate

首先，请确保您已安装 Weaviate Python 客户端：

```bash
pip install weaviate-client
```

要与 Weaviate 交互，您需要建立与实例的连接。这可以是运行在 Docker 中的本地实例，也可以是 Weaviate 云服务 (WCS) 提供的托管实例。

```python
import weaviate
import os

# 连接到本地 Weaviate 实例的示例
# client = weaviate.Client("http://localhost:8080")

# 连接到 Weaviate 云服务 (WCS) 的示例
# 替换为您的 WCS URL 和 API 密钥
wcs_url = os.getenv("WEAVIATE_URL") # 或者替换为您的 URL 字符串
api_key = os.getenv("WEAVIATE_API_KEY") # 或者替换为您的 API 密钥字符串

client = weaviate.Client(
    url=wcs_url,
    auth_client_secret=weaviate.AuthApiKey(api_key=api_key),
    # 可选：如果使用 text2vec-openai 模块，请指定 OpenAI API 密钥
    # headers={
    #     "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    # }
)

# 检查连接
if client.is_ready():
    print("成功连接到 Weaviate！")
else:
    print("连接 Weaviate 失败。")
```

请记住，将占位符连接详细信息替换为您的实际 URL 和 API 密钥，为了安全起见，可以像上面所示那样使用环境变量。

### 模式定义

在添加数据之前，您需要定义一个模式。在 Weaviate 中，模式由“类”（类似于表或集合）组成，这些类包含“属性”（字段）。每个类定义都包括其对象应如何向量 (vector)化。

让我们定义一个简单的 `Document` 类，包含 `title` 和 `content` 属性。我们将配置它使用一个预构建的转换器模型（例如 Sentence-BERT，由 `text2vec-transformers` 模块提供，假设它已在您的 Weaviate 实例中启用），以便自动为 `content` 属性生成向量。

```python
# 检查模式是否存在，以避免重复运行时出错
client.schema.delete_all() # 谨慎使用！这会删除所有模式。
# 或者删除特定的类：
# if client.schema.exists("Document"):
#     client.schema.delete_class("Document")

document_class_schema = {
    "class": "Document",
    "description": "用于存储文本文档的类",
    "vectorizer": "text2vec-transformers", # 指定向量化模块
    "moduleConfig": {
        "text2vec-transformers": {
            "poolingStrategy": "masked_mean",
            "vectorizeClassName": False # 通常设置为 False
        }
    },
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "description": "文档的标题",
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "文档的主要内容",
            "moduleConfig": { # 属性特定的模块配置
                "text2vec-transformers": {
                    "skip": False, # 确保此属性被向量化
                    "vectorizePropertyName": False # 通常设置为 False
                }
            }
        },
        {
            "name": "word_count",
            "dataType": ["int"],
            "description": "内容的字数",
        }
    ]
}

# 在 Weaviate 中创建类
try:
    client.schema.create_class(document_class_schema)
    print("模式 'Document' 创建成功。")
except Exception as e:
    print(f"创建模式时出错: {e}")

# 您可以验证模式创建
# schema = client.schema.get("Document")
# print(schema)
```

在此，我们将 `text2vec-transformers` 指定为该类的向量化 (quantization)器。这意味着 Weaviate 将自动使用该模块中配置的转换器模型，为每个 `Document` 对象生成向量嵌入 (embedding)，主要依据配置的 `content` 属性。如果您想提供自己的预计算向量，通常会将类的 `vectorizer` 设置为 `none`，并在添加数据时包含一个 `vector` 属性。

### 数据摄取和管理 (CRUD)

定义好模式后，您可以添加、检索、更新和删除数据对象。

#### 添加数据 (创建)

添加数据涉及创建符合已定义类模式的对象。为 `moduleConfig` 中指定的属性（例如我们示例中的 `content`）提供数据，允许 Weaviate 自动生成向量 (vector)。

为了提高效率，尤其是在处理大型数据集时，请使用批处理。

```python
# 配置批处理
client.batch.configure(
    batch_size=100,       # 每批对象数量
    dynamic=True,        # 动态调整批处理大小
    timeout_retries=3,   # 超时重试次数
)

documents_to_add = [
    {"title": "向量数据库简介", "content": "向量数据库是专门用于存储和查询高维向量嵌入的系统。", "word_count": 16},
    {"title": "了解 ANN 搜索", "content": "近似最近邻 (ANN) 搜索算法牺牲部分准确性，以在高维度中获得大幅度的速度提升。", "word_count": 20},
    {"title": "Weaviate 客户端库", "content": "Weaviate Python 客户端提供模式管理、数据操作和查询的方法。", "word_count": 16},
    {"title": "语义搜索原理", "content": "语义搜索旨在理解用户查询背后的意图和上下文含义，不限于关键词匹配。", "word_count": 20}
]

print("正在使用批处理添加文档...")
with client.batch as batch:
    for i, doc_data in enumerate(documents_to_add):
        # 将对象添加到批处理
        batch.add_data_object(
            data_object=doc_data,
            class_name="Document",
            # 可选：指定 UUID，否则 Weaviate 会自动生成一个
            # uuid=generate_uuid5(doc_data['title']) # 示例：使用 weaviate.util 中的 generate_uuid5
        )
        if (i + 1) % 10 == 0: # 可选：打印进度
            print(f"已将 {i+1} 个文档添加到批处理...")

# 批处理在退出 'with' 块时会自动处理。
# 检查批处理导入期间可能出现的错误
if client.batch.failed_objects:
    print(f"未能导入 {len(client.batch.failed_objects)} 个对象。")
    # 检查 client.batch.failed_objects 以获取详细信息
else:
    print(f"成功添加 {len(documents_to_add)} 个文档。")
```

#### 检索数据 (读取)

您可以通过 Weaviate 分配的 UUID 或您提供的自定义 UUID 来获取对象。

```python
# 首先，获取一个对象的 UUID（例如，我们添加的第一个对象）
# 注意：这需要首先知道或获取 UUID。
response = client.query.get("Document", ["title"]).with_limit(1).do()
if response['data']['Get']['Document']:
    object_uuid = response['data']['Get']['Document'][0]['_additional']['id']
    print(f"正在获取 UUID 为: {object_uuid} 的对象")

    # 根据 UUID 获取对象
    fetched_object = client.data_object.get_by_id(
        object_uuid,
        class_name="Document"
    )
    print("\n已获取对象:")
    print(fetched_object)
else:
    print("未找到要获取的文档。")

# 获取多个对象（有上限），无需指定 UUID
all_objects = client.data_object.get(class_name="Document", limit=10)
# print("\n所有对象 (上限 10 个):")
# print(all_objects)
```

#### 更新数据

您可以使用对象的 UUID 更新其属性。

```python
# 假设我们从上一步获得了 object_uuid
if 'object_uuid' in locals():
    new_title = "Weaviate Python 客户端指南"
    print(f"\n正在使用新标题 '{new_title}' 更新对象 {object_uuid}")

    # 方法 1: update() - 替换所有指定的属性
    # client.data_object.update(
    #     uuid=object_uuid,
    #     class_name="Document",
    #     data_object={"title": new_title} # 只有 title 被更新，如果不小心，其他属性可能会丢失
    # )

    # 方法 2: merge() - 合并指定属性，保留其他属性不变 (更安全)
    client.data_object.update( # 注意：`merge` 已被弃用/更改，请使用 `update` 并指定属性
        uuid=object_uuid,
        class_name="Document",
        data_object={"title": new_title},
        # consistency_level=weaviate.ConsistencyLevel.ONE # 可选的一致性设置
    )

    # 验证更新
    updated_object = client.data_object.get_by_id(object_uuid, class_name="Document")
    print("\n已验证更新后的对象:")
    print(updated_object)
```

*重要提示*：使用 `update` 时，只有 `data_object` 字典中包含的属性会被修改。未包含的属性保持不变。以前的版本有单独的 `merge` 方法；请查看您所用版本的客户端文档，了解其具体行为。

#### 删除数据

对象使用其 UUID 删除。

```python
# 假设我们从前面的步骤获得了 object_uuid
if 'object_uuid' in locals():
    print(f"\n正在删除 UUID 为: {object_uuid} 的对象")
    try:
        client.data_object.delete(
            uuid=object_uuid,
            class_name="Document",
            # consistency_level=weaviate.ConsistencyLevel.ALL # 可选：等待所有节点
        )
        print(f"对象 {object_uuid} 删除成功。")

        # 验证删除
        deleted_object = client.data_object.get_by_id(object_uuid, class_name="Document")
        if deleted_object is None:
            print("验证：未找到对象（删除成功）。")
        else:
            print("验证错误：对象仍存在。")

    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 404:
             print(f"对象 {object_uuid} 已被删除或未找到。")
        else:
             print(f"删除对象时出错: {e}")
```

### 使用 Weaviate 查询数据

Weaviate 的查询功能强大，支持语义搜索、关键词搜索、筛选及其组合。Python 客户端提供了一个流畅的接口，在后台构建 GraphQL 查询。

#### 语义搜索 (nearText / nearVector)

Weaviate 中语义搜索的核心依赖于 `nearText` 或 `nearVector`。

- `nearText`: 提供原始文本。Weaviate 使用配置的类向量 (vector)化器（例如 `text2vec-transformers`）嵌入 (embedding)查询文本并查找相似对象。
- `nearVector`: 提供显式查询向量。如果您在 Weaviate 外部生成查询嵌入，这会很有用。

```python
search_query = "algorithms for fast vector lookup"

print(f"\n正在执行语义搜索：'{search_query}'")

# 使用 nearText（Weaviate 嵌入查询）
response = (
    client.query
    .get("Document", ["title", "content", "_additional {certainty distance id}"]) # 指定要返回的属性和附加信息
    .with_near_text({"concepts": [search_query]})
    .with_limit(3) # 限制结果数量
    .do()
)

print("\n语义搜索结果 (nearText):")
import json
print(json.dumps(response, indent=2))

# 使用 nearVector 的示例（如果您有查询向量）
# 假设 'query_vector' 是表示嵌入的列表或 NumPy 数组
# query_vector = [...]
# response_vector = (
#     client.query
#     .get("Document", ["title", "content", "_additional {certainty distance id}"])
#     .with_near_vector({"vector": query_vector})
#     .with_limit(3)
#     .do()
# )
# print("\n语义搜索结果 (nearVector):")
# print(json.dumps(response_vector, indent=2))
```

`_additional` 字段允许检索有关搜索结果的元数据，例如 `certainty`（Weaviate 的相似度分数，越高越好）、`distance`（向量距离）和对象的 `id` (UUID)。

#### 使用 `where` 筛选

您可以将语义搜索（或其他查询）与使用 `where` 筛选器基于对象属性进行筛选结合起来。这允许根据元数据细化搜索结果。

```python
search_query = "database systems"
min_word_count = 15

print(f"\n正在执行带筛选的语义搜索：'{search_query}'，且 word_count > {min_word_count}")

response = (
    client.query
    .get("Document", ["title", "content", "word_count", "_additional {certainty distance}"])
    .with_near_text({"concepts": [search_query]})
    .with_where({
        "path": ["word_count"],                # 用于筛选的属性
        "operator": "GreaterThan",             # 比较运算符
        "valueInt": min_word_count             # 要比较的值（注意类型提示）
    })
    .with_limit(3)
    .do()
)

print("\n带筛选的语义搜索结果:")
print(json.dumps(response, indent=2))
```

`where` 筛选器支持多种运算符（如 `Equal`、`NotEqual`、`GreaterThan`、`LessThan`、用于文本模式匹配的 `Like` 等），并且可以使用 `And` / `Or` 运算符进行嵌套，以处理复杂条件。

#### 混合搜索

Weaviate 还支持混合搜索，将语义相关性（向量相似度）与传统关键词相关性（如 BM25）结合。

```python
search_query = "python client usage"

print(f"\n正在执行混合搜索：'{search_query}'")

response = (
    client.query
    .get("Document", ["title", "content", "_additional {score explainScore id}"]) # 获取混合得分
    .with_hybrid(
        query=search_query,
        alpha=0.75, # 向量搜索的权重 (1.0 = 纯向量，0.0 = 纯关键词)
        # properties=["content^2", "title"], # 可选：为关键词搜索指定属性和权重
    )
    .with_limit(3)
    .do()
)

print("\n混合搜索结果:")
print(json.dumps(response, indent=2))
```

`alpha` 参数 (parameter)控制平衡：`alpha=1` 是纯向量搜索，`alpha=0` 是纯关键词搜索 (BM25)，中间值则混合了得分。 `_additional {score}` 反映了这种组合得分。

使用 Weaviate 客户端涉及定义与您的数据和使用场景匹配的模式，运用批处理操作实现高效数据摄取，并使用其灵活的查询语言将语义搜索与元数据筛选或混合方法结合。这为构建复杂的搜索应用提供了稳固的根基，此类应用能够理解您数据中的含义。

## 参考资料

- [Weaviate Documentation: Python Client, Schema, Data Management, and Querying](https://weaviate.io/developers/weaviate/current) — Weaviate Team (2024)
  Publisher: Weaviate
  官方文档，涵盖Python客户端的安装、连接、模式定义、CRUD操作和查询，包括向量器和搜索类型的详细信息。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  介绍了 Sentence-BERT，一种生成高质量句子嵌入的方法，与 Weaviate 的 `text2vec-transformers` 模块如何进行语义搜索有关。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20Pinecone%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md) · [下一节](04-%E4%BD%BF%E7%94%A8%20Milvus%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
