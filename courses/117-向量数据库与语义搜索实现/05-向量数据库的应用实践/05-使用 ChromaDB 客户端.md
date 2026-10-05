# 使用 ChromaDB 客户端

来源：[原文](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice/working-with-chromadb)

[返回章节目录](README.md) · [返回课程目录](../README.md)

ChromaDB 提供了一种对开发者友好的向量 (vector)存储与查找方式，因其简洁性、本地优先功能以及与流行 Python 数据科学工具的结合而受到青睐。它既可在内存中运行以用于快速测试，也能将数据保存到磁盘以在会话之间保持数据，或作为客户端/服务器应用运行。这种灵活性使其成为开发、原型制作和小型部署的合适选项。

### 开始使用 ChromaDB 客户端

首先，确保已安装 `chromadb` 库。如果尚未安装，可以使用 pip 进行安装：

```bash
pip install chromadb
```

与 ChromaDB 交互的主要方式是通过其客户端对象。您实例化的客户端类型取决于您计划如何使用 ChromaDB：

1. **内存客户端：** 非常适合临时使用、测试或不需要数据持久化的情况。进程结束时数据会丢失。

   ```python
   import chromadb

   # 创建一个内存客户端
   client = chromadb.Client()
   print("Client type: In-memory")
   ```
2. **持久化客户端：** 在指定目录下将数据存储到磁盘，使数据能在不同会话中存在。

   ```python
   import chromadb

   # 创建一个持久化客户端，将数据存储在 './chroma_db'
   client = chromadb.PersistentClient(path="./chroma_db")
   print("Client type: Persistent")
   ```
3. **HTTP 客户端：** 连接到正在运行的 ChromaDB 服务器实例，通常用于生产或共享环境。您需要服务器的主机和端口。

   ```python
   import chromadb

   # 例子：连接到运行在 localhost:8000 的服务器
   # 如果不同，请替换为您的实际服务器主机和端口
   try:
       client = chromadb.HttpClient(host='localhost', port=8000)
       print("Client type: HTTP Client (Connected)")
   except Exception as e:
       print(f"Could not connect to ChromaDB server: {e}")
       print("请确保 ChromaDB 服务器正在指定的主机和端口运行。")
       # 适当处理连接错误
       client = None # 将客户端设置为 None 或退出
   ```

> ChromaDB 客户端连接到存储或服务器实例的不同方式。

### 使用集合

在 ChromaDB 中，数据被组织成*集合*。可以将集合看作关系数据库中的表或其它向量 (vector)数据库中的索引。每个集合通常存储相同维度的向量及其相关元数据。

您可以使用 `create_collection()` 或 `get_or_create_collection()` 创建新集合或获取现有集合的句柄。后者通常更方便，因为如果集合已存在，它不会引发错误。

```python
# 使用之前初始化的 'client' 变量（假设它不是 None）
if client:
    # 创建一个名为 'document_embeddings' 的新集合
    # 如果已存在，这将引发错误
    try:
        collection = client.create_collection("document_embeddings")
        print("集合 'document_embeddings' 已创建。")
    except Exception as e:
        print(f"无法创建集合: {e}")

    # 获取或创建一个集合（更常用）
    # 如果 'search_index' 存在，则获取它。否则，创建它。
    collection = client.get_or_create_collection("search_index")
    print(f"已获取或创建集合: {collection.name}")

    # 您可以在创建时指定嵌入函数
    # 默认情况下，ChromaDB 使用 Sentence Transformers (all-MiniLM-L6-v2)
    # 如果需要，稍后会介绍更多嵌入函数
    # collection = client.get_or_create_collection(
    #     name="my_custom_embeddings",
    #     embedding_function=custom_embedding_function # 替换为您的函数
    # )
else:
    print("ChromaDB 客户端未初始化。")
```

### 向集合添加数据

一旦您拥有一个集合对象，就可以使用 `add()` 方法添加数据。您需要为每个条目提供唯一 ID，通常您会提供原始文档（文本）和相关的元数据。

如果您不直接指定嵌入 (embedding)，ChromaDB 配置的嵌入函数（默认的 Sentence Transformer，或您指定的函数）将自动从 `documents` 生成它们。

```python
if client and collection: # 确保客户端和集合有效
    try:
        collection.add(
            documents=[
                "This is the first document about apples.",
                "The second document discusses oranges and citrus fruits.",
                "A final document mentioning bananas and tropical fruits."
            ],
            metadatas=[
                {"source": "doc_1", "topic": "fruit"},
                {"source": "doc_2", "topic": "fruit"},
                {"source": "doc_3", "topic": "fruit"}
            ],
            ids=["id1", "id2", "id3"] # 每个文档的唯一 ID
        )
        print("已成功向集合添加 3 个文档。")

        # 您也可以添加带有预计算嵌入的数据
        # 确保维度与集合的预期相符
        # collection.add(
        #     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
        #     metadatas=[{"source": "precomputed_1"}, {"source": "precomputed_2"}],
        #     ids=["pc_id1", "pc_id2"]
        # )

    except Exception as e:
        print(f"添加文档出错: {e}")
        # 处理可能的问题，例如重复的 ID
else:
    print("客户端或集合不可用于添加数据。")
```

`add()` 方法的主要参数 (parameter)：

- `documents`：字符串列表（文本内容）。如果未提供嵌入且已配置嵌入函数，则需要此参数。
- `embeddings`：列表/数组的列表（向量 (vector)嵌入）。如果未提供文档或您想使用预计算的向量，则需要此参数。
- `metadatas`：字典列表，其中每个字典包含对应文档/嵌入的元数据键值对。
- `ids`：唯一字符串列表，用于标识每个条目。这些是必需的，并且在集合内必须唯一。

### 查询集合

向量 (vector)数据库的主要功能是相似性查找。在 ChromaDB 中，您在集合对象上使用 `query()` 方法。您可以使用文本（文本将自动嵌入 (embedding)）进行查询，或直接提供查询向量。

```python
if client and collection: # 确保客户端和集合有效
    try:
        # 使用文本查询 - ChromaDB 会自动嵌入查询文本
        results = collection.query(
            query_texts=["Tell me about tropical fruits"],
            n_results=2 # 请求最相似的 2 个结果
        )
        print("\n查询结果 (文本查询):")
        print(results)

        # 例子：使用预计算的嵌入向量进行查询
        # 假设 'query_vector' 是正确维度的列表或 numpy 数组
        # query_vector = [0.15, 0.25, ...] # 替换为实际向量
        # results_embedding = collection.query(
        #     query_embeddings=[query_vector],
        #     n_results=1
        # )
        # print("\n查询结果 (嵌入查询):")
        # print(results_embedding)

    except Exception as e:
        print(f"\n查询集合出错: {e}")
else:
    print("\n客户端或集合不可用于查询。")
```

`query()` 方法返回一个字典，其中包含 `ids`、`distances`、`metadatas`、`embeddings` 和 `documents` 的列表，对应于找到的最近邻居。

### 使用元数据过滤

通常，您需要将语义相似性查找与基于元数据属性的过滤结合。ChromaDB 允许在 `query()` 方法中使用 `where` 参数 (parameter)来实现此功能。它接受一个字典来指定过滤条件。

```python
if client and collection: # 确保客户端和集合有效
    try:
        # 查找与“柑橘”相似但仅来自 'doc_2' 的文档
        filtered_results = collection.query(
            query_texts=["citrus"],
            n_results=1,
            where={"source": "doc_2"} # 过滤元数据字段 'source' 等于 'doc_2' 的项
        )
        print("\n查询结果 (带 'where' 过滤器):")
        print(filtered_results)

        # 示例：使用文档过滤器（简单情况不常用，更适用于复杂逻辑）
        # filtered_results_doc = collection.query(
        #     query_texts=["apples"],
        #     n_results=1,
        #     where_document={"$contains": "first"} # 过滤包含单词“first”的文档
        # )
        # print("\n查询结果 (带 'where_document' 过滤器):")
        # print(filtered_results_doc)

    except Exception as e:
        print(f"\n使用过滤器查询集合出错: {e}")
else:
    print("\n客户端或集合不可用于带过滤器的查询。")
```

ChromaDB 在 `where` 过滤器中支持多种运算符，例如 `$eq`（等于，默认）、`$ne`（不等于）、`$gt`（大于）、`$lt`（小于）、`$gte`（大于或等于）、`$lte`（小于或等于）、`$in`、`$nin`。有关完整的过滤功能，请参阅 ChromaDB 文档。

### 其他操作

ChromaDB 的客户端还提供用于管理数据的方法：

- `get()`：按 ID 获取项目，可选择应用过滤器。
- `update()`：修改现有项目（文档、嵌入 (embedding)、元数据）。
- `upsert()`：如果 ID 不存在则添加项目，如果存在则更新项目。
- `delete()`：按 ID 或通过应用过滤器删除项目。
- `peek()`：从集合中获取少量项目（用于检查）。
- `count()`：获取集合中的项目总数。

```python
if client and collection: # 确保客户端和集合有效
    try:
        print(f"\n集合中的总项目数: {collection.count()}")

        # 按 ID 获取项目
        item = collection.get(ids=["id1"])
        print(f"\n已获取项目 'id1': {item}")

        # 更新项目的元数据
        collection.update(ids=["id1"], metadatas=[{"source": "doc_1_updated", "topic": "fruit", "reviewed": True}])
        item_updated = collection.get(ids=["id1"])
        print(f"\n已更新项目 'id1': {item_updated}")

        # 删除项目
        collection.delete(ids=["id3"])
        print(f"\n删除 'id3' 后的项目数: {collection.count()}")

    except Exception as e:
        print(f"\n执行其他操作时出错: {e}")
else:
    print("\n客户端或集合不可用于其他操作。")
```

ChromaDB 提供了一个简单直接且符合 Python 习惯的接口，用于构建向量 (vector)查找应用。它侧重于易用性和灵活的部署选项，使其成为入门的绝佳工具，也适用于那些托管式大型服务可能过于繁重或不必要的应用。请记住查阅 ChromaDB 官方文档，以获取最新的 API 详情和高级功能。

## 参考资料

- [ChromaDB Documentation](https://docs.trychroma.com/) — ChromaDB Team (2024)
  ChromaDB 的官方用户指南，详细介绍了客户端设置、集合管理、数据操作和高级功能。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Pages: 3973-3983; DOI: [10.48550/arXiv.1908.10084](https://doi.org/10.48550/arXiv.1908.10084)
  介绍了 Sentence-BERT，一种生成语义有意义的句子嵌入的方法，ChromaDB 默认使用其进行文本嵌入。
- [Vector Databases: A Guide to Modern Data Storage for AI](https://www.oreilly.com/library/view/vector-databases/9781098159183/) — Alex Botez, Matt Rickard (2024)
  Publisher: O'Reilly Media
  提供了向量数据库的基础知识、架构、应用场景以及它们如何与 AI 应用集成，为 ChromaDB 的作用提供了更广泛的背景。

---

[上一节](04-%E4%BD%BF%E7%94%A8%20Milvus%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md) · [下一节](06-%E9%AB%98%E6%95%88%E7%B4%A2%E5%BC%95%E5%A4%A7%E5%9E%8B%E6%95%B0%E6%8D%AE%E9%9B%86.md)
