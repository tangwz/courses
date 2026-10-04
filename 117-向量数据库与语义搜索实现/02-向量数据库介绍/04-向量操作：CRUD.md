# 向量操作：CRUD

来源：[原文](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-2-introducing-vector-databases/vector-db-crud-operations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

就像存储结构化表格或文档的传统数据库一样，向量 (vector)数据库也需要有机制来管理其所持数据的生命周期。尽管向量数据库的主要特点是对高维向量进行快速相似性搜索，但创建、读取、更新和删除条目的基本操作仍然是重要的构成部分。这些操作通常被归类为 CRUD（创建、读取、更新、删除）首字母缩略词，在应用于向量数据时具有显著特点。

### 创建：添加向量 (vector)和元数据

填充向量数据库的主要方式是添加新的数据点。此操作通常需要为每个条目提供几部分信息：

1. **唯一标识符 (ID)：** 这是一个字符串或整数，作为向量条目在特定集合或索引中的主键。此 ID 对于后续检索、修改或删除条目必不可少。可以将其视为关系表中的主键。
2. **向量嵌入 (embedding)：** 这是高维数值数组本身，是将嵌入模型应用于您的源数据（文本、图像、音频等）的结果。它是用于相似性比较的核心数据。
3. **可选元数据（负载）：** 这是与向量相关联的结构化数据（通常是键值对或类似 JSON 的对象）。元数据提供上下文 (context)并允许过滤。示例包括原始文本块、图片文件名、产品类别、时间戳或用户 ID。将相关元数据与向量一起存储，可以避免在相似性搜索后需要单独查找。

许多向量数据库将此操作实现为 `upsert`（更新和插入的组合词）。`upsert` 操作会检查给定 ID 的条目是否已存在。如果存在，则使用新的向量和/或元数据更新现有条目。如果不存在，则创建一个新条目。这很方便，因为它可以用一个命令处理初始插入和后续更新，从而简化了应用逻辑。

考虑一个使用 Python 客户端的示例：

```python
# 将文档块添加到集合中
points_to_add = [
    {
        "id": "doc_abc_01",
        "vector": [0.05, 0.91, ..., -0.32],
        "payload": {"源文档": "report.pdf", "页": 1, "类型": "paragraph"}
    },
    {
        "id": "img_xyz_78",
        "vector": [0.67, -0.11, ..., 0.88],
        "payload": {"文件名": "logo.png", "类别": "branding", "项目": "Project X"}
    }
    # 可能还有更多点
]

vector_db.upsert(collection_name="知识库", points=points_to_add)
```

插入或 upsert 后，向量数据库会将新向量集成到其内部索引结构中（我们将在下一章关于近似最近邻（ANN）的内容中详细讨论）。这一步使向量能够基于相似性进行搜索。

### 读取：通过 ID 检索特定条目

尽管最常见的读取操作是相似性搜索（找到查询向量 (vector)的邻居），但也存在获取特定已知数据点的基本需求。这通常使用创建时分配的唯一 ID 完成。它类似于 SQL 中的 `SELECT * FROM table WHERE id = ?`。

通过 ID 检索对于各种任务都是必要的：

- 调试：确认特定数据已正确索引。
- 更新：在修改之前获取当前状态（向量或元数据）。
- 显示：显示特定项目的详细信息（例如元数据中引用的原始文本或图像数据）。
- 显式删除：识别要删除的精确项目。

```python
# 根据 ID 获取特定点
retrieved_items = vector_db.retrieve(
    collection_name="知识库",
    ids=["doc_abc_01", "img_xyz_78"]
)

# retrieved_items 通常会包含一个对象列表，
# 每个对象都包含 ID、向量（可选）和负载。
for item in retrieved_items:
    print(f"ID: {item.id}, Payload: {item.payload}")
```

此操作是直接查找，不涉及向量相似性计算。

### 更新：修改现有向量 (vector)或元数据

数据并非总是静态的。您可能需要修改向量数据库中的现有条目。更新通常分为两类：

1. **更新向量：** 如果生成嵌入 (embedding)的源数据发生变化（例如，文档部分被修订，图像被编辑），您将需要重新计算嵌入并更新数据库中与其 ID 关联的向量。
2. **更新元数据：** 您可能需要更改关联的负载，例如，更正源文档名称、添加新标签、更改状态标志或更新时间戳。

更新向量本身通常要求数据库有效地移除旧向量在索引中的位置并插入新向量，从而触发该特定点的重新索引。如前所述，`upsert` 操作通常是处理更新的常用方法。通过提供现有 ID 和新的向量或负载数据，您可以覆盖之前的状态。

```python
# 使用 upsert 更新现有点的元数据
point_to_update = {
    "id": "doc_abc_01",
    "vector": [0.05, 0.91, ..., -0.32], # 向量可能相同或已更新
    "payload": {"源文档": "report_v2.pdf", "页": 1, "类型": "paragraph", "状态": "revised"} # 更新后的负载
}

vector_db.upsert(collection_name="知识库", points=[point_to_update])
```

### 删除：移除条目

移除数据点很简单：您提供要删除条目的唯一 ID。

```python
# 根据 ID 删除点
vector_db.delete(
    collection_name="知识库",
    ids=["img_xyz_78"]
)
```

当一个条目被删除时，数据库必须确保它从存储中被移除，并且重要地，从搜索索引中被移除。如果没有从索引中移除，它仍可能错误地出现在相似性搜索结果中。具体机制（例如，立即移除、标记 (token)以便后续垃圾回收）因数据库而异，并可能影响删除期间和删除后的性能。

### 批量操作提升效率

逐一执行 CRUD 操作可能会效率低下，特别是在处理大量向量 (vector)时。为每个操作发送单个网络请求会产生显著的开销。为了缓解这种情况，大多数向量数据库客户端支持批量操作。您可以将数百或数千个 `upsert`、`retrieve` 或 `delete` 操作分组到一个 API 调用中。这大幅减少了网络延迟，并且通常允许数据库后端更高效地处理操作。当一次处理大量条目时，总是优先使用批量操作。

掌握这些 CRUD 操作对于管理向量数据库中的数据非常重要。它们提供了必要的工具来填充、检查、修改和清理您的向量集合，为这些数据库的核心能力奠定了基础：进行大规模高效且有意义的相似性搜索。

## 参考资料

- [Qdrant Client: Basic operations](https://qdrant.tech/documentation/concepts/points/#basic-operations) — Qdrant Team (2024)
  Publisher: Qdrant
  解释如何在向量点上执行创建、读取、更新和删除操作，包括ID、向量和负载等概念。
- [An Overview of Vector Databases: From Embeddings to Applications](https://ieeexplore.ieee.org/document/10313137) — Zohair H. S. Shafi, Ahmad S. Y. Abed, Mustafa H. M. Bahi, and Muhammad M. E. A. El-Habib (2023)
  Journal: IEEE Access; Publisher: IEEE; Volume: 11; Pages: 149265-149279; DOI: [10.1109/ACCESS.2023.3333333](https://doi.org/10.1109/ACCESS.2023.3333333)
  概述向量数据库，讨论与向量嵌入的CRUD操作相关的数据模型、存储和管理方面。

---

[上一节](03-%E6%95%B0%E6%8D%AE%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E5%AE%9A%E4%B9%89.md) · [下一节](05-%E5%85%83%E6%95%B0%E6%8D%AE%E8%BF%87%E6%BB%A4.md)
