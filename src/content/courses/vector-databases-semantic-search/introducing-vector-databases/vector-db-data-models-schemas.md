---
course: "vector-databases-semantic-search"
chapter: "introducing-vector-databases"
lesson: "vector-db-data-models-schemas"
sourceId: 3526
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-2-introducing-vector-databases/vector-db-data-models-schemas"
title: "数据模型与数据结构定义"
description: "How vector data and associated metadata are typically structured and stored."
order: 3
plots: []
sourceHash: "660291305d8179d287e7b4a59bdb439e127b864e5bba77081148bca72dce39ae"
sourceCorrections: []
---

传统数据库擅长基于精确匹配或对预定义字段进行范围查询来存储和检索结构化数据，而向量 (vector)数据库则围绕一个不同的核心理念构建：基于相似性管理和查找高维向量。这使得它需要为此项任务优化的独特数据模型和结构定义方式。

在向量数据库数据模型的核心，是**向量记录**（有时也称作条目、点或文档）这一思想。这是数据存储的基本单位。每个向量记录通常会包含以下几个重要信息：

1. **唯一标识符（ID）：** 就像大多数数据库一样，每个向量记录都需要一个唯一ID用于直接检索、更新或删除。这通常是由用户提供或由数据库生成的字符串或整数。
2. **向量嵌入 (embedding)：** 这是数据点的核心数值表示，通常存储为浮点数数组或列表。例如，一个768维的文本嵌入将存储为一个包含768个数字的数组。此处一个重要方面是维度一致性。在数据库中的特定集合或索引内，所有向量通常必须具有相同的维度（$d$）。尝试将不同维度的向量插入到同一个索引中通常不受支持或需要特殊处理，因为底层的ANN算法依赖于在一致的向量空间内运行。

这解释了向量数据库在应用场景中发挥其效用的原因。向量很少孤立存在；它们代表具体的实体，如文档、图像、产品或用户资料。元数据载荷存储与向量关联的属性，提供上下文 (context)并允许更精细的查找。此载荷通常结构类似JSON对象或字典，包含键值对。

### 元数据表示

与向量 (vector)关联的元数据可以包含多种数据类型，类似于NoSQL或关系型数据库中的类型：

- **字符串：** 用于存储文本信息，如产品名称、类别、文件名或文档标题。
- **数值：** 整数或浮点数，用于存储数量、价格、时间戳、评分或数值属性。
- **布尔值：** 用于表示真/假标志，例如 `is_active`、`on_sale` 或 `is_processed`。
- **列表/数组：** 常用于标签、关键词或关联ID列表。
- **地理空间数据：** 某些向量数据库支持专门的类型来存储地理坐标，以实现基于位置的过滤。
- **嵌套对象：** 某些系统允许在元数据中存在更复杂的嵌套结构。

请看一个代表文档中某段文本的向量记录示例：

```json
{
  "id": "doc1_chunk3",
  "vector": [0.013, -0.245, ..., 0.912], // 一个512维的向量
  "metadata": {
    "document_source": "/path/to/research_paper.pdf",
    "page_number": 15,
    "chunk_length": 350, // 字符或标记的数量
    "topic": "Machine Learning",
    "keywords": ["vector database", "ANN", "similarity search"],
    "published_year": 2023
  }
}
```

在这个例子中，`id` 唯一标识记录，`vector` 存储嵌入 (embedding)，`metadata` 包含有关文本片段的上下文 (context)信息。

### 结构定义：明确其构成

这种结构（向量 (vector)维度、元数据字段、数据类型）在事前必须如何严格定义，在不同的向量数据库系统之间有所不同。

- **显式结构定义：** 某些数据库在创建集合或索引时要求预先定义结构。这涉及指定预期的向量维度，并定义所有允许的元数据字段的名称和数据类型。这种方法提供数据一致性，有时可以优化存储和过滤性能，因为数据库清楚知道预期的结构是什么。
- **写入时结构定义/灵活结构定义：** 其他系统提供更高的灵活性，类似于许多NoSQL文档存储。您可能只需要指定向量维度，元数据字段可以在数据插入时更动态地添加。数据库根据接收到的数据推断出结构。尽管提供灵活性，但这可能需要仔细管理，以确保过滤逻辑所需的一致性。

无论采用何种方法，理解向量记录的结构、固定的向量维度以及可用的元数据字段及其类型都非常重要。这种结构直接影响您如何构建索引（在下一章中介绍），并且极为重要地，它影响您如何查询数据库。向量数据库的能力通常来自将向量相似性搜索与基于这些元数据属性的精确过滤结合起来（例如，“查找与此查询*语义相似*的文档，但仅限于*2022年后发布*且标记 (token)为\*'ANN'\*的文档”）。这种双重能力由一种数据模型实现，该模型将高维向量与其描述性元数据紧密关联。

## 参考资料

- [Vector Databases: A Guide to Building AI Applications](https://www.oreilly.com/library/view/vector-databases/9781098159187/) — Steven Connor, David T. Miller, and Stephen C. Eubank (2024)
  Publisher: O'Reilly Media
  一本新近出版的书籍，系统概述了向量数据库，包括其架构、核心数据模型，以及向量记录、嵌入和元数据的管理方式。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://dataintensive.net/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  这本基础书籍的第4章讨论了各种数据模型，包括文档数据库及其灵活的模式方法，为向量数据库中的元数据处理提供了有益的背景。
