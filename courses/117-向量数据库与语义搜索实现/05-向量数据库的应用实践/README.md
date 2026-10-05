# 第 5 章：向量数据库的应用实践

来源：[原章节](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice)

[返回课程目录](../README.md)

在对向量嵌入、数据库结构和近似最近邻搜索算法有了基本认识之后，我们将重点转向它们的实际使用。本章连接理论与实践，说明如何使用特定的向量数据库系统来构建可用的语义搜索方案。

您将直接操作包括 Pinecone、Weaviate、Milvus 和 ChromaDB 在内的流行向量数据库的客户端库。我们将介绍常见的工作流程：连接数据库、定义数据结构或集合、同时索引向量数据和元数据，以及执行相似性搜索（通常与元数据过滤结合使用）。

此外，我们还会讨论实用考量，比如选择合适的数据库平台（托管式或自建式）、有效索引大量数据的策略，以及监控系统运行状况和性能的基本方法。本章最后将通过一个实践练习，让您整合这些组成部分，从而搭建一个虽小但完整的语义搜索应用。

## 小节

- 1. [选择向量数据库平台](01-%E9%80%89%E6%8B%A9%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E5%B9%B3%E5%8F%B0.md)
- 2. [使用 Pinecone 客户端](02-%E4%BD%BF%E7%94%A8%20Pinecone%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
- 3. [使用 Weaviate 客户端](03-%E4%BD%BF%E7%94%A8%20Weaviate%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
- 4. [使用 Milvus 客户端](04-%E4%BD%BF%E7%94%A8%20Milvus%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
- 5. [使用 ChromaDB 客户端](05-%E4%BD%BF%E7%94%A8%20ChromaDB%20%E5%AE%A2%E6%88%B7%E7%AB%AF.md)
- 6. [高效索引大型数据集](06-%E9%AB%98%E6%95%88%E7%B4%A2%E5%BC%95%E5%A4%A7%E5%9E%8B%E6%95%B0%E6%8D%AE%E9%9B%86.md)
- 7. [监控与维护](07-%E7%9B%91%E6%8E%A7%E4%B8%8E%E7%BB%B4%E6%8A%A4.md)
- 8. [动手实践：构建小型语义搜索应用](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%B0%8F%E5%9E%8B%E8%AF%AD%E4%B9%89%E6%90%9C%E7%B4%A2%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice/quiz)
