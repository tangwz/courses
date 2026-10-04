---
course: "vector-databases-semantic-search"
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-5-vector-databases-in-practice"
sourceId: 741
chapter: "vector-databases-in-practice"
title: "向量数据库的应用实践"
order: 5
description: "使用流行的向量数据库（Pinecone、Weaviate、Milvus、Chroma）实现解决方案，处理大数据集，并考虑系统监控。"
hasQuiz: true
---

在对向量嵌入、数据库结构和近似最近邻搜索算法有了基本认识之后，我们将重点转向它们的实际使用。本章连接理论与实践，说明如何使用特定的向量数据库系统来构建可用的语义搜索方案。

您将直接操作包括 Pinecone、Weaviate、Milvus 和 ChromaDB 在内的流行向量数据库的客户端库。我们将介绍常见的工作流程：连接数据库、定义数据结构或集合、同时索引向量数据和元数据，以及执行相似性搜索（通常与元数据过滤结合使用）。

此外，我们还会讨论实用考量，比如选择合适的数据库平台（托管式或自建式）、有效索引大量数据的策略，以及监控系统运行状况和性能的基本方法。本章最后将通过一个实践练习，让您整合这些组成部分，从而搭建一个虽小但完整的语义搜索应用。
