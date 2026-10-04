# 第 7 章：构建检索增强生成 (RAG) 系统

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems)

[返回课程目录](../README.md)

大型语言模型通常无法获取特定、私有或非常新的信息。检索增强生成 (RAG) 提供了一种方法，通过将模型的响应基于外部数据源来解决这一局限。这种技术将大型语言模型的生成能力与信息检索机制结合起来。

在本章中，您将学习RAG背后的基本原理。我们将介绍如何整合LangChain和LlamaIndex等工具来构建RAG系统。您将了解向量存储和嵌入，它们对高效信息检索非常重要。我们将逐步讲解构建一个基本RAG流程的步骤，并讨论评估其表现的方法。到本章结束时，您将能够使用Python实现一个简单的RAG应用。

## 小节

- 1. [检索增强生成原理](01-%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%E5%8E%9F%E7%90%86.md)
- 2. [将 LlamaIndex/LangChain 整合用于 RAG](02-%E5%B0%86%20LlamaIndex-LangChain%20%E6%95%B4%E5%90%88%E7%94%A8%E4%BA%8E%20RAG.md)
- 3. [向量数据库和嵌入技术概述](03-%E5%90%91%E9%87%8F%E6%95%B0%E6%8D%AE%E5%BA%93%E5%92%8C%E5%B5%8C%E5%85%A5%E6%8A%80%E6%9C%AF%E6%A6%82%E8%BF%B0.md)
- 4. [搭建一个基础向量库](04-%E6%90%AD%E5%BB%BA%E4%B8%80%E4%B8%AA%E5%9F%BA%E7%A1%80%E5%90%91%E9%87%8F%E5%BA%93.md)
- 5. [构建 RAG 流程](05-%E6%9E%84%E5%BB%BA%20RAG%20%E6%B5%81%E7%A8%8B.md)
- 6. [评估 RAG 性能指标](06-%E8%AF%84%E4%BC%B0%20RAG%20%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87.md)
- 7. [实践：构建一个简单的RAG应用](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84RAG%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-7-building-rag-systems/quiz)
