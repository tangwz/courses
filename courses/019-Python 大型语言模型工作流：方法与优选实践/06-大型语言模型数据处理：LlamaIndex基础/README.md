# 第 6 章：大型语言模型数据处理：LlamaIndex基础

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics)

[返回课程目录](../README.md)

大型语言模型通常需要获取特定的外部信息，以给出相关且准确的回复，特别是在处理私有数据集或超出其初始训练截止日期的数据时。将数据直接输入到提示词中的标准方法效率不高，或受到上下文窗口限制。本章将介绍LlamaIndex，这是一个专门用于管理大型语言模型与您的外部数据源之间连接的Python库。

您将学习使用LlamaIndex所涉及的基本操作：
*   从文档和网页等各种格式中摄取数据。
*   将这些数据结构化成优化的索引以实现高效检索。
*   查询这些索引，为您的大型语言模型应用查找相关信息。

我们将介绍LlamaIndex的基本组成部分，包括其核心构成，例如节点（Nodes）和索引（Indexes），并练习从您自己的数据中加载、索引和检索信息。

## 小节

- 1. [LlamaIndex 简介](01-LlamaIndex%20%E7%AE%80%E4%BB%8B.md)
- 2. [加载数据 (文档、网页)](02-%E5%8A%A0%E8%BD%BD%E6%95%B0%E6%8D%AE%20%28%E6%96%87%E6%A1%A3%E3%80%81%E7%BD%91%E9%A1%B5%29.md)
- 3. [高效检索的数据索引](03-%E9%AB%98%E6%95%88%E6%A3%80%E7%B4%A2%E7%9A%84%E6%95%B0%E6%8D%AE%E7%B4%A2%E5%BC%95.md)
- 4. [理解节点和索引](04-%E7%90%86%E8%A7%A3%E8%8A%82%E7%82%B9%E5%92%8C%E7%B4%A2%E5%BC%95.md)
- 5. [查询您的已索引数据](05-%E6%9F%A5%E8%AF%A2%E6%82%A8%E7%9A%84%E5%B7%B2%E7%B4%A2%E5%BC%95%E6%95%B0%E6%8D%AE.md)
- 6. [动手实践：文档索引与查询](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%96%87%E6%A1%A3%E7%B4%A2%E5%BC%95%E4%B8%8E%E6%9F%A5%E8%AF%A2.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics/quiz)
