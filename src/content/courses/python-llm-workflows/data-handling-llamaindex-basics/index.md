---
course: "python-llm-workflows"
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-6-data-handling-llamaindex-basics"
sourceId: 962
chapter: "data-handling-llamaindex-basics"
title: "大型语言模型数据处理：LlamaIndex基础"
order: 6
description: "学习LlamaIndex的基本方法，以便为大型语言模型应用摄取、结构化和访问私有数据。"
hasQuiz: true
---

大型语言模型通常需要获取特定的外部信息，以给出相关且准确的回复，特别是在处理私有数据集或超出其初始训练截止日期的数据时。将数据直接输入到提示词中的标准方法效率不高，或受到上下文窗口限制。本章将介绍LlamaIndex，这是一个专门用于管理大型语言模型与您的外部数据源之间连接的Python库。

您将学习使用LlamaIndex所涉及的基本操作：
*   从文档和网页等各种格式中摄取数据。
*   将这些数据结构化成优化的索引以实现高效检索。
*   查询这些索引，为您的大型语言模型应用查找相关信息。

我们将介绍LlamaIndex的基本组成部分，包括其核心构成，例如节点（Nodes）和索引（Indexes），并练习从您自己的数据中加载、索引和检索信息。
