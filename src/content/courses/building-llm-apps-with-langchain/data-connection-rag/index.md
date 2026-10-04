---
course: "building-llm-apps-with-langchain"
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-5-data-connection-rag"
sourceId: 1288
chapter: "data-connection-rag"
title: "检索增强生成 (RAG) 的数据连接"
order: 5
description: "将您的大语言模型连接到私有数据源。了解文档加载器、文本分割器、向量存储和检索器，以构建RAG系统。"
hasQuiz: false
---

迄今为止，我们使用的大语言模型都依赖于其内部预训练知识运行。这种方式有一个明显的局限：模型无法得知其训练截止日期之后生成的数据，也无法获取私有或专业信息。本章将介绍一种方法，通过将大语言模型连接到外部数据源来解决这个问题。

这种方法称为检索增强生成 (RAG)。其基本思路是首先从外部知识库中获取相关文档，然后将这些文档作为生成回复的上下文提供给大语言模型。这使得模型输出的内容基于事实、时效性和具体信息。

在本章中，我们将逐步构建一个RAG系统。您将学习如何：

*   **加载文档：** 使用LangChain的`DocumentLoaders`从PDF和文本文件等多种文件格式中获取数据。
*   **处理文本：** 使用`TextSplitters`将大型文档分割成更小、易于管理的部分。
*   **创建嵌入向量：** 将文本片段转换为数值向量，从而实现语义比较。
*   **存储与检索：** 使用`VectorStore`对这些嵌入向量进行索引，以便进行高效的相似性搜索，其中相关性由查询向量 $v_q$ 与文档向量 $v_d$ 之间的距离来确定。
*   **构建问答链：** 将这些部分与检索器和大语言模型组合起来，构建一个能够处理您自己的数据的完整问答应用程序。
