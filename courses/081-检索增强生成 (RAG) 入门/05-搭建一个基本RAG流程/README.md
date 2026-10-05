# 第 5 章：搭建一个基本RAG流程

来源：[原章节](https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline)

[返回课程目录](../README.md)

在讲解了RAG的基本知识，包括其架构、检索方法、数据准备技术以及生成过程后，我们现在将重心放在实际使用上。本章将带领您完成一个基本、端到端的检索增强生成流程的搭建。

您将学到完成整体搭建所需的必要步骤：

*   搭建包含所需库的合适开发环境。
*   实现检索器组件，连接向量库，并执行相似性搜索。
*   集成大型语言模型（LLM）作为生成器。
*   将检索器和生成器组合成一个连贯的序列或链条。
*   简要回顾简化RAG开发的框架（如LangChain或LlamaIndex）。
*   针对搭建好的流程执行查询，以观察其运行情况。

在本章结束时，您将搭建并测试一个使用标准工具和方法的可用的RAG系统。

## 小节

- 1. [RAG 框架概述 (例如 LangChain、LlamaIndex)](01-RAG%20%E6%A1%86%E6%9E%B6%E6%A6%82%E8%BF%B0%20%28%E4%BE%8B%E5%A6%82%20LangChain%E3%80%81LlamaIndex%29.md)
- 2. [环境配置](02-%E7%8E%AF%E5%A2%83%E9%85%8D%E7%BD%AE.md)
- 3. [实现检索器](03-%E5%AE%9E%E7%8E%B0%E6%A3%80%E7%B4%A2%E5%99%A8.md)
- 4. [实现生成器集成](04-%E5%AE%9E%E7%8E%B0%E7%94%9F%E6%88%90%E5%99%A8%E9%9B%86%E6%88%90.md)
- 5. [结合检索与生成](05-%E7%BB%93%E5%90%88%E6%A3%80%E7%B4%A2%E4%B8%8E%E7%94%9F%E6%88%90.md)
- 6. [在流程中执行查询](06-%E5%9C%A8%E6%B5%81%E7%A8%8B%E4%B8%AD%E6%89%A7%E8%A1%8C%E6%9F%A5%E8%AF%A2.md)
- 7. [动手实践：端到端RAG系统](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E7%AB%AF%E5%88%B0%E7%AB%AFRAG%E7%B3%BB%E7%BB%9F.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/quiz)
