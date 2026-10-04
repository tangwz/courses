# 第 4 章：核心LLM工作流库：LangChain基础

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals)

[返回课程目录](../README.md)

在了解了如何搭建环境并直接与LLM API进行交互后，我们现在介绍旨在简化更复杂LLM应用开发的库。本章将介绍LangChain，这是一个广泛使用的Python框架，用于编排LLM工作流程。

您将了解LangChain的用途和结构。我们将讲解其基本组成部分：
*   **模型 (Models)：** 与不同LLM提供商交互的接口。
*   **提示词 (Prompts)：** 用于构建和管理发送给LLM的输入内容。
*   **输出解析器 (Output Parsers)：** 从LLM响应中提取结构化信息的方法。

我们将演示如何集成各种LLM服务，创建可复用的提示词模板，并解析模型生成的输出。本章最后将有一个实践练习，您将把学到的知识点应用于构建一个基础的LangChain应用。

## 小节

- 1. [LangChain 介绍](01-LangChain%20%E4%BB%8B%E7%BB%8D.md)
- 2. [主要构成：模型、提示词、输出解析器](02-%E4%B8%BB%E8%A6%81%E6%9E%84%E6%88%90%EF%BC%9A%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8F%90%E7%A4%BA%E8%AF%8D%E3%80%81%E8%BE%93%E5%87%BA%E8%A7%A3%E6%9E%90%E5%99%A8.md)
- 3. [在 LangChain 中使用不同的 LLM 服务](03-%E5%9C%A8%20LangChain%20%E4%B8%AD%E4%BD%BF%E7%94%A8%E4%B8%8D%E5%90%8C%E7%9A%84%20LLM%20%E6%9C%8D%E5%8A%A1.md)
- 4. [创建和使用提示模板](04-%E5%88%9B%E5%BB%BA%E5%92%8C%E4%BD%BF%E7%94%A8%E6%8F%90%E7%A4%BA%E6%A8%A1%E6%9D%BF.md)
- 5. [解析大型语言模型输出结构](05-%E8%A7%A3%E6%9E%90%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E8%BE%93%E5%87%BA%E7%BB%93%E6%9E%84.md)
- 6. [动手实践：构建一个简单的LangChain应用](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84LangChain%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/quiz)
