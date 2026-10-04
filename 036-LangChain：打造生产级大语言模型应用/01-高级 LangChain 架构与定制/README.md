# 第 1 章：高级 LangChain 架构与定制

来源：[原章节](https://apxml.com/zh/courses/langchain-production-llm/chapter-1-advanced-langchain-architecture)

[返回课程目录](../README.md)

使用 LangChain 构建生产级应用，需要不止于基本的顺序链，并理解框架的底层结构和扩展点。本章着重介绍内部机制和定制能力，这些对构建更精巧和特定用途的 LLM 驱动系统非常必要。

您将研习 LangChain 表达式语言 (LCEL)，以掌握组件的组合方式和运行原理。我们将介绍实现异步操作以提升性能、定制 LLM 包装器、提示模板和输出解析器等核心要素，并应用高级解析技术来处理难度较高的 LLM 输出。此外，您还将学习在复杂序列中管理状态的方法，以及调试运行流程的技巧。本章最后是一个构建和集成自定义链组件的实践练习。这些技能将为根据特定应用需求定制 LangChain 提供支撑。

## 小节

- 1. [LangChain 表达式语言 (LCEL) 内部机制](01-LangChain%20%E8%A1%A8%E8%BE%BE%E5%BC%8F%E8%AF%AD%E8%A8%80%20%28LCEL%29%20%E5%86%85%E9%83%A8%E6%9C%BA%E5%88%B6.md)
- 2. [异步操作与并发](02-%E5%BC%82%E6%AD%A5%E6%93%8D%E4%BD%9C%E4%B8%8E%E5%B9%B6%E5%8F%91.md)
- 3. [定制核心组件：大语言模型、提示词、解析器](03-%E5%AE%9A%E5%88%B6%E6%A0%B8%E5%BF%83%E7%BB%84%E4%BB%B6%EF%BC%9A%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8F%90%E7%A4%BA%E8%AF%8D%E3%80%81%E8%A7%A3%E6%9E%90%E5%99%A8.md)
- 4. [高级输出解析策略](04-%E9%AB%98%E7%BA%A7%E8%BE%93%E5%87%BA%E8%A7%A3%E6%9E%90%E7%AD%96%E7%95%A5.md)
- 5. [管理复杂链中的状态](05-%E7%AE%A1%E7%90%86%E5%A4%8D%E6%9D%82%E9%93%BE%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81.md)
- 6. [调试 LangChain 执行流程](06-%E8%B0%83%E8%AF%95%20LangChain%20%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.md)
- 7. [动手实践：构建自定义链组件](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E9%93%BE%E7%BB%84%E4%BB%B6.md)
