# 第 5 章：使用LLM框架构建应用

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks)

[返回课程目录](../README.md)

直接与LLM API交互能提供精确控制，但构建更复杂的应用时，常会遇到重复的代码模式，并需要管理应用状态，例如对话历史。LLM框架通过提供抽象和工具来组织开发，从而简化了这一过程。

本章介绍常见的LLM应用开发框架，主要以LangChain作为代表性例子。你将学习这些框架提供的核心组件，包括模型接口、提示模板和输出解析器。我们将讲解如何使用链将这些组件连接起来，管理对话记忆，并构建能够使用外部工具来完成任务的基础代理。目标是使你能够高效地构建更复杂且更易维护的LLM应用。实践练习将涉及使用这些框架组件构建应用。

## 小节

- 1. [大型语言模型框架介绍（如LangChain）](01-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%A1%86%E6%9E%B6%E4%BB%8B%E7%BB%8D%EF%BC%88%E5%A6%82LangChain%EF%BC%89.md)
- 2. [核心构成部分：模型、提示、解析器](02-%E6%A0%B8%E5%BF%83%E6%9E%84%E6%88%90%E9%83%A8%E5%88%86%EF%BC%9A%E6%A8%A1%E5%9E%8B%E3%80%81%E6%8F%90%E7%A4%BA%E3%80%81%E8%A7%A3%E6%9E%90%E5%99%A8.md)
- 3. [了解链](03-%E4%BA%86%E8%A7%A3%E9%93%BE.md)
- 4. [LLM应用中的记忆管理](04-LLM%E5%BA%94%E7%94%A8%E4%B8%AD%E7%9A%84%E8%AE%B0%E5%BF%86%E7%AE%A1%E7%90%86.md)
- 5. [智能体简介](05-%E6%99%BA%E8%83%BD%E4%BD%93%E7%AE%80%E4%BB%8B.md)
- 6. [智能体工具的使用](06-%E6%99%BA%E8%83%BD%E4%BD%93%E5%B7%A5%E5%85%B7%E7%9A%84%E4%BD%BF%E7%94%A8.md)
- 7. [动手实践：开发一个智能体应用](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BC%80%E5%8F%91%E4%B8%80%E4%B8%AA%E6%99%BA%E8%83%BD%E4%BD%93%E5%BA%94%E7%94%A8.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/quiz)
