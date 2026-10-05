# 第 5 章：高级 LangChain：链与代理

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents)

[返回课程目录](../README.md)

在 LangChain 的基础元素（如模型、提示和输出解析器）之上，本章侧重于构建更复杂的流程。我们将首先研究链（Chains），它们能让您按顺序连接多个 LangChain 组件，以执行多步骤任务。

接下来，我们将介绍代理（Agents）。代理使用大型语言模型（LLM）作为推理引擎来决定一系列动作，这些动作通常涉及搜索或计算等外部工具。您将学习如何构建自定义链，理解代理的执行循环，集成工具，创建一个基本代理，并应用技术来调试这些更精巧的结构。实际练习将引导您实现这些高级方法。

## 小节

- 1. [理解用于顺序操作的链](01-%E7%90%86%E8%A7%A3%E7%94%A8%E4%BA%8E%E9%A1%BA%E5%BA%8F%E6%93%8D%E4%BD%9C%E7%9A%84%E9%93%BE.md)
- 2. [构建自定义链](02-%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E9%93%BE.md)
- 3. [代理介绍：LLM作为推理引擎](03-%E4%BB%A3%E7%90%86%E4%BB%8B%E7%BB%8D%EF%BC%9ALLM%E4%BD%9C%E4%B8%BA%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E.md)
- 4. [代理可用工具](04-%E4%BB%A3%E7%90%86%E5%8F%AF%E7%94%A8%E5%B7%A5%E5%85%B7.md)
- 5. [建立一个基本智能体](05-%E5%BB%BA%E7%AB%8B%E4%B8%80%E4%B8%AA%E5%9F%BA%E6%9C%AC%E6%99%BA%E8%83%BD%E4%BD%93.md)
- 6. [调试链和代理](06-%E8%B0%83%E8%AF%95%E9%93%BE%E5%92%8C%E4%BB%A3%E7%90%86.md)
- 7. [实践：实现多步链](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%A4%9A%E6%AD%A5%E9%93%BE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/quiz)
