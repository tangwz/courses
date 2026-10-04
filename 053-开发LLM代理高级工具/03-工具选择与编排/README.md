# 第 3 章：工具选择与编排

来源：[原章节](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-3-llm-tool-selection-orchestration)

[返回课程目录](../README.md)

单个工具开发完成后，随之而来的问题是如何让LLM代理在特定情况下有效选择合适的工具，并协调多个工具进行更复杂的操作。本章将专门讨论这一点：LLM代理如何选择和排序工具。我们将研究代理驱动的工具选择方法、多步骤工具执行流程的设计，以及管理工具调用之间依赖关系的方法。你还将学习如何为工具使用实现条件逻辑，并处理顺序和并行工具执行的协调问题。

## 小节

- 1. [智能体驱动的工具选择机制](01-%E6%99%BA%E8%83%BD%E4%BD%93%E9%A9%B1%E5%8A%A8%E7%9A%84%E5%B7%A5%E5%85%B7%E9%80%89%E6%8B%A9%E6%9C%BA%E5%88%B6.md)
- 2. [设计多步工具执行流程](02-%E8%AE%BE%E8%AE%A1%E5%A4%9A%E6%AD%A5%E5%B7%A5%E5%85%B7%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.md)
- 3. [管理工具调用间的依赖关系](03-%E7%AE%A1%E7%90%86%E5%B7%A5%E5%85%B7%E8%B0%83%E7%94%A8%E9%97%B4%E7%9A%84%E4%BE%9D%E8%B5%96%E5%85%B3%E7%B3%BB.md)
- 4. [条件工具执行逻辑](04-%E6%9D%A1%E4%BB%B6%E5%B7%A5%E5%85%B7%E6%89%A7%E8%A1%8C%E9%80%BB%E8%BE%91.md)
- 5. [工具链故障恢复](05-%E5%B7%A5%E5%85%B7%E9%93%BE%E6%95%85%E9%9A%9C%E6%81%A2%E5%A4%8D.md)
- 6. [实现工具的顺序与并行使用](06-%E5%AE%9E%E7%8E%B0%E5%B7%A5%E5%85%B7%E7%9A%84%E9%A1%BA%E5%BA%8F%E4%B8%8E%E5%B9%B6%E8%A1%8C%E4%BD%BF%E7%94%A8.md)
- 7. [实践：协调多工具代理](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%8D%8F%E8%B0%83%E5%A4%9A%E5%B7%A5%E5%85%B7%E4%BB%A3%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-3-llm-tool-selection-orchestration/quiz)
