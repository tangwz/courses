# 第 2 章：开发自定义Python工具

来源：[原章节](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools)

[返回课程目录](../README.md)

在上一章建立了LLM代理工具的核心原则后，我们现在着手使用Python具体实现自定义工具。本章旨在让你掌握从零开始构建这些工具的技能。

你将学习如何将工具组织成 Python 函数或类，管理更复杂交互所需的状态，并使你的工具能与 API 和数据库等外部服务通信。我们还将讨论验证输入、为LLM使用有效格式化输出以及实现非阻塞任务的异步操作等重要事项。到本章结束时，你通过自定义 Python 代码扩展 LLM 代理能力的能力将大大加强。

## 小节

- 1. [以Python函数和类的形式实现工具](01-%E4%BB%A5Python%E5%87%BD%E6%95%B0%E5%92%8C%E7%B1%BB%E7%9A%84%E5%BD%A2%E5%BC%8F%E5%AE%9E%E7%8E%B0%E5%B7%A5%E5%85%B7.md)
- 2. [管理有状态工具中的状态和上下文](02-%E7%AE%A1%E7%90%86%E6%9C%89%E7%8A%B6%E6%80%81%E5%B7%A5%E5%85%B7%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81%E5%92%8C%E4%B8%8A%E4%B8%8B%E6%96%87.md)
- 3. [与外部服务交互：API 和数据库](03-%E4%B8%8E%E5%A4%96%E9%83%A8%E6%9C%8D%E5%8A%A1%E4%BA%A4%E4%BA%92%EF%BC%9AAPI%20%E5%92%8C%E6%95%B0%E6%8D%AE%E5%BA%93.md)
- 4. [验证和净化工具输入](04-%E9%AA%8C%E8%AF%81%E5%92%8C%E5%87%80%E5%8C%96%E5%B7%A5%E5%85%B7%E8%BE%93%E5%85%A5.md)
- 5. [大型语言模型复杂工具输出的结构化方法](05-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%A4%8D%E6%9D%82%E5%B7%A5%E5%85%B7%E8%BE%93%E5%87%BA%E7%9A%84%E7%BB%93%E6%9E%84%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 6. [异步工具操作：实现非阻塞任务](06-%E5%BC%82%E6%AD%A5%E5%B7%A5%E5%85%B7%E6%93%8D%E4%BD%9C%EF%BC%9A%E5%AE%9E%E7%8E%B0%E9%9D%9E%E9%98%BB%E5%A1%9E%E4%BB%BB%E5%8A%A1.md)
- 7. [实践：构建数据库查询工具](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E6%95%B0%E6%8D%AE%E5%BA%93%E6%9F%A5%E8%AF%A2%E5%B7%A5%E5%85%B7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-2-developing-custom-python-tools/quiz)
