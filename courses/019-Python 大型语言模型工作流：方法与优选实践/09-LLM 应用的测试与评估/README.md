# 第 9 章：LLM 应用的测试与评估

来源：[原章节](https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps)

[返回课程目录](../README.md)

使用大型语言模型构建应用，在验证它们的行为和表现时，会遇到一些特别的难题。与传统软件的输出结果通常是确定的不同，LLM 的响应可能会变化，这使得标准测试方法本身不够用。评估 LLM 生成内容的*质量*和*可靠性*，需要一些特别的方法和考量。

本章讨论了测试和评估你已学习构建的基于 Python 的 LLM 应用的实际方面。我们将介绍：

*   在测试由 LLM 驱动的系统时遇到的特有难点。
*   单元测试特定组件的方法，例如提示模板和输出解析器。
*   集成测试整个 LLM 工作流程的策略。
*   评估应用表现的方法，包括相关指标和人工反馈的作用。
*   旨在协助 LLM 评估的框架介绍。
*   LLM 应用内部日志记录和监控的良好实践。

在本章结束时，你将明白如何实施结构化的测试流程和评估方法，这些方法是针对 LLM 驱动系统特点而设计的。

## 小节

- 1. [LLM系统测试中的难题](01-LLM%E7%B3%BB%E7%BB%9F%E6%B5%8B%E8%AF%95%E4%B8%AD%E7%9A%84%E9%9A%BE%E9%A2%98.md)
- 2. [单元测试组件](02-%E5%8D%95%E5%85%83%E6%B5%8B%E8%AF%95%E7%BB%84%E4%BB%B6.md)
- 3. [集成测试流程](03-%E9%9B%86%E6%88%90%E6%B5%8B%E8%AF%95%E6%B5%81%E7%A8%8B.md)
- 4. [评估策略：指标与人工反馈](04-%E8%AF%84%E4%BC%B0%E7%AD%96%E7%95%A5%EF%BC%9A%E6%8C%87%E6%A0%87%E4%B8%8E%E4%BA%BA%E5%B7%A5%E5%8F%8D%E9%A6%88.md)
- 5. [使用框架进行评估](05-%E4%BD%BF%E7%94%A8%E6%A1%86%E6%9E%B6%E8%BF%9B%E8%A1%8C%E8%AF%84%E4%BC%B0.md)
- 6. [LLM 交互的日志记录与监控](06-LLM%20%E4%BA%A4%E4%BA%92%E7%9A%84%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95%E4%B8%8E%E7%9B%91%E6%8E%A7.md)
- 7. [实践：为LLM链设置基本测试](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BALLM%E9%93%BE%E8%AE%BE%E7%BD%AE%E5%9F%BA%E6%9C%AC%E6%B5%8B%E8%AF%95.md)

章节测验：[在线测验](https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/quiz)
