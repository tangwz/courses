# 第 7 章：输出解析、校验与应用稳定性

来源：[原章节](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability)

[返回课程目录](../README.md)

使用大型语言模型生成文本通常只是第一步。原始输出可能不一致，或者格式难以被软件直接使用。将LLM响应可靠地集成到应用中，需要特定的方法来处理这种变化。

本章侧重于处理LLM输出，以构建可靠的应用。您将学习如何引导模型生成结构化数据，将响应解析成可用格式（如JSON），验证结果的结构和内容，并实施重试机制和内容过滤等策略，以处理错误并提升应用稳定性。我们将讨论如何使用输出解析器、数据校验库，以及处理LLM输出不符合预期情况的方法。

## 小节

- 1. [LLM输出一致性面临的挑战](01-LLM%E8%BE%93%E5%87%BA%E4%B8%80%E8%87%B4%E6%80%A7%E9%9D%A2%E4%B8%B4%E7%9A%84%E6%8C%91%E6%88%98.md)
- 2. [提示生成结构化数据（再谈）](02-%E6%8F%90%E7%A4%BA%E7%94%9F%E6%88%90%E7%BB%93%E6%9E%84%E5%8C%96%E6%95%B0%E6%8D%AE%EF%BC%88%E5%86%8D%E8%B0%88%EF%BC%89.md)
- 3. [使用输出解析器](03-%E4%BD%BF%E7%94%A8%E8%BE%93%E5%87%BA%E8%A7%A3%E6%9E%90%E5%99%A8.md)
- 4. [数据验证技术（例如 Pydantic）](04-%E6%95%B0%E6%8D%AE%E9%AA%8C%E8%AF%81%E6%8A%80%E6%9C%AF%EF%BC%88%E4%BE%8B%E5%A6%82%20Pydantic%EF%BC%89.md)
- 5. [处理解析错误](05-%E5%A4%84%E7%90%86%E8%A7%A3%E6%9E%90%E9%94%99%E8%AF%AF.md)
- 6. [实施重试机制](06-%E5%AE%9E%E6%96%BD%E9%87%8D%E8%AF%95%E6%9C%BA%E5%88%B6.md)
- 7. [内容审核与内容过滤API](07-%E5%86%85%E5%AE%B9%E5%AE%A1%E6%A0%B8%E4%B8%8E%E5%86%85%E5%AE%B9%E8%BF%87%E6%BB%A4API.md)
- 8. [实践：实现可靠的输出处理](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%8F%AF%E9%9D%A0%E7%9A%84%E8%BE%93%E5%87%BA%E5%A4%84%E7%90%86.md)

章节测验：[在线测验](https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-7-output-parsing-validation-reliability/quiz)
