# 第 1 章：大语言模型红队演练的基本原理

来源：[原章节](https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-1-foundations-llm-red-teaming)

[返回课程目录](../README.md)

本章介绍大语言模型红队演练的核心内容。我们首先概述红队演练的一般做法，然后着重讲解其在大语言模型中的具体应用和重要性。您将了解到大语言模型常见的漏洞，以及红队演练生命周期中的结构化阶段。接下来的部分将讨论如何明确一次行动的目标和范围，理解攻击者视角的必要性，以及相关的法律和伦理考量。为了将这些内容付诸实践，本章最后设有一个练习，您将为一次模拟大语言模型红队演练定义范围。

## 小节

- 1. [什么是红队演练：概览](01-%E4%BB%80%E4%B9%88%E6%98%AF%E7%BA%A2%E9%98%9F%E6%BC%94%E7%BB%83%EF%BC%9A%E6%A6%82%E8%A7%88.md)
- 2. [红队测试对大型语言模型为何如此重要](02-%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E5%AF%B9%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%BA%E4%BD%95%E5%A6%82%E6%AD%A4%E9%87%8D%E8%A6%81.md)
- 3. [LLM漏洞：概述](03-LLM%E6%BC%8F%E6%B4%9E%EF%BC%9A%E6%A6%82%E8%BF%B0.md)
- 4. [LLM 红队测试生命周期](04-LLM%20%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F.md)
- 5. [LLM红队中的角色与职责](05-LLM%E7%BA%A2%E9%98%9F%E4%B8%AD%E7%9A%84%E8%A7%92%E8%89%B2%E4%B8%8E%E8%81%8C%E8%B4%A3.md)
- 6. [确立LLM红队测试的目标与范围](06-%E7%A1%AE%E7%AB%8BLLM%E7%BA%A2%E9%98%9F%E6%B5%8B%E8%AF%95%E7%9A%84%E7%9B%AE%E6%A0%87%E4%B8%8E%E8%8C%83%E5%9B%B4.md)
- 7. [掌握攻击者的思维方式](07-%E6%8E%8C%E6%8F%A1%E6%94%BB%E5%87%BB%E8%80%85%E7%9A%84%E6%80%9D%E7%BB%B4%E6%96%B9%E5%BC%8F.md)
- 8. [法律框架与负责任的披露实践](08-%E6%B3%95%E5%BE%8B%E6%A1%86%E6%9E%B6%E4%B8%8E%E8%B4%9F%E8%B4%A3%E4%BB%BB%E7%9A%84%E6%8A%AB%E9%9C%B2%E5%AE%9E%E8%B7%B5.md)
- 9. [实践：为模拟LLM红队行动界定范围](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%B8%BA%E6%A8%A1%E6%8B%9FLLM%E7%BA%A2%E9%98%9F%E8%A1%8C%E5%8A%A8%E7%95%8C%E5%AE%9A%E8%8C%83%E5%9B%B4.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intro-llm-red-teaming/chapter-1-foundations-llm-red-teaming/quiz)
