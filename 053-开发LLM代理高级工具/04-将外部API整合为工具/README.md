# 第 4 章：将外部API整合为工具

来源：[原章节](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools)

[返回课程目录](../README.md)

LLM智能体在运作时，信息通常局限于其训练数据，缺乏直接访问当前数据或在外部系统执行操作的能力。外部应用程序接口（API）提供了一种结构化的连接方式，使这些智能体能够与各种服务和数据源进行互动。本章将详细指导如何有效地封装外部API，将它们转化为您的LLM智能体可以使用的可靠工具。

您将学习此集成的主要技术，包括为API访问建立安全的身份验证和授权。我们将介绍解析各种API响应格式的方法，以及如何为LLM使用适当地组织数据。本章还将讨论操作上的必要事项，例如管理API调用速率限制和实施有效的重试机制。一个重要侧重是如何使LLM能够将自然语言请求映射到特定的API调用，以及随后如何为智能体简洁地处理和总结API数据。针对API工具集成的安全考虑将贯穿讨论。您还将有机会通过将一个公共API封装为功能性LLM工具来应用所学内容。

## 小节

- 1. [工具API访问的认证与授权](01-%E5%B7%A5%E5%85%B7API%E8%AE%BF%E9%97%AE%E7%9A%84%E8%AE%A4%E8%AF%81%E4%B8%8E%E6%8E%88%E6%9D%83.md)
- 2. [解析与转换API响应](02-%E8%A7%A3%E6%9E%90%E4%B8%8E%E8%BD%AC%E6%8D%A2API%E5%93%8D%E5%BA%94.md)
- 3. [处理 API 调用频率限制与重试](03-%E5%A4%84%E7%90%86%20API%20%E8%B0%83%E7%94%A8%E9%A2%91%E7%8E%87%E9%99%90%E5%88%B6%E4%B8%8E%E9%87%8D%E8%AF%95.md)
- 4. [自然语言到API调用的映射方法](04-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%88%B0API%E8%B0%83%E7%94%A8%E7%9A%84%E6%98%A0%E5%B0%84%E6%96%B9%E6%B3%95.md)
- 5. [将API数据汇总并呈现给LLM](05-%E5%B0%86API%E6%95%B0%E6%8D%AE%E6%B1%87%E6%80%BB%E5%B9%B6%E5%91%88%E7%8E%B0%E7%BB%99LLM.md)
- 6. [API工具整合的安全方面](06-API%E5%B7%A5%E5%85%B7%E6%95%B4%E5%90%88%E7%9A%84%E5%AE%89%E5%85%A8%E6%96%B9%E9%9D%A2.md)
- 7. [实践：封装公共API作为LLM工具](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%81%E8%A3%85%E5%85%AC%E5%85%B1API%E4%BD%9C%E4%B8%BALLM%E5%B7%A5%E5%85%B7.md)

章节测验：[在线测验](https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools/quiz)
