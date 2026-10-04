---
course: "building-advanced-llm-agent-tools"
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-4-integrating-external-apis-tools"
sourceId: 1171
chapter: "integrating-external-apis-tools"
title: "将外部API整合为工具"
order: 4
description: "学习将第三方API集成作为LLM智能体的工具，内容涵盖身份验证、数据处理和API管理。"
hasQuiz: true
---

LLM智能体在运作时，信息通常局限于其训练数据，缺乏直接访问当前数据或在外部系统执行操作的能力。外部应用程序接口（API）提供了一种结构化的连接方式，使这些智能体能够与各种服务和数据源进行互动。本章将详细指导如何有效地封装外部API，将它们转化为您的LLM智能体可以使用的可靠工具。

您将学习此集成的主要技术，包括为API访问建立安全的身份验证和授权。我们将介绍解析各种API响应格式的方法，以及如何为LLM使用适当地组织数据。本章还将讨论操作上的必要事项，例如管理API调用速率限制和实施有效的重试机制。一个重要侧重是如何使LLM能够将自然语言请求映射到特定的API调用，以及随后如何为智能体简洁地处理和总结API数据。针对API工具集成的安全考虑将贯穿讨论。您还将有机会通过将一个公共API封装为功能性LLM工具来应用所学内容。
