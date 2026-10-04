---
course: "getting-started-model-context-protocol"
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-3-implementing-tools-and-logic"
sourceId: 1365
chapter: "implementing-tools-and-logic"
title: "实现工具和逻辑"
order: 3
description: "开发 MCP 工具，让AI模型能够执行操作、运行逻辑并与API交互。"
hasQuiz: false
---

资源为大型语言模型 (LLM) 了解某个业务范围提供所需信息，但它们按设计是只读的。为了让模型能够修改系统或根据动态参数获取特定数据，例如查询数据库或调用外部API，您必须实现工具。工具与资源的区别在于，它们是服务器提供的可执行功能，当模型判断需要执行某个操作时，客户端可以调用它们。

本章侧重于这些可执行基础组件的服务器端实现。您将使用 JSON Schema 定义工具的接口，确保模型准确理解需要哪些参数以及该工具完成什么任务。我们将使用 Pydantic 应用输入验证，以确保类型安全并在执行任何逻辑之前处理边界情况。

此外，我们将研究几种模式，用于将外部API调用封装到 MCP 工具中，以及管理通过 JSON-RPC 层进行错误传递的方式。一个常见的架构考量是在计算资源和工具之间进行选择；我们将分析做出这种区分的依据。到本章结束时，您将通过构建一个功能服务器来应用所学内容，该服务器能够对本地 SQLite 数据库执行 SQL 查询。
