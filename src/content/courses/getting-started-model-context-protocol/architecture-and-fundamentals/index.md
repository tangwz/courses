---
course: "getting-started-model-context-protocol"
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-1-architecture-and-fundamentals"
sourceId: 1363
chapter: "architecture-and-fundamentals"
title: "架构与基本原理"
order: 1
description: "理解模型上下文协议的核心架构，包括客户端-宿主-服务器模型和JSON-RPC传输机制。"
hasQuiz: false
---

为大型语言模型构建集成通常需要为每个不同的数据源创建定制连接器。模型上下文协议 (MCP) 通过提供一个标准化的开放协议来解决这种碎片化问题。它将数据提供方与客户端应用程序分离，从而允许单个服务器向任何符合MCP标准的客户端提供资源。

本章明确了构建这些集成所需的技术基准。我们侧重于架构定义，而非即时的代码实现。您将考察客户端-宿主-服务器拓扑，以了解责任如何在系统中分配。我们还会分析通信层，它依赖于JSON-RPC 2.0消息来处理请求、响应和通知。

该协议支持特定的传输机制，具体取决于连接是本地还是远程。我们将比较用于本地进程通信的标准输入/输出 (stdio) 与用于基于HTTP连接的服务器发送事件 (SSE)。理解这些传输层对于后续调试连接问题是有必要的。

在本章结束时，您将能够：

*   定义MCP客户端、宿主和服务器的角色与职责。
*   分析协议中使用的JSON-RPC消息的结构。
*   针对您的具体应用场景选择适合的传输机制。
*   阐述初始化握手期间的能力协商过程。
*   配置带有必要SDK的本地开发环境。
