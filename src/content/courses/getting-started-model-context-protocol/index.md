---
course: "getting-started-model-context-protocol"
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol"
sourceId: 244
title: "模型上下文协议 (MCP) 入门"
description: "使用模型上下文协议，搭建自定义服务器，将大型语言模型与任意数据源相连。"
category: "Large Language Models"
level: 2
duration: 8
order: 110
prerequisites: "具备Python中级知识。"
color: "red"
chapterCount: 4
lessonCount: 24
outcomes: [{"topic": "MCP架构", "description": "理解客户端-主机-服务器关系及支撑协议运作的JSON-RPC消息流。"}, {"topic": "服务器实现", "description": "创建功能完备的MCP服务器，通过资源提供静态和动态数据。"}, {"topic": "工具创建", "description": "开发可执行工具，让LLMs能够执行操作并从外部API获取数据。"}, {"topic": "客户端对接", "description": "配置并调试自定义MCP服务器与标准客户端（如Claude Desktop）之间的连接。"}]
hasProject: false
---

模型上下文 (context)协议 (MCP) 定义了连接AI助手与数据系统的标准接口。本课程介绍该协议的架构定义，使开发者能够构建对外提供本地和远程资源供大型语言模型 (LLMs) 使用的服务器。我们将了解 MCP 的主要构成要素：资源、提示和工具，并使用官方SDK实现它们。您将掌握传输配置、处理JSON-RPC消息流程，并将自定义服务器与符合MCP的客户端（如Claude Desktop）进行对接。内容侧重于创建可靠上下文提供程序所需的技术规范和实现细节。
