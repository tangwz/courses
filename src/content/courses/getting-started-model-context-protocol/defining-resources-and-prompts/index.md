---
course: "getting-started-model-context-protocol"
sourceUrl: "https://apxml.com/zh/courses/getting-started-model-context-protocol/chapter-2-defining-resources-and-prompts"
sourceId: 1364
chapter: "defining-resources-and-prompts"
title: "定义资源和提示词"
order: 2
description: "学习定义用于数据访问的MCP资源和用于LLM交互模板的提示词。"
hasQuiz: false
---

模型上下文协议的核心架构规定了组件间的通信方式，但这种通信的价值体现在所交换的数据上。随着传输层和连接拓扑的搭建完成，关注点将转向那些作为大型语言模型（LLM）上下文的内容载荷。本章说明用于展现数据和交互模式的两种主要基本元素：**资源**和**提示词**。

**资源**充当从服务器读取数据的标准接口。它们允许您将文件、数据库记录或API响应展现为可寻址的内容。您将定义统一资源标识符（URI）来定位这些项，并实现取回它们所需的逻辑。实现细节包含同步读取（数据立即获取）和订阅模型（当底层数据源改变时，服务器能够推送通知）。

**提示词**提供了一种标准化用户交互的方法。您可以不依赖于临时指令，而是定义结构化模板来指导LLM的行为。本节说明如何构建提示词架构，这些架构规定了所需参数并自动嵌入必要的上下文。

到本章结束时，您将从理论定义转向实际可用的代码。您将构建一个文件读取服务器，它实现了这些基本元素，使MCP客户端能够访问和读取您的本地目录结构中的内容。
