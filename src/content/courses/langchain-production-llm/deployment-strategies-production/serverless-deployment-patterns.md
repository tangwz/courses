---
course: "langchain-production-llm"
chapter: "deployment-strategies-production"
lesson: "serverless-deployment-patterns"
sourceId: 3927
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-7-deployment-strategies-production/serverless-deployment-patterns"
title: "LangChain 的无服务器部署模式"
description: "使用无服务器函数和托管服务构建 LangChain 应用程序。"
order: 4
plots: []
sourceHash: "dd4d0c1187a559644932f794713ec29cf44944ca36977766a911f2a8ed808281"
sourceCorrections: []
---

无服务器计算为某些类型的 LangChain 应用程序提供了一种有吸引力的部署方式，主要得益于其自动扩缩容、按使用量计费以及降低运维负担。AWS Lambda、Google Cloud Functions 和 Azure Functions 等平台允许您根据事件（例如通过 API Gateway 的 HTTP 请求）运行代码，无需管理底层服务器。这与许多 LLM 交互的事件驱动特性非常契合。

然而，部署复杂的 LangChain 应用程序，尤其是涉及有状态代理或长时间运行进程的应用程序，需要仔细考量无服务器架构及其固有局限性。

### LangChain 常见的无服务器模式

1. **无状态 API 端点：**

   - \*\*模式：\*\*API Gateway -> 无服务器函数 -> LangChain 链 -> LLM
   - \*\*描述：\*\*这是最直接的模式。一个 HTTP 请求触发一个无服务器函数（例如 AWS Lambda）。该函数实例化一个 LangChain 链（通常使用 LCEL 定义），处理请求输入，调用 LLM，解析输出，并返回响应。每次调用都是独立且无状态的。
   - \*\*使用场景：\*\*简单的问答机器人、文本生成任务、以及不需要对话历史或对话历史完全由客户端管理的数据提取端点。
   - \*\*考量：\*\*冷启动可能对一段时间不活动后的第一个请求带来延迟。包大小限制可能需要审慎的依赖管理或使用层/容器镜像。

   ```python
   # 示例 (AWS Lambda 处理函数)
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate
   from langchain_core.output_parsers import StrOutputParser
   import json

   # 假设 API 密钥通过环境变量设置
   # 初始化组件（可在处理函数外部完成，以便在暖启动时重用）
   llm = ChatOpenAI(model="gpt-4o-mini")
   prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
   parser = StrOutputParser()
   chain = prompt | llm | parser

   def lambda_handler(event, context):
       try:
           # 从 API Gateway 事件体中提取主题
           body = json.loads(event.get('body', '{}'))
           topic = body.get('topic', 'computers')

           # 调用链
           result = chain.invoke({"topic": topic})

           return {
               'statusCode': 200,
               'body': json.dumps({'joke': result})
           }
       except Exception as e:
           # 基本错误处理
           return {
               'statusCode': 500,
               'body': json.dumps({'error': str(e)})
           }
   ```
2. **带外部向量 (vector)存储的 RAG API：**

   - \*\*模式：\*\*API Gateway -> 无服务器函数 -> 查询向量存储 -> 构建提示词 (prompt) -> LLM
   - \*\*描述：\*\*对于检索增强生成 (RAG)，函数首先接收查询，然后连接到外部托管向量存储（如 Pinecone、Weaviate Cloud，或无服务器函数外部的自管理存储）以检索相关文档。这些文档用于增强发送给 LLM 的提示词。
   - \*\*使用场景：\*\*文档问答系统、访问知识库的客户支持机器人。
   - \*\*考量：\*\*增加了到向量存储的网络延迟。有效管理数据库连接（例如在暖调用之间重用连接）很重要。影响函数和潜在初始连接设置的冷启动会增加总响应时间。到向量存储的认证必须安全地处理，通常通过环境变量或密钥管理服务。
   > 典型的无服务器 RAG 架构包含一个 API 网关触发一个函数，该函数与外部向量存储和 LLM 服务进行交互。
3. **长任务的异步处理：**

   - \*\*模式：\*\*API Gateway -> 初始函数（启动任务）-> 队列/调度器 -> 工作函数 -> 通知/存储
   - \*\*描述：\*\*无服务器函数有执行时间限制（例如 AWS Lambda 为 15 分钟）。对于可能超出这些限制的复杂代理交互或长时间的链执行，异步模式是必需的。初始函数接收请求，验证请求，并将消息放入队列（如 AWS SQS）或启动状态机执行（如 AWS Step Functions）。一个单独的工作函数（或状态机中的多个步骤）接取任务，执行 LangChain 处理（可能涉及多次 LLM 调用或工具使用），并存储结果（例如在数据库或 S3 存储桶中）。用户可能通过 WebSocket、电子邮件或轮询在任务完成后收到通知。
   - \*\*使用场景：\*\*复杂的报告生成、多步骤代理任务、使用 LangChain 批量处理文档。
   - \*\*考量：\*\*增加架构复杂程度。需要跟踪作业状态和交付结果的机制。步骤之间的状态管理需要仔细设计（例如通过调度器负载传递中间结果或使用外部存储）。
4. **使用外部存储的有状态对话：**

   - \*\*模式：\*\*API Gateway -> 无服务器函数（加载/保存状态）-> 带历史记录的 LangChain 链/代理 -> 外部状态存储（例如 DynamoDB, Redis）
   - \*\*描述：\*\*由于无服务器函数通常在调用之间是无状态的，管理对话历史需要外部持久化层。在执行 LangChain 逻辑之前，函数从数据库（如 DynamoDB 或 Redis）加载相关对话状态（例如使用请求中的会话 ID）。在 LLM 交互后，更新后的对话历史（通过 LangChain 的消息历史集成或 `RunnableWithMessageHistory` 包装器进行管理）会保存回去。
   - \*\*使用场景：\*\*需要多轮记忆的聊天机器人、以及需要在一个会话中回忆过去交互的代理。
   - \*\*考量：\*\*每轮对话都会增加状态存储的读/写延迟。需要仔细设计状态模式和会话管理。需要考虑与状态存储相关的成本。在高并发场景中如果处理不当，可能出现竞态条件。

### 遇到的问题和应对策略

- \*\*冷启动：\*\*函数在空闲一段时间后被调用时产生的延迟。
  - \*\*应对：\*\*使用预置并发（付费保持实例预热），优化函数包大小和初始化代码，使用启动更快的语言/运行时（尽管 Python 的冷启动通常可接受），构建应用程序以容许偶尔的延迟峰值。
- \*\*执行时间限制：\*\*单个函数调用可运行的最长时间。
  - \*\*应对：\*\*为异步处理模式（队列、状态机）设计，将复杂的任务分解为更小的函数调用，优化 LLM 调用和工具交互以提高速度。
- \*\*包大小限制：\*\*部署包（代码 + 依赖项）大小的限制。LangChain、机器学习 (machine learning)库（如 sentence-transformers）及其依赖项可能很大。
  - \*\*应对：\*\*使用 AWS Lambda 层或 Google Cloud Functions 层等平台功能来分离依赖项，审慎裁剪未使用的库，使用通常允许更大尺寸的容器镜像支持，如果可行，动态加载特定组件。
- \*\*状态管理：\*\*函数本身是无状态的。
  - \*\*应对：\*\*在请求/响应中显式传递状态（仅适用于非常简单的情形），使用外部数据库（DynamoDB, Firestore, Redis），使用托管内存服务，或集成向量 (vector)存储以实现持久的 RAG 上下文 (context)。
- \*\*VPC 网络：\*\*从无服务器函数访问虚拟私有云（VPC）内的资源（如数据库或私有 API）有时会增加网络配置的复杂程度，并可能因网络接口配置而增加冷启动时间。
  - \*\*应对：\*\*理解特定于平台的 VPC 网络配置，在合适且安全的情况下使用带公共端点的托管服务，如果需要，使用 VPC 端点。

## 参考资料

- [LangChain Documentation](https://python.langchain.com/docs/) — LangChain Community (2024)
  LangChain 官方技术资源，涵盖构建 LangChain 应用的核心概念，如链、代理、LCEL、记忆和各种集成，对文中所述的部署模式至关重要。
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/latest/dg/welcome.html) — Amazon Web Services (2024)
  AWS Lambda 的官方指南，详细介绍了调用模型、执行环境、冷启动、包部署、层和 VPC 网络配置，直接解决了部署中的挑战。
- [The Serverless Way: The Modern Guide to Serverless Computing](https://www.oreilly.com/library/view/the-serverless-way/9781098108036/) — Jeremy Edmiston (2022)
  Publisher: O'Reilly Media
  本书提供了对无服务器计算的广阔视野，包括常见的架构模式、状态管理策略以及与部署 LangChain 应用相关的操作考量。
