---
course: "python-llm-workflows"
chapter: "testing-evaluating-llm-apps"
lesson: "logging-monitoring-llm-interactions"
sourceId: 5217
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/logging-monitoring-llm-interactions"
title: "LLM 交互的日志记录与监控"
description: "记录请求、响应和性能数据的最佳实践。"
order: 6
plots: []
sourceHash: "e07d4fae60252c4ababb830e67590b900541d390a4d8625645f4fc6647477107"
sourceCorrections: []
---

鉴于大型语言模型 (LLM) 应用程序固有的变化性和复杂性，仅测试特定输入和输出不足以确保可靠性并了解长期性能。记录交互和监控系统行为成为不可或缺的做法。它们提供了调试问题、评估性能趋势、管理成本以及确保应用程序在生产环境中按预期运行所需的原始数据。与传统确定性软件不同，传统软件的日志可能主要捕获错误，而 LLM 应用程序的日志需要包含更广泛的信息，以捕捉模型交互的特点。

### 为什么要记录 LLM 交互？

有效的日志记录和监控基于以下几个原因很重要：

1. **调试和故障排除：** 当 LLM 应用程序产生意外、不正确或有问题的输出时，详细日志通常是追踪事件序列的唯一方式。这包括发送的精确提示词 (prompt)、任何检索到的上下文 (context)（在 RAG 系统中）、模型的响应以及链或代理执行路径中的中间步骤。
2. **性能分析：** 追踪延迟（响应所需时间）、令牌使用量（输入和输出）以及成功/失败率等指标有助于发现瓶颈并了解用户体验。
3. **成本管理：** LLM API 通常根据令牌使用量计费。记录每次交互的令牌数量可以进行准确的成本追踪、预算管理，并发现意外昂贵的操作。
4. **质量评估：** 记录的交互构成持续评估的数据集。分析提示和响应可以显示失败模式、需要改进提示工程 (prompt engineering)的方面，或模型幻觉 (hallucination)或偏见的例子。捕获与交互伴随的用户反馈提供了直接的质量信号。
5. **合规和审计：** 在某些应用程序中，维护交互记录可能对于合规或审计目的来说是必要的。
6. **漂移检测：** LLM 提供商可能会更新其模型。长期监控重要的性能指标和输出质量可以帮助发现由于这些更新引起的性能下降或行为变化。

### 应该记录哪些数据？

为获得全面的信息，请考虑为每次 LLM 交互或工作流执行记录以下信息：

- **时间戳：** 交互发生的时间。
- **输入提示：** 发送给 LLM 的精确最终提示词 (prompt)。
- **模型配置：** 使用的特定模型（例如 `gpt-4-turbo`、`claude-3-opus`）和重要参数 (parameter)（温度、最大令牌数）。
- **检索到的上下文 (context)（针对 RAG）：** 如果使用 RAG，记录检索到并注入到提示中的文档或文本块。
- **中间步骤：** 对于链或代理，记录每个步骤的输入/输出、工具使用情况和代理决策。
- **LLM 响应：** 从 LLM 接收到的原始输出。
- **解析/处理后的输出：** 经过任何解析或后处理步骤后的最终输出。
- **延迟：** LLM 调用所需的时间，以及可能的端到端工作流执行时间。
- **令牌数量：** 输入令牌和输出令牌的数量。
- **预估成本：** 根据令牌数量和提供商定价计算的成本。
- **错误信息：** 过程中遇到的任何异常或错误消息。
- **用户/会话 ID：** 用于关联属于同一用户或会话的交互。
- **用户反馈（可选）：** 如果可用，显式（点赞/踩，评分）或隐式反馈。
- **版本信息：** 应用程序版本、LangChain/LlamaIndex 版本、相关库版本。

### 实现日志记录

尽管可以使用 Python 内置的 `logging` 模块，但 LLM 交互的结构化特点通常从更专业的方法中获益：

1. **结构化日志记录：** 避免使用纯文本日志，改用 JSON 等结构化格式。这使得日志更容易进行程序化解析、查询和分析。每个日志条目都成为一个字典或对象，包含上述数据点的键值对。

   ```python
   import logging
   import json
   import datetime

   log_record = {
       "timestamp": datetime.datetime.utcnow().isoformat(),
       "interaction_id": "unique-interaction-123",
       "user_id": "user-abc",
       "model": "gpt-4-turbo",
       "temperature": 0.7,
       "prompt": "Summarize the following text: ...",
       "response_raw": "This is the summary...",
       "tokens_prompt": 150,
       "tokens_completion": 50,
       "latency_ms": 1234,
       "error": None
       # 添加其他相关字段：上下文、成本、反馈等。
   }

   # 使用标准日志记录与 JSON 格式化
   logging.info(json.dumps(log_record))

   # 或者，使用能更好地处理结构化日志的库
   # 例如 structlog
   ```
2. **专用 LLM 可观测性平台：** 几个平台专门用于 LLM 应用程序的日志记录和监控。像 LangSmith (来自 LangChain)、Helicone、Arize、WhyLabs 或与通用可观测性平台（如 Datadog、OpenTelemetry）集成等工具，提供专门为 LLM 工作流设计的功能：

   - 自动追踪 LangChain 链和代理。
   - 交互流程的可视化。
   - 成本和令牌追踪仪表板。
   - 用于根据预定义标准评估已记录交互的工具。
   - 与反馈机制的集成。

   这些平台通常提供 SDK，简化捕获和发送相关数据的过程。

### 监控策略

日志记录提供数据；监控将这些数据转化为可操作的信息和警报。有效监控包括：

- **仪表板：** 创建仪表板以可视化随时间变化的重要指标：
  - 交互量。
  - 平均延迟和延迟分布（p95, p99）。
  - 令牌使用量和预估成本（每次交互、每个用户、总计）。
  - 错误率。
  - 反馈得分（如适用）。
  - 特定失败模式的频率（例如，通过评估检测到的幻觉 (hallucination)）。

> LLM 应用程序通过日志记录到监控和警报系统的数据流。

- **警报：** 设置警报以应对紧急情况：
  - 错误率飙升。
  - 延迟超出定义阈值。
  - 成本或令牌使用量突然增加。
  - 在日志中检测到敏感信息（如适用）。
  - 评估分数明显下降或负面反馈增加。
- **日志分析：** 定期审查日志，特别是针对失败或标记 (token)的交互，以找出根本原因和改进方面。使用日志平台的查询功能来查找特定模式或问题。

### 安全和隐私考量

请记住，提示词 (prompt)和 LLM 响应可能包含敏感信息（个人身份信息、专有数据）。实施适当的安全措施：

- **数据脱敏/匿名化：** 如果可能，在记录前编辑敏感数据。
- **访问控制：** 确保只有授权人员才能访问日志。
- **合规性：** 遵守有关已记录数据存储和处理的相关数据隐私法规（GDPR、CCPA 等）。
- **安全存储：** 使用静态和传输加密安全存储日志。

通过系统地记录交互和监控主要指标，您可以从简单的功能测试迈向 LLM 应用程序的持续理解和改进循环。这种可观测性不仅是最佳实践；它对于构建可靠、高效和值得信赖的 AI 系统非常重要。

## 参考资料

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — OpenTelemetry Authors (2024)
  Publisher: The Cloud Native Computing Foundation
  提供用于检测、生成、收集和导出遥测数据（追踪、指标、日志）的规范和客户端库，对于在包括LLM工作流在内的任何应用程序中实现全面的可观察性至关重要。
- [LangSmith Documentation](https://docs.smith.langchain.com/) — LangChain Team (2024)
  Publisher: LangChain
  LangSmith的官方文档，该平台专为调试、监控、测试和评估LLM应用程序及代理而设计，涵盖追踪、成本跟踪和反馈集成等功能。
