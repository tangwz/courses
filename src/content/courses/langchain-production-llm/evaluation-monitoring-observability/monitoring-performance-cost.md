---
course: "langchain-production-llm"
chapter: "evaluation-monitoring-observability"
lesson: "monitoring-performance-cost"
sourceId: 3910
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/monitoring-performance-cost"
title: "应用性能与成本监控"
description: "生产环境中追踪延迟、令牌用量、错误率及相关成本的方法。"
order: 5
plots: ["plots/3910-0.json", "plots/3910-1.json"]
sourceHash: "e90057dfc576bf201b6c4453c3e9b8288a13505a2af75cd482940e1ad154e3a7"
sourceCorrections: []
---

一旦您的LangChain应用部署，仅仅确保它能运行是不够的。生产系统需要持续关注，以确认其达到性能预期并控制在预算范围内。LLM的非确定性及链式操作的复杂结构，使得监控应用性能和成本尤为必要。忽视这一点可能导致用户体验下降、开支飙升以及难以诊断间歇性问题。

有效的监控包括追踪特定性能指标（KPI）和资源消耗模式。我们来考查一下必要的衡量标准和方法。

### 性能指标（KPIs）

追踪正确的性能衡量标准可以帮助了解应用的响应速度和可靠性。

1. **延迟：** 这衡量了处理请求所需的时间。区分以下几类通常会有帮助：

   - **端到端延迟：** 从接收用户请求到发送最终响应的总时间。这直接影响用户体验。
   - **组件延迟：** 在LangChain应用的特定部分所花费的时间，例如单个LLM调用、数据获取步骤、工具执行或解析逻辑。识别缓慢的组件对于优化非常必要。
     高延迟可能源于LLM推理 (inference)缓慢、检索查询效率低、后处理复杂或网络延迟。LangSmith等工具会自动追踪执行流程，为链或代理调用中的每个步骤提供细致的延迟数据，使得查明瓶颈更为便捷。
2. **错误率：** 监控错误的频率和类型对于评估可靠性是基础。LangChain应用中常见的错误分类包括：

   - **LLM API错误：** 速率限制、认证问题、LLM提供商的服务器错误。
   - **工具执行错误：** 代理尝试使用工具时发生的故障（例如，API关闭、参数 (parameter)无效、意外输出）。
   - **解析错误：** 无法将LLM的输出解析成期望的格式（例如，格式错误的JSON、结构不正确）。
   - **数据获取错误：** 连接或查询向量 (vector)存储或其他数据源时出现的问题。
   - **应用逻辑错误：** 您自定义代码中封装或编排LangChain组件的错误。
     追踪错误率，通常表示为总请求的百分比，有助于识别系统性问题。分析错误类型可以指向根本原因，无论是基础设施不稳定、提示词 (prompt)脆弱还是工具实现中的错误。LangSmith通过自动标记 (token)遇到异常的运行，来便利错误追踪。
3. **吞吐量 (throughput)：** 这衡量了您的应用每单位时间（例如，每秒或每分钟请求数）能成功处理的请求数量。了解吞吐量限制对于容量规划和确保您的应用能扩展以满足需求很必要。它通常受单个请求的延迟和可用计算资源的影响。

### 成本管理与令牌用量追踪

LLM使用通常根据处理的令牌数量（包括输入和输出）定价。未受监控的应用可能导致意料之外的高成本。

1. **令牌用量：** 这通常是主要的成本驱动因素。精确追踪需要监控：

   - **输入令牌：** 发送到LLM的令牌（提示词 (prompt)、上下文 (context)、历史记录）。
   - **输出令牌：** 由LLM生成的令牌（响应）。
     大多数LLM提供商会在其API响应中返回令牌计数。LangChain与LLM的集成通常会自动捕获这些信息。LangSmith提供内置的每追踪令牌用量追踪和汇总功能，使您能够分析与特定请求、链或代理相关的成本。您可以可视化令牌消耗的长期趋势，或按不同的应用功能或用户群体划分用量。

   ```python
   # 使用OpenAI回调获取令牌用量的例子
   from langchain_openai import ChatOpenAI
   from langchain_community.callbacks import get_openai_callback
   from langchain_core.prompts import ChatPromptTemplate

   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   prompt = ChatPromptTemplate.from_messages([
       ("system", "You are a helpful assistant."),
       ("human", "{input}")
   ])
   chain = prompt | llm

   with get_openai_callback() as cb:
       response = chain.invoke({"input": "Tell me a short joke."})
       print(response.content)
       print(f"\nTotal Tokens: {cb.total_tokens}")
       print(f"Prompt Tokens: {cb.prompt_tokens}")
       print(f"Completion Tokens: {cb.completion_tokens}")
       print(f"Total Cost (USD): ${cb.total_cost:.6f}") # 需要设置定价信息

   # 启用追踪后，LangSmith会自动捕获此信息，无需显式回调。
   ```
2. **基础设施成本：** 除了直接的LLM API调用，还要考虑托管您的应用（服务器、容器、无服务器功能）、运行向量 (vector)数据库、数据存储和网络流量相关的成本。这些成本通常会随使用量和请求量增长。
3. **成本计算与归因：** 将令牌用量数据与LLM提供商的定价模型（例如，每1K输入令牌成本、每1K输出令牌成本）结合起来，您可以计算每个请求的估计成本或一段时间内的总成本。在生产环境中，一项必要工作是将成本归因于特定的应用功能、租户（在多租户应用中）或用户操作。这通常涉及在LangSmith或您的监控系统中，用相关元数据标记 (token)请求或追踪。

### 监控工具与方法

几种工具和方法有助于监控性能和成本：

- **LangSmith：** 如前所述，LangSmith专为观察LangChain应用而设计。它会自动捕获追踪，包括延迟明细、令牌计数、错误和相关元数据。其仪表盘可以可视化长期趋势，并根据各种标准（例如，性能阈值、错误存在、特定标签）过滤运行。
- **LangChain回调：** 您可以实现自定义的`CallbackHandler`，以在链/代理执行期间拦截事件（例如，`on_llm_end`、`on_chain_start`、`on_tool_error`）。这些回调可用于记录详细的性能数据、计算令牌用量或将衡量标准发送到外部监控系统，如Prometheus、Datadog或自定义数据库。
- **标准日志记录：** 使用Python内置的`logging`模块记录未被追踪自动捕获的应用级别事件、错误和警告。有效组织日志以便于解析和分析。
- **应用性能监控（APM）系统：** Datadog、Dynatrace、New Relic等工具，或Prometheus结合Grafana等开源替代方案，提供更全面的基础设施和应用监控。将LangChain应用衡量标准（通过回调或自定义日志记录）集成到这些系统中，可以获得系统健康的全貌。

### 可视化与警报

没有有效的可视化和警报，原始衡量标准用处不大。

- **仪表盘：** 创建仪表盘（在LangSmith、Grafana或其他APM工具中），以可视化KPI和长期成本趋势。这有助于识别性能退步、成本异常或渐进式性能下降。

  
  
  ![平均请求延迟（过去24小时）](plots/3910-0.json)
  
  

  > 过去24小时内以毫秒为单位测量的平均端到端请求延迟。

  
  
  ![LLM每日令牌消耗量](plots/3910-1.json)
  
  

  > 显示一周内主要LLM每日输入和输出令牌用量的堆叠条形图。
- **警报：** 根据您必要的衡量标准的预设阈值配置警报。例如：

  - 如果P95延迟超过2秒则发出警报。
  - 如果错误率超过1%则发出警报。
  - 如果预计每日成本超出特定预算则发出警报。
    警报在出现问题时主动通知相关团队，从而实现更快的响应和缓解。

持续监控性能和成本并非一次性设置，而是一个持续的流程。定期查看您的仪表盘，及时检查警报，并将监控数据与应用更新或使用模式变化关联起来。这种规范对于在生产环境中运行可靠、高效且经济的LangChain应用是基础。

## 参考资料

- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.oreilly.com/library/view/building-machine-learning-powered/9781492045113/) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  提供将机器学习模型投入生产的实践指导，包括监控模型性能、错误和系统健康的部分。
- [Monitoring & Observability](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE5Ok88XP5m8hnV6kDyudgyE9dd9AT3o64j9wMkhFhJ6791x9T0vE0y5bpF6gnd97bXMRm7oWvvK7RTXZM0Q0fu1iOB7OqVmA5GbhWFOgUZCHZO2BReKdz_oZ9wccAcbnO8vLpFoVD5LGi7wibh) — LangChain (2024)
  LangSmith 官方指南，用于在生产环境中跟踪 LangChain 应用程序的性能、错误和成本。
