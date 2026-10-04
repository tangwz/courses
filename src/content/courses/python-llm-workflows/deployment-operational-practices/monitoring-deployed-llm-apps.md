---
course: "python-llm-workflows"
chapter: "deployment-operational-practices"
lesson: "monitoring-deployed-llm-apps"
sourceId: 5223
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-10-deployment-operational-practices/monitoring-deployed-llm-apps"
title: "监控已部署的应用程序"
description: "用于监控已部署LLM应用性能、成本和健康状况的工具和技术。"
order: 5
plots: ["plots/5223-0.json"]
sourceHash: "91a31117b1122181648b17cb1fa632bf18223ca0566ffd8abdf3d1e0b5a43b41"
sourceCorrections: []
---

您的LLM应用程序部署并可访问后，工作并未结束。为确保其稳定运行、表现良好并控制预算，需要持续观察。监控已部署的应用程序，特别是涉及LLM的应用程序，面临独特的挑战，因为它们固有的非确定性、可能的高运营成本以及“高质量”输出的主观性。有效的监控提供所需的可见性，以维护应用程序的健康状态、优化性能和管理开支。

### LLM应用程序的监控方面

为全面了解已部署的应用程序，请关注跟踪以下几类指标：

1. **性能指标：** 这些指标直接影响用户体验。

   - **延迟：** 应用程序处理请求并返回响应需要多长时间？跟踪平均、中位数和百分位延迟（例如P95、P99），以了解典型和最差情况下的性能。高延迟会使用户感到沮丧，并表明存在瓶颈。
   - **吞吐量 (throughput)：** 应用程序每单位时间（例如每秒请求数）可以处理多少请求？这对于了解容量和扩展需求很重要。
2. **资源利用率：** 监控支持应用程序的基础设施。

   - **CPU/内存使用情况：** 跟踪应用程序容器或虚拟机对计算资源的消耗。高使用率可能需要扩容或代码优化。
   - **GPU使用情况（如果适用）：** 如果您自行托管模型或使用GPU加速任务，请监控GPU利用率和内存，以确保高效使用。
   - **网络I/O：** 监控数据传输速率，特别是当您的应用程序涉及大量数据上传/下载或频繁的外部API调用时。
3. **成本监控：** LLM API及其运行的基础设施可能产生显著成本。

   - **LLM API成本：** 跟踪每次调用提供商API（如OpenAI、Anthropic等）的token使用情况（输入和输出token）。如果可能，将成本与特定功能或用户操作关联起来。大多数提供商提供仪表板，但将成本跟踪集成到您的监控系统可以提供统一的视图。
   - **基础设施成本：** 监控来自您的云提供商的计算实例、无服务器功能、数据库、存储和网络流量的相关成本。
4. **应用程序特定和质量指标：** 这些是根据您的LLM应用程序的功能和行为定制的。

   - **错误率：** 跟踪不同类型错误的频率：来自API端点的HTTP错误、LLM API错误（例如，速率限制、内容过滤）、数据解析错误、代理系统中工具执行失败，或特定链组件中的失败。
   - **LLM响应质量：** 这通常是最难自动化但很重要的。根据您的应用程序，您可能跟踪：
     - *相关性：* 对于RAG系统，检索到的文档或最终答案与查询的相关程度如何？
     - *幻觉 (hallucination)率：* LLM生成事实不准确或无意义信息的频率如何？这通常需要定期的人工评估或根据真实数据进行自动化检查。
     - *格式依从性：* 如果您期望结构化输出（例如JSON），LLM未能生成正确格式的频率如何？
     - *有害性/安全性：* 监控响应是否违反安全准则。
   - **用户反馈：** 如果可能，收集关于响应质量的明确（例如点赞/点踩）或隐含（例如用户点击重试）反馈。

### 监控工具和技术

通常需要结合多种工具和技术才能进行有效监控：

- **结构化日志：** 在整个应用程序中实现全面的日志记录。记录重要事件：传入请求、传出LLM API调用（包括提示，减去敏感数据）、接收到的响应、检索到的上下文 (context)（针对RAG）、代理做出的决策、遇到的错误以及时间信息。使用JSON等结构化格式，以便下游系统更轻松地解析和分析。Python内置的`logging`模块可以为此进行配置。
- **应用性能监控（APM）：** APM工具能够提供对应用程序性能的详细可见性。它们会自动对您的代码进行插桩（对于FastAPI或Flask等常用框架通常只需最少配置），以追踪请求流经不同组件的情况、测量数据库查询时间、跟踪外部API调用并收集系统指标。例子包括Datadog、New Relic、Dynatrace，以及供应商中立的OpenTelemetry标准，该标准提供用于生成遥测数据（追踪、指标、日志）的库和规范。
- **LLM可观测性平台：** 这是一类不断增长的工具，专门为监控LLM应用程序而设计。LangSmith（来自LangChain）、Weights & Biases (W&B Prompts)、TruLens或Arize AI等平台提供针对LLM工作流的功能：

  - 追踪复杂的LangChain/LlamaIndex序列（链、代理）。
  - 记录提示、响应和中间结果。
  - 计算并跟踪每次LLM调用的成本。
  - 将评估指标（例如，相关性、忠实度）集成到追踪中。
  - 调试特定失败或质量不佳的交互。
- **云提供商监控工具：** 运用您的云提供商提供的监控服务（例如AWS CloudWatch、Google Cloud Monitoring、Azure Monitor）。这些工具对于跟踪基础设施层面的指标（CPU、内存、网络）、收集日志、设置基本仪表板以及根据支出阈值配置成本警报非常有用。

### 实施有效监控

仅仅收集数据是不够的；您需要使其可操作。

- **仪表板：** 在仪表板上可视化最重要的指标。这使您能够快速评估应用程序的健康状况并发现趋势或异常。一个好的仪表板可能会显示：

  - 过去24小时的P95延迟。
  - 按类型划分的错误率细目。
  - 每日/每周LLM API成本趋势。
  - 资源利用水平。
  - 质量指标（如果可用）。

  
  
  ![API请求延迟 (P95) 随时间变化](plots/5223-0.json)
  
  

  > API请求在一周内的P95延迟，显示周中可能存在性能问题。
- **告警：** 配置告警以便在突破重要阈值时主动通知您。例子包括：

  - 10分钟内错误率超过5%。
  - P99延迟超过3秒。
  - LLM API成本相对于基线突然激增。
  - CPU利用率持续高于85%。
  - 检测到大量响应被标记 (token)为违反安全规定。
- **反馈回路：** 监控数据对于迭代改进价值巨大。使用监控获得的洞察来：

  - 识别并修复性能瓶颈。
  - 根据质量指标或用户反馈优化提示或RAG策略。
  - 调整资源分配以优化成本。
  - 根据错误率优先处理bug修复。

监控不是一次性设置，而是一个持续过程。随着您的应用程序演进、模型更新或使用模式改变，您的监控策略必须进行调整。持续观察是生产环境中运行可靠、高效和高质量LLM应用程序的根本。

## 参考资料

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — OpenTelemetry Authors (2025)
  OpenTelemetry 标准的官方文档，涵盖应用程序可观测性的追踪、指标和日志。
- [LangSmith Documentation](https://docs.smith.langchain.com/) — LangChain Team (2024)
  LangSmith 的官方文档，一个用于追踪、调试和评估 LLM 应用程序的 LLM 可观测性平台。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew T. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出了一个语言模型全面评估框架，对于在生产环境中定义和跟踪质量指标有帮助。
