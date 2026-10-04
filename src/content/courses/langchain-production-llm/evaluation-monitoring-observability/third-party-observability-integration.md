---
course: "langchain-production-llm"
chapter: "evaluation-monitoring-observability"
lesson: "third-party-observability-integration"
sourceId: 3913
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability/third-party-observability-integration"
title: "与第三方可观测性平台结合"
description: "连接LangChain应用程序监控与Datadog、Grafana或OpenTelemetry等平台。"
order: 6
plots: []
sourceHash: "e576ee94b81792afe9b6238f6e0c7d7302cafa8f0ca3baafca3258035dd9b6ca"
sourceCorrections: []
---

LangSmith提供了用于LangChain应用追踪和调试的非常有用的专用工具，但生产环境常需要与更广范围的、现有的可观测性平台结合。许多组织已在Datadog、Grafana/Prometheus、Splunk、Jaeger或Honeycomb等系统上实现标准化，以获取其整个技术栈的统一视图。将LangChain应用监控与这些平台结合，您就可以将LLM应用行为与基础设施性能、其他微服务以及业务指标关联起来，同时运用现有的告警和事件管理流程。

本节介绍如何将LangChain应用程序的运行数据（日志、指标和追踪数据）传输到这些第三方系统。

### LangChain可观测性的要点

有效的可观测性通常依赖于三种数据类型：

1. **日志：** 带有时间戳的离散事件记录。在LangChain中，这包括应用程序启动、错误、警告、特定函数调用（如工具使用），以及如果已配置并符合隐私要求，还包括可能的输入/输出。
2. **指标：** 随时间变化的数值测量数据。对于LangChain应用，重要指标包括LLM调用延迟、令牌计数（提示、完成、总数）、预估费用、检索延迟、工具调用次数、每个组件的错误率和缓存命中率。
3. **追踪：** 表示请求整个生命周期的内容，因为它会经过各种组件或服务。LangSmith在处理LangChain内部追踪方面表现出色，但传播或导出这些追踪数据可以与上游/下游服务关联。

### 日志集成

LangChain使用Python的标准`logging`库。这使得集成相对简单。您可以配置Python的日志处理器，将日志转发到您选择的平台支持的各种目的地。

常见方法包括：

- **配置日志处理器：** 使用平台特有的Python日志处理器（例如`datadog_api_client.v2.logs`、用于Splunk HEC的库，或由Fluentd或Logstash等代理监控的标准处理器，如`logging.handlers.SysLogHandler`或`logging.FileHandler`）。
- **结构化日志：** 以结构化格式（如JSON）输出日志，大多数日志聚合平台都可以轻松解析。`python-json-logger`等库可以提供帮助。

```python
# 示例：JSON日志的基本配置
import logging
import sys
from pythonjsonlogger import jsonlogger

# 获取根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 使用流处理器输出到标准输出 (可由代理收集)
logHandler = logging.StreamHandler(sys.stdout)

# 使用JSON格式化器
formatter = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)

# 添加处理器
logger.addHandler(logHandler)

# 现在，LangChain（和您的应用）使用标准日志记录器输出的日志将是JSON格式
logging.info("应用程序已启动。")

# LangChain组件日志示例
# (假设LangChain组件内部使用标准日志记录)
# try:
#     result = my_chain.invoke({"input": "some query"})
# except Exception as e:
#     logging.error("链执行失败", exc_info=True)
```

确保发送给第三方的日志中已清除敏感信息（个人身份信息、API密钥），除非平台具有经过批准的针对此类数据的特定安全处理机制。

### 导出指标

指标提供了关于性能和资源消耗的可量化 (quantization)数据。LangChain指标的集成包括对应用程序进行插桩，以收集相关数据点并导出它们。

- **插桩：** LangChain中最有效的方法是使用**回调函数**。通过创建自定义的`BaseCallbackHandler`，您可以将测量逻辑与应用程序逻辑分离。此处理器可以拦截`on_llm_start`和`on_llm_end`等事件，以计算延迟并提取令牌使用情况，而无需更改链的结构。
- **导出：** 在您的回调处理器中，使用可观测性平台提供的客户端库（例如`prometheus_client`、`datadog`、`statsd`）来发送指标。这些库通常允许您定义指标类型（计数器、量规、直方图），并将数据推送到平台或公开一个用于抓取数据的端点。

```python
# 示例：使用回调函数，通过Prometheus对LLM调用延迟进行插桩
import time
from typing import Any, Dict, List
from prometheus_client import Histogram, Counter, start_http_server
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

# 定义Prometheus指标
LLM_LATENCY = Histogram(
    'langchain_llm_latency_seconds',
    'LLM调用延迟（秒）',
    ['model']
)
TOKEN_USAGE = Counter(
    'langchain_token_usage_total',
    '令牌使用计数',
    ['model', 'type']
)

class PrometheusMetricsHandler(BaseCallbackHandler):
    """捕获Prometheus指标的回调处理器。"""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """LLM开始运行时运行。"""
        # 使用run_id安全地处理并发调用
        run_id = kwargs.get("run_id")
        self.start_times[run_id] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """LLM结束运行时运行。"""
        run_id = kwargs.get("run_id")
        start_time = self.start_times.pop(run_id, None)
        
        # 从响应中确定模型名称或使用默认值
        model = response.llm_output.get("model_name", "未知")

        if start_time:
            latency = time.time() - start_time
            LLM_LATENCY.labels(model=model).observe(latency)

        # 如果有令牌使用情况则记录
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            TOKEN_USAGE.labels(model=model, type="prompt").inc(usage.get("prompt_tokens", 0))
            TOKEN_USAGE.labels(model=model, type="completion").inc(usage.get("completion_tokens", 0))

# 启动Prometheus客户端HTTP服务器（通常在应用启动时执行一次）
# start_http_server(8000)

# 使用回调处理器初始化LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[PrometheusMetricsHandler()]
)

# 现在，LLM的使用将自动记录指标
# result = llm.invoke("解释可观测性")
```

应考虑导出的重要指标：

- 总请求延迟
- 每个组件的延迟（LLM、检索器、工具）
- 每个请求/调用的令牌计数（提示、完成、总数）
- 每个请求/调用的预估费用
- 每个组件的错误计数和错误率
- 缓存命中/未命中率
- 检索得分或相关性指标（如果适用）

### 使用OpenTelemetry集成追踪

现代追踪常依赖于OpenTelemetry (OTel)，一个用于生成和收集遥测数据的开放标准。LangChain与OpenTelemetry生态系统结合良好，通常通过自动插桩包实现。

与Jaeger、Tempo、Honeycomb或Datadog APM等平台结合通常包括：

1. **安装OTel包：** 将所需的OpenTelemetry API、SDK、导出器和插桩包添加到您的项目。

   ```bash
   pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-langchain
   ```
2. **配置导出器：** 配置OTel SDK以将追踪数据导出到您选择的后端。这通常涉及设置环境变量或以编程方式配置SDK，使其指向后端的OTel端点（通常是OTel Collector或平台的直接摄取端点）。

   - **环境变量（常见）：** 许多OTel库会根据`OTEL_EXPORTER_OTLP_ENDPOINT`、`OTEL_SERVICE_NAME`等标准环境变量自动配置。
   - **编程配置：** 在应用程序代码中明确设置追踪器提供者、处理器和导出器。
3. **启用自动插桩：** 使用插桩库，例如`opentelemetry-instrumentation-langchain`（通常由Traceloop或OpenInference等第三方提供）。这些库会自动修补LangChain的内部执行方法，为链、LLM和检索器生成Span，无需手动编写追踪器代码。

主要优势在于*分布式追踪*：看到单个请求的路径不仅在LangChain应用程序内部，还跨越其交互的其他服务（例如，初始API网关、工具调用的后续微服务）。

> 可观测性数据从LangChain应用程序流经一个可选的收集器到专用后端平台。应用程序与后端直接集成也是可行的。

### 平台与方法的选择

可观测性平台的选择通常视您组织内现有的工具而定。然而，请考虑：

- **兼容性：** 该平台是否轻松支持Python应用程序和OpenTelemetry？
- **功能：** 评估日志搜索、指标仪表板/告警以及追踪可视化/分析的功能。它如何处理LLM应用中常见的高基数数据（例如，不同的提示、用户ID）？
- **成本：** 可观测性数据可能非常庞大。了解定价模型（按摄取GB、按主机、按追踪计费），并采用采样策略（头部采样、尾部采样）来管理追踪数据成本，如果需要，还可能应用于指标/日志。
- **关联性：** 确保平台能轻松地使用共享标识符（如`trace_id`）关联日志、指标和追踪数据。
- **安全性：** 再次强调防止敏感数据通过遥测泄露的重要性。使用掩码、过滤，或确保平台提供足够的安全控制。

将LangChain应用程序监控与您组织的标准可观测性栈结合，可以在更大系统背景下全面认识其行为。它借助在工具和专业知识方面的现有投入，从而实现为您的生产LLM应用程序提供更快的故障排除、性能优化和更可靠的运行。

## 参考资料

- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — OpenTelemetry Authors (2024)
  理解OpenTelemetry原理、API、SDK和导出器的权威来源，涵盖各种编程语言（包括Python）的跟踪、指标和日志。
- [Observability Engineering: Achieving Production Excellence](https://www.oreilly.com/library/view/observability-engineering/9781492076438/) — Charity Majors, Liz Fong-Jones, and George Miranda (2022)
  Publisher: O'Reilly Media
  关于构建和维护可观测系统的指南，涵盖日志、指标和跟踪，以及现代分布式应用程序所需的工程实践。
- [Prometheus Documentation](https://prometheus.io/docs/) — Prometheus Authors (2024)
  Publisher: Cloud Native Computing Foundation
  Prometheus监控系统的官方文档，详细介绍了如何仪表化应用程序、暴露指标以及配置Prometheus进行抓取和告警。
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html) — Vinay Sajip (2024)
  Python官方关于其标准`logging`模块的指南，用于理解如何配置记录器、处理器、格式化器和过滤器以实现有效的应用程序日志记录。
