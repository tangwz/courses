# LLMOps的日志记录与可观测性平台

来源：[原文](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/llm-logging-observability)

[返回章节目录](README.md) · [返回课程目录](../README.md)

有效的监控很大程度上依赖于全面的日志记录以及通过收集的数据观察系统行为的能力。尽管标准应用受益于既定的日志记录和可观测性实践，但大型语言模型引入了独特的数据类型、规模和故障模式，这需要专门的方法。仅仅收集基础设施基本指标和应用日志不足以理解生产环境中LLM的性能、成本和质量影响。

### LLM可观测性数据范围

为了获得LLM操作的有意义信息，您需要捕获远超典型应用日志的数据点。考虑对系统进行插桩以收集：

1. **请求和响应负载：**

   - **输入提示词 (prompt)：** 提供给模型的准确文本或结构化输入。*注意：* 请留意数据隐私和个人身份信息（PII）；考虑进行编辑或抽样。
   - **生成的响应：** LLM的完整输出。
   - **时间戳：** 用于请求开始、结束以及可能的重要内部阶段。
   - **延迟指标：** 总端到端延迟，首个token生成时间，以及每输出token延迟（$延迟 / 输出\_Token\_数量$）。
   - **Token数量：** 处理的输入token和输出token数量。这对于成本计算和性能分析是根本性的。
   - **请求元数据：** 用户标识符（如有必要可匿名化），会话ID，请求的模型版本，命中的API端点，以及相关的A/B测试变体。
2. **基础设施指标：**

   - **GPU/TPU 利用率：** 推理 (inference)或训练期间加速器的百分比利用率。
   - **加速器内存使用：** 每个请求或平均消耗的VRAM量。对于识别瓶颈和优化批处理非常重要。
   - **网络I/O：** 带宽使用情况，对于分布式训练/推理以及加载大型模型/数据集尤其重要。
   - **CPU和系统内存：** 主机上的使用情况，特别是用于数据预处理、后处理或编排任务。
3. **成本数据：**

   - **估计的每个请求成本：** 根据输入/输出token数量以及特定LLM API的定价模型或自托管基础设施的推断成本计算。
   - **汇总成本：** 追踪每天/每周/每月的总成本，按模型、用户组或功能细分。
4. **质量和行为信号：**

   - **内容安全分数：** 应用于提示词或响应的毒性分类器、PII检测器或其他内容过滤器的输出。
   - **幻觉 (hallucination)指标：** 从之前讨论的技术中获得的指标（例如，自洽性检查、不确定性分数、与知识库的事实核查）。
   - **相关性/实用性分数（如果适用）：** 来自自动化评估或人工反馈机制（评分、标记 (token)）的分数。
   - **工具使用信息（用于智能体）：** 调用了哪些工具，它们的输入/输出，成功/失败状态。
5. **RAG系统指标（如果适用）：**

   - **检索延迟：** 查询向量 (vector)数据库并检索相关文档所需的时间。
   - **检索到的文档ID/分数：** 哪些文档被获取以及它们的相关性分数。
   - **向量数据库性能：** 查询吞吐量 (throughput)，索引时效性，资源利用率。
6. **系统事件和错误：**

   - 应用错误（例如，超时、解析错误、无效输入）。
   - 基础设施事件（例如，自动扩缩容操作、节点故障、部署触发器）。
   - 速率限制事件。

### 选择和实施可观测性平台

选择正确的平台或工具组合对于管理这种数据涌入非常重要。考虑这些因素：

- **可扩展性：** LLM，尤其是流行的LLM，可以生成庞大的遥测数据量（日志、指标、追踪）。平台必须高效处理高吞吐量 (throughput)和大量存储需求。
- **数据处理：** 摄取、解析和索引结构化（例如JSON）和半结构化数据的能力很重要。LLM负载通常包含嵌套信息。
- **查询和关联：** 需要一种强大的查询语言来切片、分解和聚合数据。重要的是，平台应允许日志（例如特定请求日志）、指标（例如该请求期间的GPU利用率）和追踪（请求在系统中的路径）之间轻松关联。
- **可视化：** 需要灵活的仪表板功能来可视化LLM特有的指标，如token计数分布、延迟直方图、成本趋势和质量分数随时间波动。
- **集成：** 平台应与您现有技术栈顺畅集成：云提供商、容器编排器（Kubernetes）、推理 (inference)服务器（Triton, vLLM）、ML框架，以及可能专门的LLM监控工具。
- **成本：** 可观测性平台在大规模使用时可能变得昂贵。根据数据量、保留期、使用的功能和用户席位评估定价模型。

**常用平台选择：**

- **通用可观测性套件：** 像Datadog、Dynatrace、New Relic、Grafana Cloud（或自托管的Grafana、Loki、Tempo、Mimir/Prometheus）这样的平台提供广泛的功能。它们通常需要配置和自定义插桩（例如使用OpenTelemetry）才能有效捕获LLM特定的详细信息。
- **ML/LLM专用可观测性工具：** 像Arize AI、WhyLabs、Fiddler AI和TruEra这样的解决方案是专为监控机器学习 (machine learning)模型而构建的。它们通常提供预设的漂移、数据质量、性能监控器，并且越来越多地关注LLM特有的问题，如幻觉 (hallucination)检测和毒性监控。它们可以与通用可观测性平台集成或作为补充。
- **日志后端：** Elasticsearch (ELK栈)、Splunk或专用日志数据库可以处理大量数据，但与集成平台相比，构建关联和可视化层可能需要更多精力。
- **用于日志的向量 (vector)数据库：** 一种新兴方法是直接将请求/响应对或嵌入 (embedding)记录到向量数据库中。这允许对日志进行语义搜索（“查找与此问题请求相似的请求”），这对于调试复杂问题可能非常有用。

### 结构化日志记录非常重要

考虑到LLM交互的复杂性，以JSON等结构化格式记录数据非常重要。避免不透明的纯文本日志。

```python
# 使用标准Python日志记录与JSON格式化器的示例
import logging
import json_log_formatter
import time
import uuid

formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler()
json_handler.setFormatter(formatter)

logger = logging.getLogger('llm_inference')
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

def process_request(user_id, prompt, model_version):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # --- 模拟LLM调用 ---
    time.sleep(0.5) # 模拟工作
    response = "This is a generated response."
    input_tokens = len(prompt.split()) # 简化token化
    output_tokens = len(response.split()) # 简化token化
    # --- 模拟LLM调用结束 ---
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    log_extra_data = {
        'request_id': request_id,
        'user_id': user_id,
        'model_version': model_version,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'latency_ms': round(latency_ms, 2),
        'prompt_length': len(prompt),
        'response_length': len(response),
        # 在此处添加质量分数、成本估算等
    }
    # 如果需要，单独记录提示词/响应，同时考虑大小/PII
    logger.info(f"LLM request processed for user {user_id}", extra=log_extra_data)

# 使用示例
process_request("user-123", "Explain the importance of observability.", "gpt-4-turbo-2024-04-09")
```

这种结构化方法让您可以在所选平台中，根据`user_id`、`model_version`、`input_tokens`或`latency_ms`等特定字段轻松过滤、聚合和分析日志。

### 追踪路径：LLMOps中的分布式追踪

现代LLM应用通常涉及多个服务：API网关、数据预处理步骤、对外部工具或知识库（如RAG中的向量 (vector)数据库）的潜在调用、LLM推理 (inference)服务本身，以及后处理/过滤逻辑。理解性能瓶颈或故障需要追踪请求在这些组件间的路径。

分布式追踪工具（实现OpenTelemetry等标准）在服务边界间传播唯一的追踪ID。这使得平台能够重构请求的完整生命周期，并可视化在每个组件中花费的时间。

> LLM请求的分布式追踪简化图，可能涉及RAG组件。每个箭头表示服务之间的调用，标注了在*下游*服务或网络跳转中花费的时间。

可观测性平台使用追踪数据生成服务映射和详细的火焰图，突出显示对性能优化很重要的依赖关系和延迟贡献。

### 针对重要事项发出警报

随着丰富数据流入您的可观测性平台，您可以超越简单的“服务器宕机”警报。根据以下内容配置警报：

- **性能下降：** P95或P99延迟超过阈值，首个token生成时间显著增加。
- **成本异常：** 每日估计成本突然飙升，特定用户或模型的每个请求的token使用量高。
- **质量问题：** 毒性分数增加，幻觉 (hallucination)指标上升，用户反馈评分下降，标记 (token)为待审核的请求激增。
- **基础设施瓶颈：** GPU持续高利用率（>90%），GPU内存使用量接近限制，网络饱和。
- **错误率：** 推理 (inference)服务HTTP 5xx错误增加，输入验证错误增多。
- **漂移检测：** 监控工具检测到提示词 (prompt)/响应分布或概念漂移显著时触发的警报。



[交互图表：LLM 推理 P95 延迟随时间变化](https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/llm-logging-observability#plot-vbpp87)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        "2024-05-01",
        "2024-05-02",
        "2024-05-03",
        "2024-05-04",
        "2024-05-05",
        "2024-05-06",
        "2024-05-07"
      ],
      "y": [
        750,
        780,
        810,
        1500,
        1450,
        820,
        790
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "P95 延迟 (ms)",
      "line": {
        "color": "#4263eb"
      },
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        "2024-05-04"
      ],
      "y": [
        1550
      ],
      "mode": "markers+text",
      "name": "部署 v1.2",
      "text": [
        "部署 v1.2"
      ],
      "textposition": "top center",
      "marker": {
        "color": "#f03e3e",
        "size": 8,
        "symbol": "triangle-up"
      }
    }
  ],
  "layout": {
    "title": "LLM 推理 P95 延迟随时间变化",
    "xaxis": {
      "title": "日期"
    },
    "yaxis": {
      "title": "P95 延迟 (ms)",
      "rangemode": "tozero"
    },
    "showlegend": false,
    "margin": {
      "l": 50,
      "r": 20,
      "t": 40,
      "b": 40
    },
    "height": 300
  }
}
```

</details>



> P95 推理延迟显示显著的飙升，可能与新的模型部署相关。有效的可观测性能够关联此类事件。

总之，使用适当的日志记录和可观测性平台对于生产环境中管理LLM是根本。它有助于为性能调优、成本控制、质量保证以及触发微调 (fine-tuning)或再训练等维护活动提供可执行信息，最终确保LLM应用的长期健康和价值。

## 参考资料

- [Engineering MLOps: An Introduction to Operationalizing Machine Learning](https://www.oreilly.com/library/view/engineering-mlops/9781098135899/) — Carl Wallen (2023)
  Publisher: O'Reilly Media
  本书涵盖了 MLOps 的实践方面，包括生产机器学习系统的监控和可观测性模式。
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — The OpenTelemetry Authors (2025)
  OpenTelemetry 的官方文档，它是收集遥测数据的行业标准，包括用于理解复杂应用程序流程的分布式跟踪。
- [Distributed Systems Observability: A Guide to Distributed Tracing, Metrics, and Logging](https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/) — Sridharan, Cindy (2018)
  Publisher: O'Reilly Media
  本书涵盖了分布式系统可观测性（日志、指标、跟踪）的基本原则，为复杂的 LLM 部署奠定了基础。

---

[上一节](07-%E5%BB%BA%E7%AB%8B%E5%8F%8D%E9%A6%88%E5%BE%AA%E7%8E%AF%E4%BB%A5%E5%AE%9E%E7%8E%B0%E6%8C%81%E7%BB%AD%E6%94%B9%E8%BF%9B.md) · [下一节](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E8%AE%BE%E7%BD%AE%E5%9F%BA%E6%9C%AC%E7%9A%84LLM%E7%9B%91%E6%8E%A7.md)
