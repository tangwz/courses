# 动手实践：构建 RAG 监控仪表盘

来源：[原文](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/hands-on-rag-monitoring-dashboard)

[返回章节目录](README.md) · [返回课程目录](../README.md)

本实践指南将带您逐步了解如何为您的生产环境 RAG 系统实施一个量身定制的监控仪表盘。一个设计得当的仪表盘能让您一目了然地看到系统运行状况、性能趋势和质量指标，从而快速发现问题，并为持续优化提供有依据的决策。我们假设您已具备收集日志和指标的机制；我们侧重于 *收集哪些* RAG 特定数据以及 *如何* 有效地将它们可视化。

### 您的仪表盘所需的核心 RAG 指标

在构建任何可视化内容之前，您必须确定真正反映 RAG 系统运行状态和效用的指标。这些指标通常分为几类，基于本章前面讨论的评估指标：

1. **检索表现：**

   - **延迟：** 检索步骤的 p50、p90、p99 延迟。对用户体验非常重要。
   - **命中率/召回率：** 找到相关文档的查询百分比（可以估算或基于离线评估）。
   - **向量 (vector)一致性：** 监测向量空间中的漂移，例如，通过跟踪查询向量与其排名前 k 的检索文档向量之间随时间变化的平均距离。
   - **向量数据库健康状况：** 查询错误率、索引大小、活跃连接数。
2. **生成表现：**

   - **LLM 延迟：** 生成模型 p50、p90、p99 的延迟。
   - **令牌消耗：** 每次请求的平均输入和输出令牌数，使用的总令牌数（对成本很重要）。
   - **质量得分：** 跟踪如忠实度、答案相关性等指标（可能来自定期的 RAGAS/ARES 运行或人工反馈循环）。
   - **幻觉 (hallucination)率：** 包含未经证实或错误信息的响应百分比。
   - **护栏指标：** 内容安全触发频率、被过滤的请求。
3. **端到端系统指标：**

   - **整体延迟：** 从用户查询到最终响应的 p50、p90、p99 延迟。
   - **吞吐量 (throughput)：** 每秒请求数 (RPS) 或每分钟请求数 (RPM)。
   - **错误率：** 整体系统错误率，以及特定组件（检索器、生成器、API 集成）的错误率。
   - **资源使用情况：** 核心服务的 CPU、GPU、内存使用量。
4. **用户交互与反馈：**

   - **用户评分：** 来自点赞/点踩、星级评分的平均分数。
   - **反馈类别：** 报告问题的计数（例如，“不相关”、“有害”、“过时”）。
   - **会话时长/参与度：** 如果适用，用户如何与 RAG 输出交互。

### 为您的 RAG 管道植入检测点

为了填充您的仪表盘数据，您的 RAG 应用程序必须发出这些指标。这需要在您的代码中添加日志记录和指标收集点。尽量使用结构化日志或直接将指标发送到时间序列数据库（如 Prometheus）或集中式日志系统（如 ELK 堆栈）。

考虑一个基于 Python 的 RAG 管道。您可以像这样添加检测：

```python
import time
import logging
from PrometheusClient import Counter, Histogram # 虚构的 Prometheus 客户端

# 配置基本日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RAG_Pipeline')

# 定义 Prometheus 指标（示例）
rag_requests_total = Counter('rag_requests_total', 'Total number of RAG requests processed', ['pipeline_stage'])
rag_request_latency = Histogram('rag_request_latency_seconds', 'Latency of RAG requests', ['pipeline_stage'])
retrieval_recall = Histogram('rag_retrieval_recall', 'Recall scores for retrieval', buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
llm_hallucination_rate_gauge = Gauge('rag_llm_hallucination_rate_percent', 'Current estimated LLM hallucination rate')

def retrieve_documents(query: str) -> list:
    rag_requests_total.labels(pipeline_stage='retrieval_input').inc()
    start_time = time.monotonic()
    try:
        # 模拟检索
        logger.info(f"Retrieving documents for query: {query[:30]}...")
        retrieved_docs = [{"id": "doc1", "content": "Relevant content..."}] 
        # 在实际系统中，如果可能，您会计算实际召回率
        # 例如，如果您有办法检查真实数据是否在检索到的文档中
        # recall_score = calculate_recall(retrieved_docs, ground_truth_for_query)
        # retrieval_recall.observe(recall_score)

        latency = time.monotonic() - start_time
        rag_request_latency.labels(pipeline_stage='retrieval').observe(latency)
        logger.info(f"Retrieval completed in {latency:.4f} seconds.")
        rag_requests_total.labels(pipeline_stage='retrieval_output').inc()
        return retrieved_docs
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        rag_requests_total.labels(pipeline_stage='retrieval_error').inc()
        raise

def generate_answer(query: str, context_docs: list) -> str:
    rag_requests_total.labels(pipeline_stage='generation_input').inc()
    start_time = time.monotonic()
    try:
        # 模拟生成
        logger.info(f"Generating answer for query: {query[:30]}...")
        answer = "This is a generated answer based on the context."
        # 您可能会定期根据离线评估或反馈更新幻觉率
        # 例如，如果评估作业运行并报告 5% 的幻觉：
        # llm_hallucination_rate_gauge.set(5.0)

        latency = time.monotonic() - start_time
        rag_request_latency.labels(pipeline_stage='generation').observe(latency)
        logger.info(f"Generation completed in {latency:.4f} seconds.")
        rag_requests_total.labels(pipeline_stage='generation_output').inc()
        return answer
    except Exception as e:
        logger.error(f"Generation error: {e}")
        rag_requests_total.labels(pipeline_stage='generation_error').inc()
        raise

# 端到端流程示例
def process_query(query: str):
    overall_start_time = time.monotonic()
    try:
        documents = retrieve_documents(query)
        answer = generate_answer(query, documents)
        overall_latency = time.monotonic() - overall_start_time
        rag_request_latency.labels(pipeline_stage='end_to_end').observe(overall_latency)
        logger.info(f"Query processed. Answer: {answer}, Latency: {overall_latency:.4f}s")
        return answer
    except Exception as e:
        logger.error(f"Overall query processing error: {e}")
        # 适当地处理错误
        return "An error occurred while processing your request."

# process_query("What are advanced RAG optimization techniques?")
```

这个示例使用了虚构的 Prometheus 客户端，但其原理是在每个重要步骤记录延迟、计数和其他可量化 (quantization)指标。

### 设计和填充仪表盘小部件

随着指标数据的流动，您可以开始构建您的仪表盘。Grafana、Kibana（用于 ELK 堆栈）等工具，或使用 Plotly/Dash 等库的自定义解决方案，都是常见的选择。

**通用设计原则：**

- **针对受众的视图：** 运维人员可能需要详细的资源使用情况，而产品经理可能侧重于用户满意度和质量指标。
- **逻辑分组：** 将相关指标分组（例如，“检索表现”部分，“生成质量”部分）。
- **清晰的可视化：** 时间序列数据使用折线图，比较使用柱状图，当前状态使用仪表盘，详细列表（例如，最近的错误）使用表格。
- **可操作性：** 每个小部件都应有助于回答问题或发现潜在问题。

**Plotly JSON 示例小部件：**

让我们设计几个常见的 RAG 仪表盘小部件。

1. **端到端请求延迟 (P90) - 时间序列：**

   
   
   [交互图表：端到端请求延迟 (P90)](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/hands-on-rag-monitoring-dashboard#plot-3of3va)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "x": [
           "2023-10-26 10:00",
           "2023-10-26 10:05",
           "2023-10-26 10:10",
           "2023-10-26 10:15",
           "2023-10-26 10:20"
         ],
         "y": [
           1200,
           1250,
           1100,
           1300,
           1220
         ],
         "type": "scatter",
         "mode": "lines+markers",
         "name": "P90 延迟 (毫秒)",
         "line": {
           "color": "#339af0"
         }
       }
     ],
     "layout": {
       "title": "端到端请求延迟 (P90)",
       "xaxis": {
         "title": "时间"
       },
       "yaxis": {
         "title": "延迟 (毫秒)",
         "rangemode": "tozero"
       },
       "height": 300,
       "margin": {
         "l": 50,
         "r": 20,
         "t": 40,
         "b": 40
       }
     }
   }
   ```
   
   </details>
   
   

   > 随时间变化的 P90 端到端延迟，有助于发现性能下降或提升。
2. **答案相关性得分（周平均）- 柱状图：**
   假设您有一个每周生成平均答案相关性得分的流程（例如，自动化 RAGAS 运行、人工标注）。

   
   
   [交互图表：每周平均答案相关性得分](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/hands-on-rag-monitoring-dashboard#plot-1pxk6hy)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "x": [
           "Week 40",
           "Week 41",
           "Week 42",
           "Week 43"
         ],
         "y": [
           0.85,
           0.82,
           0.88,
           0.86
         ],
         "type": "bar",
         "name": "平均答案相关性",
         "marker": {
           "color": "#20c997"
         }
       }
     ],
     "layout": {
       "title": "每周平均答案相关性得分",
       "xaxis": {
         "title": "周"
       },
       "yaxis": {
         "title": "得分 (0-1)",
         "range": [
           0,
           1
         ]
       },
       "height": 300,
       "margin": {
         "l": 50,
         "r": 20,
         "t": 40,
         "b": 40
       }
     }
   }
   ```
   
   </details>
   
   

   > 跟踪每周平均答案相关性得分能帮助了解 RAG 系统提供相关信息的能力。
3. **检索与生成延迟分解 (P95) - 堆叠柱状图或分组柱状图（分组示例）：**

   
   
   [交互图表：当前各组件 P95 延迟](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/hands-on-rag-monitoring-dashboard#plot-1b3cnjk)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "x": [
           "Component Latency P95"
         ],
         "y": [
           450
         ],
         "type": "bar",
         "name": "检索 P95 (毫秒)",
         "marker": {
           "color": "#748ffc"
         }
       },
       {
         "x": [
           "Component Latency P95"
         ],
         "y": [
           800
         ],
         "type": "bar",
         "name": "生成 P95 (毫秒)",
         "marker": {
           "color": "#ff922b"
         }
       }
     ],
     "layout": {
       "title": "当前各组件 P95 延迟",
       "barmode": "group",
       "yaxis": {
         "title": "延迟 (毫秒)"
       },
       "height": 350,
       "margin": {
         "l": 50,
         "r": 20,
         "t": 50,
         "b": 40
       }
     }
   }
   ```
   
   </details>
   
   

   > 比较检索和生成组件的 P95 延迟，以找出瓶颈。
4. **LLM 幻觉 (hallucination)指示器 - 仪表盘图：**
   许多仪表盘工具都提供仪表盘图。如果您使用 Plotly，您可以创建一个指示器：

   
   
   [交互图表：估计幻觉率 (%)](https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-6-advanced-rag-evaluation-monitoring/hands-on-rag-monitoring-dashboard#plot-16aodf6)
   
   <details>
   <summary>图表数据（JSON）</summary>
   
   ```json
   {
     "data": [
       {
         "type": "indicator",
         "mode": "gauge+number",
         "value": 7,
         "gauge": {
           "axis": {
             "range": [
               0,
               20
             ]
           },
           "bar": {
             "color": "#fa5252"
           },
           "steps": [
             {
               "range": [
                 0,
                 5
               ],
               "color": "#69db7c"
             },
             {
               "range": [
                 5,
                 10
               ],
               "color": "#ffec99"
             }
           ],
           "threshold": {
             "line": {
               "color": "red",
               "width": 4
             },
             "thickness": 0.75,
             "value": 10
           }
         },
         "domain": {
           "x": [
             0,
             1
           ],
           "y": [
             0,
             1
           ]
         }
       }
     ],
     "layout": {
       "title": "估计幻觉率 (%)",
       "height": 250,
       "margin": {
         "l": 30,
         "r": 30,
         "t": 60,
         "b": 30
       }
     }
   }
   ```
   
   </details>
   
   

   > 一个显示当前估计幻觉率的指示器，带有严重程度的颜色编码。
5. **失败查询或低评分响应的表格：**
   大多数仪表盘工具都允许您显示从日志系统中查询的表格数据。这可以显示：

   - 查询文本
   - 时间戳
   - 错误信息（如果适用）
   - 用户反馈评分（如果适用）
   - 用于调试的关联跟踪 ID

### 设置警报

仪表盘非常适合目视检查，但警报对于主动的问题管理来说是必要的。根据重要指标阈值配置警报。例如：

- **高延迟：** 端到端 P99 延迟连续 5 分钟超过 3 秒。
- **错误率升高：** 整体系统错误率在 10 分钟内超过 2%。
- **LLM API 故障：** 与 LLM API 调用相关的生成组件错误率 > 5%。
- **质量下降：** 平均答案相关性得分周环比下降 10%。
- **检测到漂移：** 基准查询的向量 (vector)分布或检索命中率出现显著变化。
- **高幻觉 (hallucination)率：** 估计幻觉率超过可接受的阈值（例如，> 8%）。

这些警报应通过 Slack、PagerDuty 或电子邮件等渠道通知适当的团队（例如，SRE、ML 工程师）。

### RAG 的高级仪表盘技术

随着您的 RAG 系统日趋成熟，考虑更复杂的仪表盘功能：

- **相关性分析：** 探究不同指标之间的关系。例如，特定数据源的检索延迟增加是否与该数据源相关查询的用户满意度评分降低相关联？
- **钻取：** 允许用户点击高层级指标（例如，整体错误率），并钻取查看每个组件的错误率或具体错误日志。
- **版本感知监控：** 如果您部署新版本的检索器模型、LLM 或提示策略，请按版本标记 (token)您的指标。这使得您的仪表盘能够比较不同版本之间的性能，这对 A/B 测试和回滚决策非常有价值。
- **成本跟踪：** 将成本数据（例如，LLM API 成本、向量 (vector)数据库成本、计算成本）直接集成到您的仪表盘中，以监控使用模式和优化的财务影响。
- **异常检测：** 实施异常检测算法，可以在您的指标中标记异常模式，即使它们没有超出预设限制。

### RAG 仪表盘中的常见反模式

在创建 RAG 监控仪表盘时，请避免以下常见错误：

- **指标过多（虚荣指标）：** 显示所有可能的指标可能导致信息过载。侧重于可操作的指标，这些指标能指示系统运行状况或质量问题。
- **缺乏背景：** 没有背景的数字（例如，“延迟是 500 毫秒”）用处不大。显示趋势、比较（例如，与上一个周期、与 SLO 比较）和分布。
- **忽视以用户为中心的指标：** 过分侧重于系统性能（延迟、吞吐量 (throughput)）而忽视用户感知的质量（相关性、事实性、有用性），可能导致系统运行很快但没有实际效用。
- **不常更新：** 对于从批处理评估（如 RAGAS 分数）获得的指标，确保它们在仪表盘上定期更新，以保持相关性。
- **未集成警报：** 仅凭仪表盘是被动的。重要问题需要主动警报。
- **静态仪表盘：** 仪表盘应不断发展。随着您的 RAG 系统变化，或随着您对其行为了解更多，您的仪表盘应更新新的指标或优化的可视化。

通过精心为您的管道植入检测点，选择有意义的 RAG 特定指标，并设计清晰、可操作的可视化，您可以创建一个监控仪表盘，它将成为维护和改进生产环境 RAG 系统的必要工具。请记住，这是一个迭代过程；根据操作经验和不断变化的系统要求持续优化您的仪表盘。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  提供关于机器学习系统监控和可观测性的全面指南，包括针对生产RAG的指标选择、仪表盘构建和警报设置的实用建议。
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation Systems](https://arxiv.org/abs/2309.15217) — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023)
  Journal: arXiv preprint arXiv:2309.15217; DOI: [10.48550/arXiv.2309.15217](https://doi.org/10.48550/arXiv.2309.15217)
  介绍了RAG系统自动化评估框架，提供了忠实度和答案相关性等指标，这些对于生产RAG仪表盘的持续质量监控至关重要。
- [Prometheus: Up & Running: Infrastructure and Application Monitoring](https://learning.oreilly.com/library/view/prometheus-up-running/9781492095258/) — Brian Brazil, Björn Rabenstein (2021)
  Publisher: O'Reilly Media
  关于使用Prometheus进行指标收集和监控的权威指南，涵盖了检测、查询语言以及与Grafana等仪表盘工具的集成，直接适用于构建RAG监控后端。

---

[上一节](08-%E6%9E%84%E5%BB%BARAG%E7%B3%BB%E7%BB%9F%E5%81%A5%E5%BA%B7%E4%BB%AA%E8%A1%A8%E7%9B%98.md) · [下一节](../07-%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E3%80%81%E5%8F%AF%E9%9D%A0%E6%80%A7%E5%92%8C%E5%8F%AF%E7%BB%B4%E6%8A%A4%E6%80%A7/01-RAG%E7%B3%BB%E7%BB%9F%E7%9A%84%E9%AB%98%E5%8F%AF%E7%94%A8%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1.md)
