---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-testing-monitoring-maintenance"
lesson: "monitoring-tool-performance"
sourceId: 6617
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-6-llm-tool-testing-monitoring-maintenance/monitoring-tool-performance"
title: "工具性能与可靠性监控"
description: "搭建监控系统，追踪生产环境中LLM代理工具的性能、使用情况和可靠性。"
order: 2
plots: ["plots/6617-0.json"]
sourceHash: "db294cf7fa53e2b8ba7af51b8975fe1553a01d78957dba713c5b4a03f5822b0f"
sourceCorrections: []
---

部署LLM代理工具后，工作并未完全结束。就像汽车需要定期检查才能平稳运行一样，您的工具也需要持续监控，以确保它们可靠高效地工作。没有监控，您就如同在没有清晰可见度的情况下操作；直到工具给您的代理或其用户带来重大问题，您才知道它是否正在变慢、频繁失败或被LLM错误使用。设置有效监控涉及专注于重要指标、实际执行方案以及如何解读所收集的数据。有效监控是一个持续过程，直接有助于代理功能的维护和改进。

### 工具健康指标

要了解LLM代理工具的健康状况和行为，您需要追踪特定指标。这些指标大致可分为性能、可靠性和使用情况三类。

#### 性能指标

性能指标说明您的工具运行效率如何。

- **延迟（执行时间）：** 这是工具从被调用到返回结果所花费的时间。通常会追踪平均延迟，但也会追踪百分位延迟，例如 $P_{95}$（95百分位）或 $P_{99}$（99百分位），以便了解最差情况下的表现。延迟的突然增加可能表明工具本身、其依赖项或运行环境存在潜在问题。
- **吞吐量 (throughput)：** 该指标衡量单位时间内工具被调用的次数（例如，每秒或每分钟请求数）。监控吞吐量有助于容量规划，并了解代理对每个工具的使用强度。

#### 可靠性指标

可靠性指标显示您的工具是否始终按预期运行。

- **错误率：** 导致错误的工具执行所占的百分比。区分源自工具内部逻辑的错误（例如，程序错误、无法连接数据库）和由于LLM提供的无效输入导致的错误非常重要（尽管后者也可能指向工具描述不明确或LLM理解方面的问题）。
- **成功率：** 错误率的倒数，表示成功执行工具的百分比。
- **可用性：** 对于依赖外部服务或拥有自身基础设施的工具尤其适用。这衡量了工具可运行和可访问的时间百分比。

#### 使用情况指标

使用情况指标有助于了解LLM代理如何与您的工具交互。

- **调用频率：** 某个特定工具被调用的频率是多少？这可以突出显示代理工具库中使用最多（和最少）的工具，从而指导优化和维护工作。
- **输入模式：** 分析LLM提供给工具的输入类型，可以发现LLM是否正确使用了工具，或者工具的输入架构是否需要改进。例如，是否存在常见的格式错误输入？
- **输出特性：** 检查工具生成的输出，可以帮助确保它们向LLM提供有用且格式正确的信息。

### 实施监控

设置监控包括对工具进行插桩以发出这些指标，然后收集、存储和可视化它们。

#### 插桩

插桩是指向工具中添加代码以捕获和发送监控数据的过程。对于基于Python的工具，通常可以使用装饰器或上下文 (context)管理器巧妙地实现。

以下是使用装饰器测量执行时间并记录成功或失败的Python示例：

```python
import time
import logging

# 假设logger已配置，例如：
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("ToolMetrics") 
# 仅用于演示，我们将直接打印。在实际系统中，您会使用适当的日志记录器和指标客户端。

def monitor_tool_calls(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        status = "success"
        tool_name = func.__name__
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            status = "failure"
            # 在实际系统中：logger.error(f"工具 {tool_name} 失败: {e}", exc_info=True)
            print(f"DEBUG: 工具: {tool_name}, 异常: {e}") # 占位符
            raise
        finally:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            # 在实际系统中，您会将其发送到指标系统：
            # metrics_client.timing(f"tool.{tool_name}.latency", latency_ms)
            # metrics_client.increment(f"tool.{tool_name}.{status}_count")
            print(f"DEBUG: 工具: {tool_name}, 状态: {status}, 延迟: {latency_ms:.2f}ms") # 占位符
    return wrapper

@monitor_tool_calls
def example_api_tool(query: str):
    # 模拟一次API调用
    time.sleep(0.15) 
    if query == "cause_error":
        raise ValueError("模拟API错误")
    return {"data": f"Result for '{query}'"}

# 调用示例：
# example_api_tool("search_term")
# try:
#   example_api_tool("cause_error")
# except ValueError:
#   pass # 预期如此
```

在此示例中，对 `example_api_tool` 的每次调用都会记录其延迟和状态（成功/失败）。这些数据随后会被发送到日志记录或指标收集系统。

#### 监控系统组件

典型的监控设置包括：

1. **指标收集：** 集成到您的工具或其环境中的代理或库（如Prometheus客户端库、OpenTelemetry SDKs），用于收集发出的指标。
2. **指标存储：** 一种时间序列数据库（TSDB），旨在高效存储和查询带时间戳的数据（例如，Prometheus, InfluxDB, VictoriaMetrics）。
3. **可视化与仪表盘：** 连接到TSDB的工具（如Grafana或Kibana），用于创建图表和仪表盘，让您能够查看趋势和异常。
4. **告警：** 一个系统（通常是监控套件的一部分，如Prometheus Alertmanager），当指标的预设阈值被突破时，它会触发通知。

以下图表显示了监控数据的通用流程：

> 工具监控系统中的典型数据流，从指标发出到开发人员收到通知。

### 工具行为可视化：仪表盘

仪表盘对于使监控数据一目了然非常重要。一个精心设计的仪表盘可以快速显示性能下降、错误率飙升或异常使用模式。

例如，您可能有一个仪表盘，显示关键工具随时间变化的平均延迟和错误率：



![工具性能: 'WeatherFetcher'](plots/6617-0.json)



> 这个仪表盘快照显示了工具在几个小时内的平均延迟和错误率，特别指出了05:00左右发生的性能下降事件，随后恢复正常。

仪表盘中应包含的主要元素有：

- 延迟、吞吐量 (throughput)和错误率的时间序列图表。
- 用于显示当前状态的仪表或单一统计面板。
- 列出最常用或最慢工具的表格。
- 按类型或工具划分的错误细目。

### 设置告警

告警是主动通知，当工具的行为显著偏离常态时会通知您，以便您在问题升级前进行处理。

- **基于阈值的告警：** 当指标超出预设值时触发（例如，延迟连续5分钟超过500毫秒，错误率超过5%）。
- **异常检测告警：** 更先进的系统可以学习正常模式，并在出现非预期偏差时发出告警，即使这些偏差没有超过固定阈值。

设置告警时，请考虑以下因素：

- **严重程度：** 区分严重告警（例如，工具完全不可用）和警告（例如，延迟适度增加）。
- **通知渠道：** 电子邮件、Slack、PagerDuty 或其他适合您团队工作流程的系统。
- **可操作信息：** 告警应包含足够多的上下文 (context)信息（哪个工具、什么指标、当前值与阈值的对比），以帮助快速诊断问题。

通过仔细调整阈值并确保告警确实反映了问题，避免告警疲劳。

### 监控外部API依赖

如果您的LLM代理工具封装了外部API，您的监控范围需要扩展到这些依赖项。

- 追踪对外部API本身的调用延迟，而不仅仅是您的封装器处理时间。
- 监控外部服务返回的API错误代码（例如，4xx, 5xx HTTP状态码）。
- 留意API速率限制使用情况。您的工具应优雅地处理达到速率限制的情况，并且您的监控应显示这是否频繁发生。
- 在与外部API交互的工具中实现断路器模式。监控这些断路器的状态（打开、半开、关闭），以了解外部依赖的健康状况。

### 分析监控数据

收集数据只是工作的一部分；正确解读数据才是价值所在。

- **寻找趋势：** 延迟是否在几周内逐渐增加？某个特定工具是否变得更容易出错？
- **关联数据：** 如果您看到某个工具的错误出现峰值，这是否与部署、LLM行为变化或下游服务问题相关联？
- **区分工具问题与LLM行为：** 这可能具有挑战性。例如：
  - 如果工具在面对特定类型的格式错误输入时持续失败，LLM可能难以正确格式化对该工具的请求。这可能表明需要更清晰的工具描述或为LLM提供少量示例。
  - 如果工具无论输入如何都发生内部崩溃或高延迟，问题很可能在工具本身。
    将工具的输入和输出（在下一节日志记录中涵盖）与指标一起记录下来，对于这种区分非常有帮助。

### 通过监控持续改进

监控并非一次性设置。它是一个持续过程，为改进您的工具和整个LLM代理系统提供反馈回路。

- 使用性能数据来识别瓶颈并优化慢速工具。
- 分析错误模式，使工具更具弹性并改进错误处理。
- 观察使用模式，以了解哪些工具最有价值，以及哪些工具可能被LLM使用不足或误解。

通过勤奋监控您的LLM代理工具，您可以将它们从黑盒 (black box)转变为系统中可观察的组件。这种可见性对于构建可靠、高性能和易于维护的AI应用是根本的。

## 参考资料

- [Site Reliability Engineering: How Google Runs Production Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHRKBObIJnGZvS2qUlMZF4U2BOZhCgNXTmS_YG3Abgfkyos0J2h4Zr5ZeZ0H6QCSZhcBZbb7JhS_MVcf6d1yrIrTJEkIWB-NvB46xE7sz_GCIsujvahaKMbCT-_sUPiLQk8jWqj-pjfP5QkF0j4wTb3GLpIJjWPfQT6gZp27EtazLF-nxMsuGakhCBTs9qv) — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy (2016)
  Publisher: O'Reilly Media, Incorporated.
  一本关于站点可靠性工程的基础书籍，涵盖监控、警报和维护可靠系统的重要概念，直接适用于本节讨论的原则。
- [Prometheus Documentation](https://prometheus.io/docs/) — Prometheus Authors (2024)
  Prometheus 的官方文档，一个广泛使用的开源监控系统，提供有关指标收集、时间序列存储和警报的实用指导，如本节所述。
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — OpenTelemetry Authors (2019–present)
  OpenTelemetry 的官方文档，这是一个与供应商无关的应用程序检测标准，用于生成和导出遥测数据（指标、追踪、日志），对于有效监控至关重要。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media, Inc.
  提供关于构建可靠、可扩展和可维护的分布式系统的全面见解，包括关于可靠性模式和可观察性重要性的讨论，为 LLM 代理工具的监控提供更广阔的背景。
