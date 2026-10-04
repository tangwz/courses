---
course: "data-governance-quality-observability-production"
chapter: "automated-reliability-ci-pipelines"
lesson: "alerting-incident-management"
sourceId: 8118
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-5-automated-reliability-ci-pipelines/alerting-incident-management"
title: "告警与事件管理"
description: "有效的告警策略，旨在减少噪音并提升响应时间。"
order: 4
plots: ["plots/8118-0.json"]
sourceHash: "43f5e0bcb5bb825e3b97809c085cf6177a7c433875d497740a9ccff4bdd03146"
sourceCorrections: []
---

自动化测试和断路器为您的数据流水线提供了技术保障。然而，停止一条故障流水线只是问题的一半。另一半则是有效通知工程团队，以便他们解决问题。如果没有有效的告警策略，一条停滞的流水线就只会变成无声的故障。

本节主要说明如何将流水线信号转化为可操作的通知。我们将改变“事无巨细皆告警”的思路，这种思路会导致疲劳，转而采用基于服务等级目标（SLO）和错误预算的智能事件管理框架。

### 通知层级

数据工程中一个普遍的错误是将每个失败的测试或异常都视为紧急情况。当工程师每天收到数百条通知时，他们不可避免地会停止关注。这种被称为告警疲劳的现象，是导致长时间中断的主要原因之一。为了预防这种情况，我们将信号分为三个不同的类别。

1. **日志：** 这些是事件记录。它们用于历史分析和调试，但不会触发通知。成功的任务完成和次要警告都属于此类。
2. **工单：** 这些是非紧急问题，最终需要人工介入，但并非立即。例子包括存储成本略有增加，或不影响下游用户的次要数据质量警告。这类问题应生成Jira或Asana任务。
3. **即时通知（Pages）：** 这些是需要立即关注以防止业务影响的严重事件。即时通知会在凌晨2点叫醒工程师。只有违反SLO或完全停止数据摄取（ingestion）的故障才需要这种程度的紧急处理。

实现这种层级结构需要在您的基础设施中设置一个路由层。您的测试框架检测到错误，但一个独立的逻辑会根据严重性来确定该错误的去向。

> 数据可靠性信号流。错误在被路由到适当的渠道之前会进行严重性评估，以确保严重事件与信息噪音隔离。

### 使用SLO定义告警逻辑

为了以编程方式确定严重性，我们依赖于服务等级目标（SLO）。SLO是服务等级指标的目标值，例如“99.9%的数据必须在上午9点前可用。”

SLO的倒数是**错误预算**。如果您的SLO是99.9%，那么您的错误预算就是0.1%。只要您的故障在此预算范围内，就不会触发告警。这种方法可以容忍不影响用户体验的轻微、临时性问题。

但是，当预算消耗过快时，我们必须发出告警。这通过**消耗率**来衡量。消耗率表示错误预算相对于时间窗口的消耗速度。

错误预算消耗率的公式是：

$\text{消耗率} = \frac{\text{错误率}}{1 - \text{SLO}}$

如果消耗率为1，您正在以恰好允许的速度消耗预算。如果消耗率为10，您消耗预算的速度是允许的十倍，这意味着您将在几天内用尽每月额度。

有效的告警策略只在消耗率在持续时间内超过特定阈值时触发即时通知。这可以防止针对单行异常的告警，同时捕获系统性故障。

### 在代码中实现告警路由

在基于Python的生态系统中，您可以实现一个路由函数，作为数据测试和通知服务之间的中间件。此函数接受一个异常或测试结果对象，并根据元数据标签进行分派。

设想一种情况，我们使用自定义异常类来携带严重性元数据。

```python
class DataQualityException(Exception):
    def __init__(self, message, severity):
        self.message = message
        # 严重级别：'信息', '警告', '严重'
        self.severity = severity
        super().__init__(message)

def handle_alert(exception, context):
    """
    根据严重性将告警路由到正确的渠道。
    """
    payload = {
        "text": f"流水线故障: {exception.message}",
        "source": context['pipeline_id'],
        "timestamp": context['execution_time']
    }

    if exception.severity == 'CRITICAL':
        # 高紧急度：触发PagerDuty API
        send_to_pagerduty(payload)
        # 也记录到控制台以便追溯
        print(f"[严重] {payload}")
        
    elif exception.severity == 'WARNING':
        # 中紧急度：发送到Slack渠道
        send_to_slack(payload)
        
    else:
        # 低紧急度：仅记录
        logging.info(f"质量检查失败: {exception.message}")

def send_to_pagerduty(payload):
    # PagerDuty API调用的模拟实现
    # requests.post('https://events.pagerduty.com/v2/enqueue', json=payload)
    pass
```

这种简单的模式将测试的定义与告警基础设施解耦。一个测试只需声明“我出故障了，而且这很严重”。处理程序则决定*如何*传达这种严重性。

### 可视化事件严重性

在配置您的可观测性平台（如Grafana、Datadog或Monte Carlo）时，可视化缺陷数量与告警紧急程度之间的关系很有帮助。并非所有数据质量问题都意味着流水线中断。

下表展示了轻微数据问题的持续背景噪音（应记录）与突破事件阈值的峰值之间的区别。



![错误率与告警阈值对比](plots/8118-0.json)



> 峰值检测触发。系统会忽略低级别噪音（左侧），但在错误计数突破定义阈值（中间）时触发事件，从而识别出真正的异常。

### 事件管理生命周期

一旦告警被触发并路由给工程师，事件管理流程就开始了。这是一个结构化的工作流程，旨在解决问题并防止再次发生。它包含四个阶段：

1. **检测：** 自动化系统识别异常。我们已在之前的章节中通过可观测性监控器广泛介绍过这一点。
2. **响应：** 值班工程师确认告警。这里的首要目标是**分类处理**。工程师判断这是否为误报或实际问题。如果有效，他们评估影响范围（哪些下游仪表盘或机器学习 (machine learning)模型受到影响？）。
3. **补救：** 重点转向解决眼前问题。在数据工程中，补救通常包括恢复最近的代码更改（回滚）或重新加载数据分区（回填）。目标是恢复服务，而不是立即解决根本原因。
4. **分析（事后总结）：：** 故障排除后，团队会进行根本原因分析（RCA）。这是最有价值的步骤。团队会分析为什么不合格数据进入系统，以及为什么预提交钩子或预发布测试未能发现它。

分析阶段的成果应始终是一个新的自动化测试或更严格的断路器。这创建了一个反馈循环，使每次事件都让平台更具弹性。

### 管理告警衰减

随着时间的推移，告警规则往往会衰减。为拥有100万行的数据集设置的阈值，当数据集增长到1000万行时可能过于敏感。这会导致误报。

为了保持健康的告警文化，您必须定期审查您的告警定义。

- **抑制抖动告警：** 如果一个监控器每小时多次触发又自行解决（抖动），请立即抑制它。它没有任何价值，反而会分散对实际问题的注意力。
- **动态阈值：** 在可能的情况下，用基于标准差或历史趋势的动态阈值（例如，“行数 < 平均值下方3个标准差”）取代静态阈值（例如，“行数 < 1000”）。
- **分组告警：** 配置您的告警平台以分组相关告警。如果一个中心维度表失败，50个下游表可能也会失败。您需要的是一条通知，说明“中心表失败 + 50个受影响”，而不是51条单独的通知。

通过严格管理告警质量，您可以确保当即时通知响起时，工程团队信任该信号并迅速响应。

## 参考资料

- [Site Reliability Engineering: How Google Runs Production Systems](https://learning.oreilly.com/library/view/site-reliability-engineering/9781491929114/) — Betsy Beyer, Chris Jones, Jennifer Petoff, Niall Richard Murphy (2016)
  Publisher: O'Reilly Media; Pages: 552
  涵盖SRE基本概念，包括可靠性、事件管理以及服务水平目标（SLO）和错误预算的原理。
- [The Site Reliability Workbook: Practical Ways to Implement SRE](https://books.google.com/books/about/The_Site_Reliability_Workbook.html?id=YpFsDwAAQBAJ) — Betsy Beyer, Niall Richard Murphy, David K. Rensin, Kent Kawahara, Stephen Thorne (2018)
  Publisher: O'Reilly Media
  提供实施SRE实践的具体策略和示例，包含构建有效警报系统和管理警报疲劳的指导。
- [Implementing Service Level Objectives: A Practical Guide to SLIs, SLOs, and Error Budgets](https://www.barnesandnoble.com/w/implementing-service-level-objectives-alex-hidalgo/1136709848) — Alex Hidalgo (2020)
  Publisher: O'Reilly Media; Pages: 402
  提供了关于定义和应用服务水平目标、服务水平指标和错误预算以实现稳健警报和事件预防的专业指南。
