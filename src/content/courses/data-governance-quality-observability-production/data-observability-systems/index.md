---
course: "data-governance-quality-observability-production"
sourceUrl: "https://apxml.com/zh/courses/data-governance-quality-observability-production/chapter-3-data-observability-systems"
sourceId: 1415
chapter: "data-observability-systems"
title: "数据可观测系统"
order: 3
description: "设计数据可观测系统。在生产环境中监控数据新鲜度、数据量和架构漂移。"
hasQuiz: false
---

单元测试能有效验证已知预期，例如确保某列不含空值，或某个整数落在特定区间内。然而，断言通常无法发现系统性问题。即使数据管道从未运行，测试套件也可能通过；今天的有效架构也无法保证明天的兼容性。可观测性通过基于外部输出来持续了解系统的内部状态，从而弥补了这些不足。

本章将从静态测试转向动态监控。我们将确立数据管道健康状况追踪所需的技术支柱：日志、指标和追踪。您将学习如何为数据工作流添加监控，以检测那些经常绕过标准质量关卡的“静默故障”。

本课程将涉及针对三类主要异常情况的监控实现：

*   **新鲜度：** 衡量数据生成与其在数仓中可用之间的延迟。
*   **数据量：** 识别行数或数据大小的显著偏差。例如，当当前数据量 $V_t$ 与历史基线显著偏离时，您可能会触发警报，其正式表达为：
    $$| V_t - \mu | > k \cdot \sigma$$
    其中 $\mu$ 为移动平均值，$\sigma$ 为标准差。
*   **架构漂移：** 以编程方式识别列类型变化、字段添加或删除，这些变化可能威胁到向后兼容性。

在本模块结束时，您将能够构建一个针对这些情况发出警报的监控系统，确保在可靠性问题影响下游使用者之前被识别。
