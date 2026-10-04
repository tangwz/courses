---
course: "langchain-production-llm"
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-5-evaluation-monitoring-observability"
sourceId: 788
chapter: "evaluation-monitoring-observability"
title: "评估、监控与可观测性"
order: 5
description: "使用LangSmith及其他工具，为LangChain应用实施健全的评估框架、监控和可观测性。"
hasQuiz: false
---

将应用从开发环境部署到生产环境，需要重视可靠性、性能评估和持续的健康检查。构建精密的LangChain链和智能体只是过程的一部分；确保它们持续提供正确的结果、在性能预算范围内运行，并在出现问题时能够诊断，是另一个不可或缺的方面，尤其考虑到LLM输出的可变性。

本章介绍LangChain应用运行阶段所需的实践和工具。您将学习如何实施结构化的方法进行评估、观察和分析。我们会涉及：

*   使用LangSmith作为一个核心平台，用于追踪执行、调试问题和监控生产环境中的行为。
*   制定针对您应用目标、有意义的评估指标。
*   创建自动化评估流程，以系统地测试应用输出与已定义数据集的符合性。
*   借助详细的追踪信息，以理解并解决链和智能体内部的故障或意外行为。
*   监控主要运行指标，例如延迟、错误率、token使用量及相关财务成本。
*   将应用监控数据整合到常见的第三方可观测性系统中。
*   建立收集应用响应人工反馈的机制，以指导改进。

这些技术为部署上线后的LangChain应用保持可靠和高效提供了保障。
