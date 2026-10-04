---
course: "mlops-for-large-models-llmops"
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance"
sourceId: 693
chapter: "llm-monitoring-observability-maintenance"
title: "监控、可观测性与维护"
order: 5
description: "为生产环境中的大型语言模型设计和实施可靠的监控与维护策略。"
hasQuiz: false
---

部署大型语言模型是重要一步，但运营工作在首次上线后仍需长期进行。保持性能、管理成本并确保生产环境中的大型语言模型（LLM）可靠性，需要专门的监控、可观测性和维护策略，这些策略应适应其独有特点。大型语言模型的规模、成本和特有故障模式，决定了它们所需的监控方法不同于标准应用监控。

本章着重介绍保持生产环境中大型语言模型健康且高效所需的实践。您将学习如何：

*   定义并追踪大型语言模型特有的性能指标，例如推理延迟和令牌吞吐量。
*   监控底层基础设施，密切关注GPU利用率和内存消耗。
*   部署系统以追踪与大型语言模型训练和服务相关的高昂运营成本。
*   检测语言模型特有的数据漂移和语义漂移。
*   监控大型语言模型输出质量，包括发现幻觉或不期望出现的内容模式等问题的技术方法。
*   建立反馈循环以收集数据，用于模型改进和再训练触发条件的设定。
*   在LLMOps环境中有效地使用日志记录和可观测性工具。

我们将介绍所需的工具和技术，帮助您了解模型行为，主动处理问题，并确保其持续产生价值。
