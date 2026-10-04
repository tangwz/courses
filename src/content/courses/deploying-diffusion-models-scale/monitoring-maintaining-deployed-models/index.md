---
course: "deploying-diffusion-models-scale"
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-5-monitoring-maintaining-deployed-models"
sourceId: 1002
chapter: "monitoring-maintaining-deployed-models"
title: "监控和维护已部署模型"
order: 5
description: "实践MLOps方法，监控生产环境中扩散模型的性能、成本和质量。"
hasQuiz: false
---

成功部署扩散模型是第一步；在生产环境中保持其健康和效率是一个持续过程。本章将重点转向模型上线后的运行管理。

你将学习有效监控已部署扩散模型的必要方法。我们将研究如何识别和跟踪重要的性能指标，包括生成延迟 ($L_{gen}$)、请求吞吐量 ($T_{req}$)、错误率以及GPU使用率 ($U_{gpu}$) 等硬件使用情况。我们将涉及设置全面的日志和追踪来诊断问题，使用常见的监控工具和平台，以及建立检测性能退化或输出质量变化的方法。此外，还将讨论管理基础设施成本和使用CI/CD原则实施安全模型更新流程的策略。本章结束后，你将明白如何使你的规模化扩散模型部署平稳可靠地运行。
