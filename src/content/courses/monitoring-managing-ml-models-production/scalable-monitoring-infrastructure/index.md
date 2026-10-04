---
course: "monitoring-managing-ml-models-production"
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-5-scalable-monitoring-infrastructure"
sourceId: 706
chapter: "scalable-monitoring-infrastructure"
title: "可扩展监控的基础设施和工具"
order: 5
description: "了解构建可靠和可扩展的ML监控解决方案所需的工具和基础设施设计。"
hasQuiz: false
---

在生产环境中监控机器学习模型会产生大量数据，并需要能在负载下稳定运行的系统。在明确了*要监控什么*之后（从数据漂移到性能下降），现在将重点转向建设和管理必要的基础设施，以便高效地支持这些大规模监控活动的实际操作层面。

本章将讨论相关的工程挑战。您将学习到：

*   记录数据量大的环境中预测数据和监控输出的策略。
*   使用时序数据库（$TSDBs$），它们专门设计用于高效处理带有时间戳的指标数据。
*   设计可以水平扩展的监控管道分布式系统架构。
*   将您的监控组件与Kubeflow、MLflow和SageMaker等常见MLOps平台集成。
*   专门用于ML监控的开源和商业工具概览。
*   创建有用的仪表盘和设置有意义的警报，以便及时了解模型健康状况的方法。

我们将探讨如何选择和配置这些组件，以构建一个符合生产机器学习需求的监控系统。
