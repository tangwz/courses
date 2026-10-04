---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-29-serving-llms-at-scale"
sourceId: 1107
chapter: "serving-llms-at-scale"
title: "大规模部署大型语言模型"
order: 29
description: "应对涉及可靠高效地部署和提供大型语言模型服务的基础设施和软件难题。"
hasQuiz: false
---

在大型语言模型训练、对齐和优化后，接下来的工程任务是有效部署它，以便应用程序能使用其能力。部署拥有数十亿参数的模型会带来与硬件使用率、延迟、吞吐量和运行稳定性相关的特定难题。仅仅拥有一个训练好的模型文件是不够的；需要专门的基础设施和一套方法来高效处理实际使用情况。

本章将讨论在生产环境中部署大型语言模型的实际事宜。您将学习如何：

*   为与模型交互设计合适的应用程序编程接口（API）。
*   使用NVIDIA Triton推理服务器和TorchServe等专用模型部署框架来管理部署。
*   实施处理并发用户请求和优化资源使用的方法。
*   在多个模型副本或硬件加速器之间进行推理请求负载均衡的方法。
*   监控已部署大型语言模型的主要指标，例如延迟、吞吐量、成本和系统健康状况的方法。

我们将考察构建可靠且可扩展的服务系统所需的软件和基础设施组件，这些系统能够应对大型语言模型的需求。
