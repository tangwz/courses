---
course: "introduction-to-mlops"
chapter: "model-deployment-and-serving"
lesson: "what-is-model-deployment"
sourceId: 7248
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-5-model-deployment-and-serving/what-is-model-deployment"
title: "什么是模型部署？"
description: "定义机器学习模型部署，并解释其作为让模型提供价值的最后一步所起的作用。"
order: 1
plots: []
sourceHash: "f6d7eacf41adb9530f94cb8b1131a900b90d95d5c8275bc8f52ac32b996e4442"
sourceCorrections: []
---

在训练并验证机器学习 (machine learning)模型后，你会得到一个文件，名称可能是 `model.pkl` 或 `my_model.h5`。就其本身而言，这个文件只是一个静态资产。它包含了从数据中学到的模式，但它无法为任何人回答问题或做出预测。模型部署是将这个训练好的模型文件转化为可运行状态，以便其他软件应用程序可以使用它的过程。

简单来说，模型部署是连接模型开发与模型集成的桥梁。它是将机器学习模型放入线上生产环境，使其能够接收输入并返回预测结果所需的一系列步骤。可以将训练好的模型看作一份精心编写的食谱。食谱本身很有价值，但只有当你搭建好厨房、准备好食材并按照说明进行烹饪和上菜时，它才会变成一顿美餐。部署就是搭建那个厨房的行为。

### 从文件到服务

训练模型的环境与使用模型的环境在本质上是不同的。

- **开发环境：** 这通常是数据科学家的本地机器或云端笔记本。它是一个受控的临时空间，专为实验而设计。其目标是生成一个性能良好的模型产物。
- **生产环境：** 这是一个运行业务应用程序的线上服务器系统。此环境要求高可用性、可靠性、可扩展性和安全性。其目标是为用户或其他系统提供一致且快速的服务。

模型部署将模型从开发环境中的静态文件转换为生产环境中的动态、可调用服务。这一转换涉及多个技术步骤，本章将引导你完成这些步骤，包括封装模型的软件依赖项以及通过应用程序编程接口 (API) 对外开放模型。

> 从开发环境中的静态模型文件到生产环境中的活跃预测服务的转变。

### 部署的目标

模型部署的最终目标是使模型的智能变得可供使用。不再是由数据科学家手动运行预测，而是由网站或移动应用等应用程序通过编程方式向部署的模型发送新数据，并接收返回的预测。

例如，部署的欺诈检测模型可以接收来自电子商务网站的交易详情，并立即返回欺诈风险评分。部署的语言翻译模型可以接收来自聊天应用程序的文本，并返回翻译版本。

成功部署模型意味着它具备以下特点：

1. **可访问性：** 其他服务可以通过网络找到它并与之通信。
2. **可靠性：** 它运行稳定，不会崩溃，并能优雅地处理错误。
3. **可扩展性：** 它可以处理日益增长的预测请求而不会降低速度。

这个过程是 MLOps 的核心部分，因为它是模型开始提供实际业务价值的地方。接下来的章节将介绍打包、提供服务以及管理这一部署过程的实用技术。

## 参考资料

- [Introducing MLOps: How to Go from Experimentation to Production](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHjSlZSzafgFd5GIvusuFD4jqgav5GrLBVYbo6p2FSsNTRSQyt5gIQHboaTnGcZ4XV0Qe-1dV2gsznbUDzvsHX0b7xFfw9LgtX6aXrXtfhGdLgGOCxGJhGzPhzMBxkOD5T9jGY3P-IxFsnn7WLLmXJLw3OnYgTIfRFgiH0t0nR5q9OdV37zPGPAh8lpSNzvCtwKJRfV3z2JBequ6B1MZkQvCItivH_e157cdEPr2_57jBbrJrHHwr2PcmxJhQ==) — Mark Treveil, Nicolas Omont, Clément Stenac, Kenji Lefevre, Du Phan, Joachim Zentici, Adrien Lavoillotte, Makoto Miyazaki, Lynn Heidmann (2020)
  Publisher: O'Reilly Media, Incorporated; Pages: 150
  一份全面的 MLOps 指南，包含模型部署的定义和原则作为其核心组成部分。
- [MLOps: A guide to operations for machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2024)
  Publisher: Google Cloud
  一份来自主要云服务提供商的 MLOps 实践和概念指南，包含部署阶段。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://www.barnesandnoble.com/w/building-machine-learning-powered-applications-emmanuel-ameisen/1136306566) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media; Pages: 257
  详细介绍了将机器学习模型从开发产物转换为生产服务的流程。
