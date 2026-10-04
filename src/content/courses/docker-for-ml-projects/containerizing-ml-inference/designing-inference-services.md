---
course: "docker-for-ml-projects"
chapter: "containerizing-ml-inference"
lesson: "designing-inference-services"
sourceId: 5312
sourceUrl: "https://apxml.com/zh/courses/docker-for-ml-projects/chapter-5-containerizing-ml-inference/designing-inference-services"
title: "设计推理服务"
description: "在容器中构建可靠、可扩展推理服务的原则。"
order: 1
plots: []
sourceHash: "08d620006cb887741c112b98e27c6baf7e98f63eb8a3f6cce92e13cb85e23c9a"
sourceCorrections: []
---

机器学习 (machine learning)模型训练完成后，重心会从实验和研究转移到实际部署。提供预测服务与模型训练所需的方法有所不同。模型训练通常涉及对大数据集进行批处理，而推理 (inference)通常需要处理单个或小批量请求，并要求低延迟。将推理过程容器化提供了一种可靠且可扩展的模型部署方式。*在*编写代码*之前*设计推理服务，是构建易于维护和高效系统的重要一步。

### 定义服务接口

设计推理 (inference)服务的第一步是定义其约定：客户端将如何与其交互？这包括明确以下几点：

1. **端点：** 客户端将请求发送到哪里？对于网络服务，这通常是一个URL路径（例如，`/predict`）。
2. **方法：** 将使用哪种HTTP方法？`POST`方法在向推理端点发送数据时非常常用。
3. **输入格式：** 客户端应如何组织输入数据？JSON由于其简洁性和广泛支持，是网络API的事实标准。明确预期的键和数据类型（例如，将特征名映射到值的字典）。对于对性能要求高的应用，或者需要直接通过multipart/form-data处理二进制数据（如图像）的情况，可以考虑Protobuf等其他选项。
4. **输出格式：** 服务将如何返回预测结果？同样，JSON是典型选择，包含预测结果和可能的置信度分数或其他元数据。明确定义其结构。
5. **错误处理：** 如何传达错误？使用标准HTTP状态码（例如，输入无效时的`400 Bad Request`，模型故障时的`500 Internal Server Error`），以及一致的JSON错误消息格式，这必不可少。

良好定义的服务接口使服务可预测，并更易于客户端应用进行集成。

### 同步与异步处理

考虑服务将如何处理请求：

- **同步：** 客户端发送请求，并在同一连接中等待预测结果计算并返回。这实现起来更简单，适合预测耗时几毫秒或几秒的低延迟模型。大多数实时预测API（例如，欺诈检测、推荐片段）都使用这种模式。
- **异步：** 客户端发送请求后，服务会立即确认收到（例如，通过`202 Accepted`状态和作业ID），并在后台处理预测。客户端必须使用作业ID轮询一个端点或接收回调（例如，通过Webhook）才能稍后获取结果。这种模式更适合长时间运行的推理 (inference)任务（例如，视频分析、大批量预测），这些任务可能会超出典型的网络请求超时时间。

您的选择很大程度上取决于模型的推理时间和使用应用的要求。容器化服务可以实现这两种模式。

> 该图对比了同步和异步推理模式。同步提供即时响应，而异步则通过后台处理来应对耗时较长的任务。

### 无状态性与可扩展性

对于基于网络的推理 (inference)服务，无状态性是一个基本设计原则。每个传入请求都应独立处理，不依赖于同一容器实例中先前请求存储的信息。

为什么无状态性很重要？

- **可扩展性：** 无状态服务易于横向扩展。如果负载增加，您只需在负载均衡器后运行更多相同的容器实例。每个实例都可以处理任何传入请求。
- **可靠性：** 如果一个容器实例失败，请求可以被路由到其他健康的实例，而不会丢失会话信息。
- **简洁性：** 在多个实例之间管理状态会增加很多复杂性（例如，会话复制、分布式锁）。

避免在请求之间将用户特定数据或请求历史记录*存储在*容器的内存或本地文件系统中。如果*确实*需要状态（例如，用于用户特定的模型变体或缓存复杂的查询），请使用外部服务，如数据库、键值存储（Redis）或专门的状态管理系统。

### 模型加载策略

决定机器学习 (machine learning)模型何时以及如何加载到内存中：

- **启动时加载：** 模型在容器启动时加载，早于其开始接受请求。这会导致容器启动时间变慢，但能确保第一个请求被快速处理，不会产生模型加载延迟。这是生产服务中最常见的方法。
- **首次请求时延 (latency)迟加载：** 模型仅在第一个请求到达时才加载。这会使容器启动更快，但会为第一个请求带来明显延迟。这在开发或低流量场景中可能是可以接受的。
- **动态加载：** 模型根据请求参数 (parameter)进行加载/卸载。这更复杂，但如果您需要不频繁地提供许多不同模型，这会很有用。

在生产环境中，启动时加载通常是出于性能和可预测性的首选。确保您的容器分配了足够的内存来存放模型。

### 错误处理与日志记录

错误处理很重要。您的服务必须妥善处理各种故障模式：

- **无效输入：** 请求格式错误、数据缺失、数据类型不正确。返回清晰的错误消息和`4xx`HTTP状态码（例如，`400 Bad Request`）。
- **模型错误：** 预测期间的问题（例如，意外的输入值导致数值不稳定、模型版本不兼容）。返回`5xx`HTTP状态码（例如，`500 Internal Server Error`或可能是`503 Service Unavailable`）。
- **资源耗尽：** 内存或CPU不足。这可能表现为响应缓慢或容器崩溃。健康检查（稍后会介绍）可以帮助发现这个问题。
- **下游依赖：** 无法连接到外部数据库或进行特征查找所需的服务。

在您的服务中实现全面的日志记录。记录每个请求的重要信息（输入摘要、预测输出、延迟）以及错误的详细堆栈跟踪。对日志进行结构化（例如，JSON格式），以便日志聚合系统易于解析。容器内的标准输出（`stdout`）和标准错误（`stderr`）是由Docker守护进程或容器编排器管理的日志的常见目的地。

周全地设计这些方面，可以在您开始构建Dockerfile和编写API代码之前打下坚实基础，从而得到一个更可靠、更易于维护的容器化推理 (inference)方案。接下来的章节将介绍使用Flask/FastAPI等工具和Dockerfile优化来实际实现这些设计原则。

## 参考资料

- [Designing Machine Learning Systems: An Introduction to MLOps](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  涵盖设计和部署机器学习系统的基本概念，包括模型服务策略、推理API设计和操作考量。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491920911/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  探讨构建可靠、可扩展和可维护系统的原则，对于理解推理服务中的无状态性、同步与异步模式以及可靠数据处理非常有帮助。
- [Building Microservices: Designing Fine-Grained Systems](https://www.oreilly.com/library/view/building-microservices/9781492034025/) — Sam Newman (2021)
  Publisher: O'Reilly Media
  提供构建微服务架构的指导，适用于容器化推理服务，涵盖API契约、通信模式和独立部署。
- [RESTful Web Services](https://www.oreilly.com/library/view/restful-web-services/9780596529260/) — Leonard Richardson, Sam Ruby, and David Thomas (2007)
  Publisher: O'Reilly Media
  关于设计RESTful Web API的经典著作，对于定义服务接口、HTTP方法和预测端点的错误处理至关重要。
