---
course: "prompt-engineering-llm-application-development"
chapter: "application-development-considerations"
lesson: "simple-deployment-options"
sourceId: 3097
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-8-application-development-considerations/simple-deployment-options"
title: "简单部署选项（无服务器、容器）"
description: "基于网络的LLM应用程序基本部署策略概述"
order: 6
plots: []
sourceHash: "52296902b7049cf3abee4229ee1126d342033d61e6746e8f689d5adb1a9bc98b"
sourceCorrections: []
---

在组织好应用程序代码、管理好密钥、估算好成本并实现测试和缓存之后，最后一步是让你的应用程序可用。部署包括将应用程序打包并在基础设施上运行，以便用户或其他系统可以与其交互。尽管部署可能变得相当复杂，但仍有一些更简单的选择，非常适合大型语言模型应用程序的起步，特别是无服务器功能和容器。这些方法抽象化了大部分底层基础设施管理，使你能够更侧重于应用程序逻辑。

### 无服务器部署（函数即服务）

无服务器计算，通常作为函数即服务（FaaS）实现，允许你在响应事件时运行应用程序代码，而无需管理底层服务器。像AWS Lambda、Google Cloud Functions和Azure Functions这样的平台负责配置、扩展、修补和操作执行代码所需的服务器。

**工作原理：** 你将代码（例如，一个Python函数）上传到FaaS平台。平台通常提供一种方式来触发此函数，通常通过HTTP请求（使用API Gateway等服务）。当触发发生时，平台会分配资源，运行你的函数，然后在没有更多请求时进行缩减，通常缩减到零。

**与大型语言模型应用程序的关联：**

- **事件驱动：** 许多大型语言模型应用程序的交互是事件驱动的。用户通过API端点提交查询，触发调用大型语言模型API的函数，处理响应并返回它。
- **自动伸缩：** FaaS平台根据传入请求量自动调整函数实例的数量。这对于使用量可能不可预测或突发的大型语言模型应用程序很有用。
- **按用量付费：** 你通常只为你函数运行所消耗的计算时间和请求数量付费。这可能具有成本效益，特别是对于流量可变或较低的应用程序，与大多数大型语言模型API基于使用量的计费方式相符。

**考量：**

- **冷启动：** 如果函数最近没有被调用，平台在初始化执行环境时可能会有轻微的延迟（“冷启动”）。这种延迟会影响交互式应用程序的用户体验。存在缓解此问题的策略（例如，预置并发），但它们会增加复杂性和成本。
- **执行时间限制：** FaaS平台强制执行最大执行时间限制（例如，AWS Lambda通常为15分钟）。复杂的LLM链或非常长的生成任务可能会超出这些限制。
- **无状态性：** 函数通常设计为无状态。如果你的应用程序需要在调用之间维护对话历史或其他状态，你必须依赖外部存储，如数据库（例如，DynamoDB、Firestore）或专用内存缓存（例如，Redis）。
- **部署打包：** 依赖项（如`openai`库或`langchain`）必须与你的函数代码一起打包，通常在一个zip文件中。管理依赖项和部署包大小需要注意。

无服务器计算由于其操作简单和成本模式，通常是简单的LLM支持的API、聊天机器人或后端处理任务的良好起点。

### 容器化（Docker）

容器化将应用程序的代码及其所有依赖项、库和配置文件打包成一个称为容器镜像的单一单元。Docker是最流行的容器化技术。这个镜像随后可以在不同环境中一致地运行。

**工作原理：** 你使用`Dockerfile`定义应用程序环境。此文件指定一个基础操作系统镜像（例如，一个Python镜像），列出复制代码、安装依赖项（通过`pip install -r requirements.txt`）、暴露网络端口以及定义启动应用程序的命令（例如，`CMD ["python", "app.py"]`）。你将这个`Dockerfile`构建成一个不可变镜像。这个镜像随后可以在任何支持Docker的机器或云服务上作为容器运行。

> Docker工作流程的简化视图：代码和指令被构建成一个镜像，然后在一个宿主系统上作为容器运行。

**与大型语言模型应用程序的关联：**

- **环境一致性：** 容器确保你运行大型语言模型应用程序的环境（包括Python版本、库版本，甚至系统依赖项）在任何地方都是相同的，从开发到生产。这消除了许多部署难题。
- **依赖项管理：** 打包复杂的依赖项（在使用LangChain等框架或集成多个工具的应用程序中很常见）变得更加直接。
- **可移植性：** 容器镜像可以存储在注册表（如Docker Hub、AWS ECR、Google Artifact Registry）中，并拉取到各种平台运行：本地机器、虚拟机或托管容器服务（例如，AWS Fargate、Google Cloud Run、Azure Container Instances）。
- **灵活性：** 与无服务器功能相比，容器对执行环境提供更多控制，并且受平台特定的执行时间或包大小限制较少。

**考量：**

- **基础设施：** 你需要基础设施来运行你的容器。这可能是一个你自己管理的虚拟机，或者更常见的是云提供商提供的托管容器服务。尽管像Cloud Run或Fargate这样的服务抽象了服务器管理，但它们仍然需要配置。
- **镜像大小：** LLM应用程序通常依赖大型库，这可能导致容器镜像很大。优化镜像大小对于更快的部署和降低存储成本很重要。
- **资源管理：** 你需要配置容器所需的CPU和内存。资源配置不足可能导致性能问题或崩溃，而资源配置过多则会增加成本。
- **学习曲线：** 尽管Docker基础知识易于上手，但与部署简单的无服务器功能相比，掌握Dockerfile、网络和选择合适的托管环境需要一个学习过程。

容器提供更大的灵活性和控制，使它们适合更复杂的应用程序、具有特定运行时要求的应用程序，或者作为使用Kubernetes等编排工具构建更具可伸缩性架构的一步。

### 选择一个选项

最佳选择取决于你的具体需求：

- **如果选择无服务器（FaaS）：**
  - 你的应用程序相对简单且事件驱动（例如，一个基本的API端点）。
  - 自动伸缩和按用量付费是主要优势。
  - 你想从最简单的操作模式开始。
  - 执行时间限制和潜在的冷启动可以接受。
- **如果选择容器（Docker）：**
  - 你需要在开发和生产之间保持严格的环境一致性。
  - 你的应用程序具有复杂的依赖项或需要特定的系统库。
  - 你需要比FaaS通常允许的更长的执行时间。
  - 你预期需要对基础设施有更多控制，或者计划向容器编排（如Kubernetes）发展。
  - 跨不同云提供商或本地部署的可移植性是一个要求。

无服务器功能和容器都提供了有效的方法来部署大型语言模型应用程序，而无需管理物理服务器的负担。通常，开发者会从无服务器开始，用于初始原型或简单的端点，随着应用程序的复杂性和操作要求增加，可能会迁移到容器。了解这些权衡有助于你选择合适的起点，使你的大型语言模型应用程序投入使用。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker Inc. (2024)
  提供关于 Docker 的全面指南和参考资料，涵盖从基本概念到高级使用的容器化技术。
- [What is AWS Lambda?](https://docs.aws.amazon.com/lambda/latest/dg/lambda-get-started.html) — Amazon Web Services (2024)
  AWS Lambda 的官方指南，详细介绍其功能、工作原理以及无服务器函数部署的最佳实践。
- [Google Cloud Run documentation](https://cloud.google.com/run/docs) — Google Cloud (2024)
  Publisher: Google Cloud
  Google Cloud Run 的官方文档，展示如何将无状态容器部署为无服务器网络服务或作业，并进行自动扩缩。
- [Docker Up & Running: Building and Deploying Scalable Applications](https://www.oreilly.com/library/view/docker-up-running/9781492040982/) — Karl Matthias and Sean P. Kane (2019)
  Publisher: O'Reilly Media
  理解 Docker 和容器化的实用指南，涵盖核心概念、镜像构建以及在容器中运行应用程序。
