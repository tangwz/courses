# MLOps 策略的目标

来源：[原文](https://apxml.com/zh/courses/introduction-to-mlops/chapter-1-foundations-of-mlops/goals-of-mlops-strategy)

[返回章节目录](README.md) · [返回课程目录](../README.md)

实施 MLOps 策略能将机器学习 (machine learning)项目从实验性质的练习转变为可靠的业务功能。这种转变需要一套明确的目标，这些目标不仅是让模型在开发机器上运行一次，还为构建自动化、可重复且可扩展的系统提供了框架，最终将模型的潜力转化为实际的业务价值。

### 实现整个机器学习 (machine learning)生命周期的自动化

MLOps 的首要目标是在机器学习生命周期的每个阶段引入自动化。在许多项目中，从获取新数据到重新训练模型的过程往往是手动且耗时的，涉及多个步骤以及团队之间的交接。这种方式不仅慢，而且容易出错，也难以扩大规模。

MLOps 旨在用自动化流水线取代这些手动步骤。可以将其看作模型的装配线。这条流水线将数据摄取、预处理、模型训练和验证连接成一个统一的工作流，并可以自动触发。例如，可以配置流水线在有新数据可用时或按定期计划运行。这种自动化大幅缩短了更新模型所需的时间，让团队能够腾出精力去解决更重要的问题。

### 确保可重复性

机器学习 (machine learning)中最常见的挑战之一是“在我的机器上可以运行”的问题。数据科学家可能在 Notebook 中构建了一个高性能模型，但当其他人尝试重新创建时，结果却不一致，或者干脆运行失败。这种缺乏可重复性的情况使得排查问题、审计结果或放心地在先前工作基础上进行开发变得不可能。

MLOps 策略将可重复性设定为一项必须满足的要求。这通过对影响最终模型的每个组件进行系统化的版本控制来实现：

- **代码：** 用于数据处理、训练和评估的脚本使用 Git 等版本控制系统进行跟踪。
- **数据：** 对用于训练模型的具体数据集进行版本化。即使数据的细微变化也可能导致产生完全不同的模型。
- **模型：** 训练好的模型产物，连同其性能指标和参数 (parameter)，都会被存储并进行版本控制。

通过同时跟踪这三个要素，你可以为生成的每个模型创建完整且可审计的记录。如果需要回滚到以前的版本，或者需要了解特定模型的具体创建过程，你将拥有所需的全部信息。

### 构建可靠性与扩展性

一个在开发环境中表现良好的模型，只有当它能够部署到既可靠又可扩展的生产系统中时才有用。可靠性意味着模型服务系统能够持续可用，并稳定地返回预测结果。扩展性则意味着它能够处理日益增长的请求数量或数据量，而不会出现性能下降。

MLOps 的目标直接支持了这一点。通过使用 Docker 容器化等实践，你可以将模型及其依赖项打包成一个独立的单元，从而在不同环境中保持运行的一致性。通过实施监控，你可以跟踪模型的运行健康状况，例如延迟和错误率。这种对运行卓越性的关注确保了模型不仅能工作，而且在生产环境的工作负载压力下也能表现良好。

### 促进协作与治理

机器学习 (machine learning)不是一项单打独斗的活动。它需要数据科学家、机器学习工程师、软件开发人员和运维团队的专业知识。如果没有统一的流程，这些团队往往各自为政，导致摩擦和效率低下。

MLOps 创建了一个统一的工作流，弥合了这些不同角色之间的鸿沟。它提供了一种通用的语言和一套共享的工具，使团队能够更有效地协作。数据科学家可以专注于实验，而机器学习工程师可以在同一个结构化框架内专注于模型的生产化。

此外，这种结构化的方法有助于实现强有力的治理。跟踪谁训练了哪个模型、使用了什么数据、验证表现如何以及何时部署变得非常容易。这种程度的可追溯性不仅是良好的实践，对于许多行业来说，更是监管的要求。

> MLOps 策略的主要目标共同协作，以支持在生产环境中运行可靠的机器学习系统这一核心目标。

## 参考资料

- [Introducing MLOps: How to Go from Idea to Production](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFeKPfi9XJXUiwgDeUr-oj4XKPBMfBP4GlwnxPRz9CGdN5LAMtOCaehAYFXjfVuYEbvMmptAetbd7wNhMbiKHRANz0lvM3OAO4YGuX5nzj099zK5JyOr0HQOy_M01zzCpCqgXn3ZMbZPamVsM16ax-56QouA5AzAODDsumsgfU6F-VmyAtV5ZEtAaOrmNxYYiXW4_xk8uR1ySPxZ01W) — Mark Treveil, Nicolas Omont, Aurélien Géron, Clément Stenac, Cécile Tran, Andreas Brauchli, Noah Steward, João Moura, Michel Tugendhat, Larysa Visengeriyeva, Harley Davidson, Alexey Goldin, Justin Francis, John D. K. Miller, Roger B. Chen, David S. D. Jones, and Sallyann Freudenberg (2020)
  Publisher: O'Reilly Media
  一份MLOps的实用指南，概述了其原则、益处以及如何实施完整的MLOps策略，涵盖自动化、可复现性、可靠性和协作。
- [MLOps: Continuous delivery and automation for machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — V. Lakshmanan, S. G. Chandrasekaran, S. S. Padmanabhan, V. J. Raman, M. J. F. Johnson, and D. L. Martin (2022)
  Publisher: Google Cloud
  一份来自谷歌云的权威白皮书，定义了MLOps，概述了其原则，并详细说明了如何为机器学习实施持续交付和自动化，直接支持本节所述目标。
- [MLOps: A Comprehensive Definition, Principles, and a Framework for Machine Learning Operations](https://dl.acm.org/doi/10.1145/3549727) — Jörg Kietzmann, Andreas K. Steiner, Michael K. Reiss, Christian J. F. Maass (2022)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 55; Pages: 1-38; DOI: [10.1145/3549727](https://doi.org/10.1145/3549727)
  提供对MLOps、其基本原则和结构化框架的全面学术定义。它为MLOps策略的目标和组成部分提供了基于研究的视角。

---

[上一节](03-MLOps%20%E4%B8%8E%20DevOps%EF%BC%9A%E7%9B%B8%E4%BC%BC%E7%82%B9%E4%B8%8E%E4%B8%8D%E5%90%8C%E7%82%B9.md) · [下一节](05-MLOps%20%E7%9A%84%E6%A0%B8%E5%BF%83%E5%8E%9F%E5%88%99.md)
