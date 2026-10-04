---
course: "planning-optimizing-ai-infrastructure"
chapter: "leveraging-cloud-platforms-for-ai"
lesson: "comparing-managed-ai-services-iaas"
sourceId: 6995
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-3-leveraging-cloud-platforms-for-ai/comparing-managed-ai-services-iaas"
title: "托管AI服务与IaaS的比较"
description: "评估使用高级托管AI平台（如SageMaker、Vertex AI）与基于原始基础设施即服务（IaaS）构建之间的权衡。"
order: 2
plots: []
sourceHash: "bfff0d18316a5dcdf35a32d22a95f318c5f863a535eb695b94b732a46ae88f45"
sourceCorrections: []
---

在使用云平台进行AI工作时，您将面临的首要选择之一是两种不同的服务模式：基础设施即服务（IaaS）和托管AI平台。这个选择代表了控制与便捷之间的基本权衡。您的决定将对团队的工作流程、开发速度和运维职责产生很大影响。

### 基础设施即服务（IaaS）：底层构件

IaaS为您提供原始的计算、存储和网络组件。可以将其视为在云中租用一台裸机服务器。您几乎负责硬件虚拟化层之上的所有事务。

采用IaaS方法，您的工作流程通常包括：

1. **配置虚拟机（VM）：** 您选择一个虚拟机实例，例如Amazon EC2实例、Google Compute Engine VM或Azure虚拟机。您选择CPU、内存以及连接的GPU类型和数量。
2. **配置环境：** 您连接到机器（通常通过SSH）并扮演系统管理员的角色。这包括安装操作系统、NVIDIA驱动程序、CUDA工具包以及特定版本的Python和机器学习 (machine learning)库，例如PyTorch或TensorFlow。
3. **管理数据和代码：** 您负责将数据集传输到机器的存储中并管理代码库。
4. **运行工作负载：** 您从命令行手动运行训练脚本或部署推理 (inference)服务器，或通过自定义自动化脚本进行。

IaaS的主要优点是**控制**。您可以构建一个完全自定义的环境，根据特定或不寻常的需求进行定制。如果您的工作依赖于专有软件或托管平台不支持的非常特殊的库版本，这会很有用。

然而，这种控制的代价是**高昂的运维开销**。您的团队必须具备管理系统依赖、应用安全补丁和解决低层基础设施问题的专业知识。开始使用通常较慢，因为在任何机器学习工作开始之前都需要大量的设置。

### 托管AI服务：集成平台

托管AI服务是更高级别的平台，它们抽象了底层基础设施。Amazon SageMaker、Google Cloud的Vertex AI和Azure机器学习 (machine learning)等服务专为机器学习生命周期设计。它们将计算资源与一套用于数据标注、模型训练、超参数 (parameter) (hyperparameter)调优和部署的工具捆绑在一起。

使用托管服务，您的工作流程会有明显变化：

1. **定义任务：** 您通常通过网页界面、SDK或配置文件定义训练任务或端点，而不是配置虚拟机。
2. **指定资源：** 您仍然选择实例的类型和数量（例如，`ml.g4dn.xlarge`），但您不直接管理这些实例。
3. **提供脚本或容器：** 您将服务指向您的训练脚本或Docker容器。平台负责供应基础设施、运行您的代码，并在任务完成后销毁资源。
4. **使用集成工具：** 您获得内置功能，用于自动化超参数调优、实验跟踪以及带自动伸缩功能的一键式模型部署。

这里的主要好处是**生产力**。数据科学家可以更多地专注于模型开发，减少基础设施管理。从构想到训练好的模型所需时间通常大大缩短。这些平台还提供清晰的生产路径，并具备集成的MLOps能力。

权衡之下是**灵活性**的降低。您在平台提供的环境中操作，这可能对库版本或系统配置有所限制。也存在一定程度的厂商锁定，因为使用特定平台SDK构建的管道不容易移植到另一个云提供商。

### 职责比较

这两种模式之间的差异可以通过查看谁负责技术堆栈的每一层来呈现。对于IaaS，您的团队责任范围延伸至堆栈的较深层。对于托管服务，云提供商处理大部分运维负担。

> IaaS与托管AI服务的职责堆栈。使用IaaS时，您从操作系统向上管理环境。使用托管服务时，您主要专注于应用程序代码，而提供商管理平台和底层软件。

### 做出正确选择

选择合适的模式取决于您的团队技能、项目需求和业务目标。

**选择IaaS，如果：**

- 您需要运行自定义软件，或需要托管平台不支持的环境。
- 您拥有能够构建和维护基础设施的强大DevOps或MLOps团队。
- 按小时计费的成本优化是主要驱动因素，并且您拥有稳定、长时间运行的工作负载，设置开销是合理的。
- 您需要避免平台层面的厂商锁定，并希望您的运维脚本具有最大可移植性。

**选择托管AI服务，如果：**

- 您的主要目标是加速机器学习 (machine learning)开发和实验周期。
- 您的团队主要由希望专注于建模的数据科学家和研究人员组成。
- 您需要开箱即用的完整MLOps方案，包括实验跟踪、自动化调优和简化部署。
- 您愿意为便捷性、减少运维负担和更快的上市时间支付更高的费用。

采用**混合方法**也很常见。一个团队可能使用IaaS（原始虚拟机）处理繁重、自定义的数据预处理任务，然后使用托管服务的训练和托管能力进行模型阶段。这使得您能够混合搭配服务，为管道的每个部分使用最合适的工具。这个选择并非一成不变；您可以随着团队和项目的成熟调整策略。

## 参考资料

- [What is Amazon EC2?](https://docs.aws.amazon.com/ec2/latest/devguide/what-is-amazon-ec2.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供了Amazon EC2的详细概述，展示了计算资源的核心IaaS功能。
- [What is Amazon SageMaker?](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html) — Amazon Web Services (2024)
  介绍了Amazon SageMaker用于机器学习生命周期的集成功能，作为托管AI服务的示例。
- [Building Machine Learning Powered Applications: Going from Idea to Product](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGwzLpXvL_mYAqPNc-MQzRRNzkxacWAmUKqYTSJc-6gGqosJjkMgM9j2qT1Cw1Ft_CCQFJAwoTYLbmDZ8E8zDR2ux9dJWHAJcujVBKM6GHvU7z5NtN_Y9x9s-hBC7Me1RDNjWeXX2PSEZ-n_CP9nEbguyR5Z3RnU1KsIEqmkOMv3gQLLxBzOGbEPV7b7xmaBPq9Yb7pX3iMKPSmaSMLr5CcLALNjo=) — Emmanuel Ameisen (2020)
  Publisher: O'Reilly Media
  提供了构建和部署机器学习应用程序的实用指导，包括对基础设施选择和托管服务简化的MLOps实践的讨论。
