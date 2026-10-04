---
course: "optimizing-rag-for-production"
chapter: "rag-scalability-reliability-maintainability"
lesson: "cicd-pipelines-rag"
sourceId: 6259
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-7-rag-scalability-reliability-maintainability/cicd-pipelines-rag"
title: "使用CI/CD流水线自动化RAG部署"
description: "为RAG系统更新的自动化测试和部署实施CI/CD（持续集成/持续部署）流水线。"
order: 5
plots: []
sourceHash: "f0dd56314c009c24706346eb606d157d23ad87a10899b177b2551c1fcade3a51"
sourceCorrections: []
---

为确保RAG系统在生产环境中可靠运行并能有效维护，自动化部署流程是一个重要因素。像RAG这样由多个相互关联组成部分组成的复杂系统，手动部署容易出现错误、不一致，并可能迅速成为迭代和改进的瓶颈。实施持续集成（CI）和持续部署（CD）流水线是行业常规做法，它为软件生命周期带来了精确性、速度和可靠性，RAG系统也不例外。对于RAG来说，CI/CD不仅自动化代码部署，还自动化更新模型、知识库和评估指标的复杂流程。

### RAG部署中自动化的必要性

RAG系统包含多个运行部分：检索模型、嵌入 (embedding)流水线、向量 (vector)数据库、生成器大型语言模型、数据摄取工作流和应用程序逻辑。每个组成部分可能有其各自的更新周期和依赖项。尝试在不同环境（开发、预发布、生产）中手动管理这些更新效率低下，并增加配置偏差和运行失败的风险。

CI/CD流水线提供了一种结构化、自动化的方法来实现：

- **一致性**：确保每次部署遵循相同的流程，减少“在我的机器上能用”的问题。
- **速度**：通过自动化构建、测试和部署阶段，实现更快的迭代。这对于RAG系统尤其重要，因为可能需要频繁更新知识库或微调 (fine-tuning)模型。
- **可靠性**：在各个阶段引入自动化测试，以便及早发现问题，防止其进入生产环境。
- **降低风险**：最大限度地减少与手动部署步骤相关的人为错误。
- **可追溯性**：提供清晰的审计追踪记录，记录部署了什么、何时部署以及由谁部署，这对于调试和合规性很重要。

### 设计RAG系统的CI/CD流水线

RAG系统的典型CI/CD流水线涉及多个阶段，由代码、配置、模型或知识库源数据的更改触发。

> 一个通用的CI/CD流水线，用于RAG系统，展示了从源代码管理到生产环境的各个阶段。

看一下主要的阶段和要点：

#### 1. 源代码管理和版本控制

RAG系统中涉及的每个工件都应该进行版本控制。这包括：

- **应用程序代码**：RAG流水线代码本身、API端点和任何用户界面。
- **模型工件**：大型模型文件可能存储在专用工件存储（如带版本控制的S3、GCS或Azure Blob Storage，或MLflow等模型注册表）中，但它们的引用、配置以及训练/微调 (fine-tuning)脚本应在版本控制中。
- **知识库管理脚本**：用于数据摄取、预处理、分块和嵌入 (embedding)生成的脚本。
- **配置文件**：各种环境的设置、模型参数 (parameter)和流水线定义（例如，`Dockerfile`、Kubernetes清单、Terraform脚本）。
- **评估数据集和脚本**：用于测试检索器准确性和生成器质量的基准数据集。

使用Git进行源代码管理是常规做法。分支应用于开发新功能、修复错误或更新模型/知识库，并通过拉取/合并请求在合并到主部署分支之前强制执行代码审查和自动化检查。

#### 2. 持续集成 (CI)

当更改推送到代码仓库时，CI服务器（例如，Jenkins、GitLab CI、GitHub Actions）自动化执行以下任务：

- **构建**：
  - 如有必要，编译代码。
  - 为每个服务（例如，检索器API、生成器API、数据摄取工作器）构建Docker容器。这可确保一致的运行环境。
- **单元测试**：
  - 测试RAG应用程序代码的独立函数和模块（例如，文本处理工具、API处理程序）。
- **集成测试**：
  - 测试组件之间的交互。例如，验证检索器服务能否连接到向量 (vector)数据库并获取文档，或者生成器服务能否正确处理来自检索器的上下文 (context)。
- **知识库预处理和嵌入（如果数据/逻辑有变化）**：
  - 如果知识库源数据或分块/嵌入逻辑发生变化，此阶段可能涉及生成一组新的嵌入。这可能需要大量计算，因此可以优化为仅在必要时或按计划运行，输出结果进行版本控制和存储。
- **模型验证**：
  - 对于作为系统一部分的嵌入模型或大型语言模型（特别是如果经过微调），根据验证数据集运行检查，以确保性能没有下降。这可能涉及计算检索的precision@k或针对基准的生成任务的ROUGE/BLEU分数等指标。
- **RAG专项评估（离线）**：
  - 利用RAGAS或ARES等框架，或自定义评估脚本，根据基准数据集评估RAG流水线的质量。忠实性、答案相关性和上下文准确性等指标可以作为质量门。如果这些指标低于预设阈值，则构建会失败。
- **工件存储**：
  - 将构建好的Docker镜像存储到容器注册表（例如，Docker Hub、ECR、GCR）中。
  - 将模型工件、带版本号的嵌入或其他构建输出存储到工件仓库或云存储中。

#### 3. 持续交付/部署 (CD)

CI阶段通过后，CD阶段自动化发布软件到不同的环境。

- **部署到预发布环境**：
  - 将新版RAG系统自动部署到尽可能模拟生产环境的预发布环境。
  - 这包括部署更新的服务、模型，以及在知识库修改时可能需要使用新嵌入来更新预发布向量数据库。
- **在预发布环境进行端到端测试**：
  - 运行全面测试，模拟真实用户查询和与整个RAG系统的交互。
  - 在这里，您测试完整流程：查询 -> 检索 -> 上下文增强 -> 生成 -> 响应。
  - 用户验收测试（UAT）也可能在此环境中进行，可能涉及质量保证或产品团队对重要功能的A人为检查。
- **预发布环境中的知识库更新**：
  - 如果知识库更新，测试更新过程本身。这可能涉及将新文档索引到预发布向量数据库中，并验证数据完整性和可搜索性。
- **部署到生产环境**：
  - 在预发布环境验证成功（并可能获得手动批准）后，更改会部署到生产环境。
  - **部署策略**：
    - **蓝绿部署**：维护两个相同的生产环境（“蓝色”和“绿色”）。将新版本部署到非活动环境（例如，绿色）。测试后，将流量切换到绿色环境。如果出现问题，可以通过将流量切换回蓝色来即时回滚。这对于RAG尤其有用，因为它确保向量索引和模型在处理实时流量前完全就绪。
    - **金丝雀发布**：逐步将新版本推出到一小部分用户或请求。监控性能和用户反馈。如果一切正常，逐步增加流量到新版本。这限制了潜在问题的影响范围。
    - **滚动更新**：逐个或分批更新实例，确保整体服务可用。这对于RAG流水线中的无状态服务而言很常见。
  - **基础设施即代码 (IaC)**：使用Terraform或AWS CloudFormation等工具，以代码形式定义和管理基础设施（计算实例、数据库、负载均衡器）。这使环境的配置和更新可重复且一致。

### 在CI/CD中处理RAG的特定组件

自动化RAG部署需要特别关注其独特组件。

#### 知识库生命周期管理

知识库（向量 (vector)索引）是一个重要的有状态组件。CI/CD流水线必须优雅地处理其更新：

- **自动化摄取流水线**：用于获取新数据、进行预处理（分块、清洗）、生成嵌入 (embedding)，并将其加载到向量数据库中的脚本。这些可以由新数据到达或按计划触发。
- **索引版本控制/切换**：更新索引时，避免停机或结果不一致。策略包括：
  - 并行构建新版索引，并在新索引就绪并验证通过后，原子性地切换别名指向新索引（部分向量数据库支持此功能）。
  - 对于较小的更新，可能可以进行增量索引，但这需要仔细管理一致性。
- **验证**：索引更新后，运行健全性检查（例如，文档数量、样本查询）以确保更新成功。

#### 模型管理

嵌入模型和大型语言模型是RAG的核心。

- **模型版本控制**：将模型存储在模型注册表（例如，MLflow、SageMaker Model Registry、Vertex AI Model Registry）中，该注册表对模型及其关联元数据进行版本管理。
- **自动化再训练/微调 (fine-tuning)**：如果您微调自己的模型，请将再训练流水线集成到CI/CD中。这些可以由性能下降（通过评估监控）或新训练数据可用性触发。
- **模型部署**：CI/CD流水线应将正确版本模型（嵌入和生成器）部署到每个环境。配置管理在此处很重要，用于指向正确的模型URI或端点。

#### 将评估作为质量门

如第6章所述，高级评估很必要。在CI/CD中：

- 定义RAG指标（忠实性、答案相关性、上下文 (context)准确性、检索召回率）的可接受阈值。
- 在不同流水线阶段（例如，CI构建后、预发布部署后）自动化执行评估套件（例如，使用RAGAS、ARES或自定义脚本）。
- 如果指标低于阈值，则使流水线失败，防止有问题变更进入生产环境。

### RAG CI/CD 工具

各种工具可以支持您的RAG CI/CD流水线：

- **CI/CD 平台**：Jenkins、GitLab CI/CD、GitHub Actions、AWS CodePipeline、Azure DevOps、Google Cloud Build。
- **容器化与编排**：Docker、Kubernetes (EKS, GKE, AKS)。
- **基础设施即代码**：Terraform、AWS CloudFormation、Azure Resource Manager、Google Cloud Deployment Manager。
- **模型注册表与实验跟踪**：MLflow、Kubeflow、Weights & Biases、Amazon SageMaker Model Registry、Vertex AI Model Registry。
- **向量 (vector)数据库**：Pinecone、Weaviate、Milvus、Qdrant、Elasticsearch（用于密集向量 (dense vector)搜索）的托管服务或自托管实例。
- **工件仓库**：JFrog Artifactory、Sonatype Nexus、特定语言的仓库（PyPI、npm），或云存储（S3、GCS、Azure Blob）。

### 自动化RAG部署的难点

尽管益处很多，但为RAG系统设置CI/CD存在挑战：

- **流水线复杂性**：协调多个服务、模型和数据流水线的构建、测试和部署可能很复杂。
- **资源密集型测试**：运行完整的嵌入 (embedding)生成或大规模大型语言模型评估可能耗时且成本高昂。优化这些步骤，例如在常规CI运行时使用较小的样本数据集，以及在夜间构建或预生产阶段进行全面评估。
- **有状态组件管理**：更新向量 (vector)数据库或其他有状态存储需要仔细规划以避免数据丢失或服务中断。
- **环境一致性**：保持预发布环境真正代表生产环境，尤其是在数据量和基础设施规模方面，可能很困难，但对于可靠测试很重要。
- **安全**：使用HashiCorp Vault或托管秘密服务等工具在整个流水线中安全地管理秘密信息（API密钥、数据库凭据）是必要的。

通过周密设计和实施根据RAG系统具体需求量身定制的CI/CD流水线，您可以显著提升生产部署的稳定性、可靠性和敏捷性。这种自动化构成确保先进RAG应用程序长期成功和可维护性的必要运维实践的重要部分。

## 参考资料

- [Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation](https://www.informit.com/store/continuous-delivery-reliable-software-releases-through-9780321601919) — Jez Humble, David Farley (2010)
  Publisher: Addison-Wesley Professional
  关于持续集成和持续部署原则与实践的奠基性著作，为自动化软件发布流程提供了详细指导。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready ML Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  全面介绍了构建和部署生产级机器学习系统的方法，涵盖MLOps实践、数据管理、模型生命周期和ML的CI/CD。
- [RAGAS: A Framework for Automated Evaluation of Retrieval Augmented Generation Systems](https://arxiv.org/abs/2309.15217) — Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert (2023)
  Journal: arXiv preprint arXiv:2309.15217; DOI: [10.48550/arXiv.2309.15217](https://doi.org/10.48550/arXiv.2309.15217)
  介绍了一个用于评估检索增强生成系统的自动化框架，这对于将质量门集成到RAG CI/CD流水线中至关重要。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Google Cloud (2024)
  Publisher: Google Cloud
  来自Google Cloud的实用指南，介绍了如何建立MLOps实践，包括机器学习系统的持续交付和自动化流水线，适用于RAG部署。
