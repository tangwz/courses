---
course: "mlops-for-large-models-llmops"
chapter: "advanced-llmops-systems-workflows"
lesson: "integrating-llmops-cicd"
sourceId: 3239
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-6-advanced-llmops-systems-workflows/integrating-llmops-cicd"
title: "将LLMOps与CI/CD系统集成"
description: "连接LLMOps工作流程与更广泛的软件开发CI/CD实践。"
order: 7
plots: []
sourceHash: "0c3f1b05501b1d81b6532a406b523e96fedd4f4b5102ea31bf3886b8b4c58e4b"
sourceCorrections: []
---

将专门的LLMOps工作流程与已有的持续集成和持续部署（CI/CD）实践结合，是实现大型语言模型成熟自动化操作的重要一步。将提示工程 (prompt engineering)、RAG系统和自动化再训练等操作方面整合到更广泛的软件交付生命周期中，可以确保这些系统的一致性、速度和可靠性。弥合LLM特有操作与标准CI/CD系统之间的差距是一个重要的方面。

基本目标是创建统一的管道，让改动（无论是对模型源代码、训练数据、超参数 (parameter) (hyperparameter)、提示模板还是基础设施配置的改动）都能自动触发一系列验证、测试、构建（训练/微调 (fine-tuning)）和部署步骤，最终在生产环境中生成更新的LLM服务。这为大型模型的复杂场景带来了DevOps的优势，例如更快的迭代周期、更少的人工干预、改进的协作和增强的可追溯性。

### 为什么集成LLMOps和CI/CD？

如果不进行集成，LLM的开发和部署通常会独立存在，与主应用程序开发生命周期分离。这可能导致：

- **流程不一致**：数据科学/机器学习 (machine learning)团队与运维/DevOps团队之间的手动交接增加了错误和部署失败的风险。
- **迭代缓慢**：手动触发每次改动的训练、评估和部署会减缓试验和改进的速度。
- **可追溯性差**：难以追踪是哪一版本的代码、数据和配置生成了特定的已部署模型。
- **重复工作**：为模型生命周期管理构建单独的自动化，而不是调整现有成熟的CI/CD基础设施。

将LLMOps集成到CI/CD中，通过对大型模型的部署应用成熟的软件工程规范来解决这些问题。

### 调整CI/CD以适应LLM的规模和复杂性

标准CI/CD管道通常处理相对快速的构建和测试，并部署无状态应用程序服务。而LLMOps带来了独特的需求：

1. **处理大型制品**：LLM及其相关数据集可从千兆字节到太字节不等。传统的CI/CD制品库可能难以应对。集成需要与为大型数据和模型设计的系统连接，例如云存储（S3、GCS、Azure Blob Storage），并结合使用DVC（数据版本控制）等版本控制工具或专用的模型注册表（MLflow、Vertex AI Model Registry、SageMaker Model Registry）。Git LFS可以处理模型代码和较小的配置文件，但不能处理拍字节规模的数据。
2. **管理长时间运行的流程**：LLM训练或广泛的微调 (fine-tuning)可能需要数小时、数天甚至数周。标准CI/CD作业通常有以分钟或小时计的超时时间。管道必须适应这些长持续时间。这通常包括：

   - **异步操作**：CI/CD管道可能在专用平台（如Kubeflow Pipelines、AWS SageMaker、Azure ML、Google Vertex AI Pipelines）上触发训练作业，然后轮询其状态或等待完成后的回调。
   - **解耦管道**：将初始CI阶段（代码检查、环境设置）与更长的CD阶段（训练、评估、部署）分离。CI管道可能会验证代码和配置，发布触发器或制品以启动独立的、长时间运行的CD工作流程。
3. **访问专用硬件**：训练和评估通常需要GPU或TPU。CI/CD运行器必须配置为访问这些资源。这可能包括：

   - 使用具有GPU选项的云提供商托管构建服务。
   - 在启用GPU的虚拟机或支持GPU的Kubernetes节点上设置自托管运行器（使用设备插件）。
   - 通过API直接在专用训练集群上协调作业。
4. **实施复杂的测试和评估**：除了帮助代码的单元测试外，LLM管道还需要数据验证、模型性能评估（延迟、吞吐量 (throughput)）和输出质量评估（相关性、毒性检测、幻觉 (hallucination)检查）等阶段。这些评估步骤可能生成指标、报告，甚至在进行生产部署之前需要人工循环验证步骤。
5. **环境和依赖管理**：LLMOps涉及复杂的软件堆栈（PyTorch、TensorFlow、DeepSpeed、PEFT库、特定CUDA版本）。容器化（使用Docker）几乎是确保本地开发、CI/CD运行器和生产部署目标之间环境一致性的必要条件。

### 设计集成式LLMOps CI/CD管道

集成式管道将标准软件检查与LLM特有阶段结合。这是一个概要：

1. **触发器**：由Git仓库中特定分支的提交（包含模型代码、训练脚本、提示模板、部署配置）或数据源中检测到的变化（例如，用于微调 (fine-tuning)的新标注数据）启动。
2. **CI阶段（快速反馈）**：
   - 检出代码、配置、数据清单。
   - Linting和静态分析（针对Python代码、配置文件）。
   - 单元测试（针对实用函数、数据处理逻辑）。
   - 容器构建与扫描（构建用于训练/推理 (inference)的Docker镜像，扫描漏洞）。
   - 基本配置验证。
3. **LLM训练/微调阶段（可能长时间运行）**：
   - 触发外部训练工作流程（例如，SageMaker训练作业、Kubeflow管道）。传递必要的代码、数据引用和超参数 (parameter) (hyperparameter)。
   - CI/CD控制器监控外部作业。
   - 完成后，检索模型制品、日志和初始指标。
4. **LLM评估阶段**：
   - 针对基准数据集运行自动化评估脚本。
   - 检查性能指标（目标硬件上的延迟、吞吐量 (throughput)）。
   - 评估质量指标（困惑度、BLEU、ROUGE、自定义指标、毒性分数）。
   - 生成评估报告。
   - （可选）基于评估报告的人工审查门。
5. **模型注册**：如果评估通过，则在模型注册表中版本化并注册模型制品，将其链接到源提交、数据集版本和评估结果。
6. **CD阶段（部署）**：
   - 打包模型以供服务（应用量化 (quantization)，创建优化推理配置）。
   - 部署到预演环境（使用金丝雀或影子部署等模式）。
   - 针对预演端点运行集成/冒烟测试。
   - （可选）生产发布的手动审批门。
   - 提升到生产环境（逐步发布）。
   - 部署后监控检查。

### 工具考量

您不一定需要全新的CI/CD系统。目标是集成。常见方法包括：

- **扩展现有CI/CD**：Jenkins、GitLab CI、GitHub Actions或CircleCI等工具可以协调整体工作流程。它们可以使用插件或自定义脚本与以下系统交互：
  - 云提供商服务（AWS CLI、Azure CLI、gcloud）。
  - MLOps平台（MLflow CLI、Kubeflow Pipelines SDK）。
  - 容器编排（kubectl）。
  - 制品和模型注册表。
- **工作流程协调器**：Argo Workflows或Apache Airflow等工具，通常运行在Kubernetes上，非常适合定义复杂、长时间运行的有向无环图（DAG），这些图既包含标准任务，也包含ML特有操作，包括GPU资源管理。它们有时可以充当主要的CI/CD引擎，或由更传统的CI工具触发。
- **具有CI/CD功能的MLOps平台**：Vertex AI Pipelines、SageMaker Pipelines和Azure ML Pipelines等平台提供用于构建、评估和部署模型的原生组件，通常具有与传统CI/CD重叠的触发和协调能力。它们可以单独用于MLOps部分，或集成到更广泛的CI/CD系统中。

以下是一个简化的CI/CD工作流程图，侧重于LLM微调 (fine-tuning)和部署，由代码或配置更改触发。

> 一个简化的LLMOps CI/CD管道，显示从代码提交到微调、评估、注册以及部署到预演和生产环境的各个阶段，并强调异步作业执行以及与训练集群和模型注册表等外部系统的交互。

### 集成优化方法

- **配置即代码**：将管道定义（例如，`Jenkinsfile`、`.gitlab-ci.yml`、GitHub Actions工作流文件）、超参数 (parameter) (hyperparameter)、数据配置和基础设施设置与模型代码一起存储在版本控制中。
- **模块化管道设计**：将管道分解为逻辑的、可重用的阶段或组件（例如，数据验证、训练、评估、部署）。这提高了可维护性，并允许针对不同场景进行更简单的重组（例如，只运行评估）。
- **环境一致性**：在开发、测试（CI/CD运行器）和生产环境中持续使用容器化，以最大限度地减少环境相关差异。
- **安全秘密管理**：使用内置CI/CD秘密管理或HashiCorp Vault等外部保管库，妥善管理API密钥、云服务凭据、数据库密码等。
- **职责明确分离**：即使是集成，也要保持逻辑分离。CI/CD系统负责协调，而专用工具处理各自的任务（例如，SageMaker用于训练，MLflow用于追踪，Triton用于服务）。
- **管道可观测性**：为管道阶段配置日志和指标。追踪管道执行时间、成功/失败率和资源消耗，以识别瓶颈和问题。

将LLMOps工作流程集成到CI/CD系统中，需要调整现有实践以处理大型模型独特的规模、持续时间和资源需求。通过仔细选择工具，设计异步和模块化管道，并有效管理制品和环境，您可以构建能够可靠高效地交付更新LLM的自动化系统，从而弥合模型开发与生产运维之间的差距。

## 参考资料

- [Machine learning operations (MLOps): A guide to building and managing ML pipelines](https://cloud.google.com/solutions/machine-learning-operations-mlops-guide) — Google Cloud (2023)
  为MLOps提供了一个框架，详细说明了机器学习生命周期中的CI/CD集成，并讨论了数据和模型版本控制、自动化测试以及部署策略。
- [MLOps: An Introduction to Machine Learning Operations](https://www.oreilly.com/library/view/introducing-mlops/9781492083290/) — Mark Treveil, Nicolas Omont, Clément Stenac, Kenji Lefevre, Du Phan, Joachim Zentici, Adrien Lavoillotte, Makoto Miyazaki, Lynn Heidmann (2020)
  Publisher: O'Reilly Media
  提供了实现MLOps实践的实用指导，包括关于机器学习系统的持续集成、持续交付和持续训练的章节。
