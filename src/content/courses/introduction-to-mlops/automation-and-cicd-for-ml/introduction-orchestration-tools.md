---
course: "introduction-to-mlops"
chapter: "automation-and-cicd-for-ml"
lesson: "introduction-orchestration-tools"
sourceId: 7241
sourceUrl: "https://apxml.com/zh/courses/introduction-to-mlops/chapter-4-automation-and-cicd-for-ml/introduction-orchestration-tools"
title: "编排工具简介"
description: "简要介绍用于调度、管理和监控复杂机器学习工作流及流水线的工具。"
order: 6
plots: []
sourceHash: "5af6c3e9b8821ca70aa3b0ef666d538b57c9b7ba70103c70f395d73ef219897d"
sourceCorrections: []
---

许多机器学习 (machine learning)流水线在初期可以设计为单个脚本。然而，生产系统很少以这种方式运行。一个典型的机器学习工作流由多个相互关联的阶段组成：数据验证、预处理、模型训练、评估和部署。如果“数据验证”步骤失败，则不应继续进行“模型训练”。如果“训练”成功，则希望自动触发“评估”。管理这种顺序、处理失败以及安排运行，需要的不仅仅是一个简单的脚本。编排工具正是为了实现这一功能。

编排工具充当机器学习流水线的指挥家。它本身并不执行具体任务，如数据处理或模型训练。相反，它指导应该运行哪个任务、何时运行，以及在任务成功或失败时该怎么做。它管理从开始到结束的整个工作流。

### 为什么编排必不可少

没有编排器的自动化流水线就像一条没有中央控制系统的工厂装配线。每个站点可能都在工作，但整体流程是脆弱且难以管理的。编排器提供的控制系统使流水线变得可靠且易于扩展。

以下是它们解决的主要问题：

- **依赖管理：** 编排器确保任务按正确的顺序执行。例如，只有在模型训练任务成功完成后，模型评估任务才会开始。这可以防止错误并避免计算资源浪费。
- **调度：** 你可以安排流水线在特定时间或间隔自动运行。一个常见的案例是持续训练 (CT)，即每周在不进行任何人工干预的情况下，使用新数据对模型进行重新训练。
- **错误处理和重试：** 如果由于临时网络故障导致任务失败，该怎么办？编排器可以配置为在放弃之前自动重试失败的任务几次，而不是让整个流水线直接报错。
- **监控和日志：** 编排器提供集中式的仪表板来查看流水线的状态。你可以看到哪些任务正在运行、哪些已经完成、哪些失败了。如果任务失败，你可以轻松访问其日志以诊断问题。
- **扩展性：** 对于大规模任务，编排器可以将工作分配到多台机器上，通过尽可能并行运行任务，让你能够处理更多数据并加快模型训练速度。

### 工作流的蓝图：有向无环图 (DAG)

编排工具使用一种称为**有向无环图**（**DAG**）的结构来定义工作流。这听起来可能有点抽象，但想法其实很直观。DAG 只是描述一组任务及其之间依赖关系的一种方式。

- **有向 (Directed)：** 任务之间的连接是有方向的。这表示一个任务必须在下一个任务开始之前完成。例如，一个箭头从“处理数据”指向“训练模型”。
- **无环 (Acyclic)：** 图中不包含任何循环。一个任务不能是其自身的祖先，这防止了工作流陷入无限循环。它有明确的起点和终点。

让我们将一个简单的机器学习 (machine learning)流水线可视化为 DAG。

> 代表机器学习流水线的有向无环图 (DAG)。每个框代表一个任务，箭头定义了执行顺序。

在此图中，编排器明白必须先完成“数据接入”，然后是“数据验证”，依此类推。在数据成功处理之前，它不会尝试训练模型。从“模型评估”到“模型部署”的虚线代表条件逻辑：模型只有在通过评估标准后才会被部署。

### 常见的编排工具

有几种工具可用于编排流水线。虽然它们都使用 DAG 结构，但在设计和主要用途上有所不同。

#### Apache Airflow

Apache Airflow 是一个成熟的开源平台，用于编排通用工作流。它并非专门为机器学习 (machine learning)设计，但由于其灵活性，在这一领域得到了广泛使用。

- **工作流即代码：** 你使用 Python 定义 DAG，这为编写复杂逻辑提供了极大的灵活性。
- **丰富的界面：** Airflow 带有详细的用户界面，用于可视化流水线、监控进度和管理运行。
- **可扩展性：** 它拥有庞大的提供商和钩子 (hooks) 生态系统，使其能够连接到几乎任何外部系统，如云存储、数据库和计算服务。

#### Kubeflow Pipelines

Kubeflow 是一个致力于使 Kubernetes 上的机器学习工作流变得简单、可移植且易于扩展的项目。Kubeflow Pipelines 是负责编排的组件。

- **Kubernetes 原生：** 它从底层开始就是为了在 Kubernetes（一个容器编排平台）上运行而设计的。这使得它非常适合已经在使用 Kubernetes 的团队。
- **专注于机器学习：** 它带有专门为机器学习任务构建的组件和集成，例如实验跟踪和元数据管理。
- **基于组件：** 流水线通过连接预先打包的容器化组件来构建，这提高了可重用性。

#### GitHub Actions

正如你在我们的动手实践中将看到的，像 GitHub Actions 这样的工具也可以作为轻量级编排器。虽然它不是像 Airflow 或 Kubeflow 那样的专用编排平台，但它是执行 CI/CD 和简单的事件驱动流水线的出色工具。

- **事件驱动：** 工作流通常由代码库事件触发，例如 `git push` 或拉取请求 (pull request)。
- **配置简单：** 工作流在 YAML 文件中定义，易于编写和理解。
- **与源代码控制集成：** 由于它内置在 GitHub 中，对于自动执行与代码库直接相关的任务（如运行测试和构建容器镜像）来说，这是一个非常方便的选择。

选择哪种工具取决于项目的规模和团队的环境。对于简单的 CI 任务，GitHub Actions 就足够了。对于复杂的、有计划的重新训练流水线，像 Airflow 或 Kubeflow 这样的专用工具可以提供更多关于调度、监控和恢复的功能。

## 参考资料

- [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/) — Hannes Hapke, Catherine Nelson, and Rahul Agarwal (2020)
  Publisher: O'Reilly Media
  为构建、部署和维护机器学习管道提供了实用指导，包括编排的必要性和模式。
- [Apache Airflow Documentation](https://airflow.apache.org/docs/) — The Apache Software Foundation (2024)
  Apache Airflow 的官方文档，详细介绍了其架构、DAG 定义和工作流编排的操作方面。
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/) — The Kubeflow Authors (2024)
  Kubeflow Pipelines 的官方文档，概述了其用于构建和管理 ML 工作流的 Kubernetes 原生设计。
- [About GitHub Actions](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHjb7M6wOHM6wfnld1ND1XXUn217kCjcXQezGNVMU-bhxy6sHrbgz-qmTMgNm66aZzfhISY-cYIRPK3bTpOFAsxYMnJeQawSyT_FAkxbM0E_ahAUxsdJVHnuYsjX8rStx-9cX2KMIWO4qWn9A==) — GitHub Docs (2022)
  Publisher: GitHub, Inc.
  介绍 GitHub Actions 的官方文档，用于自动化软件开发工作流，包括 CI/CD 和事件驱动任务。
