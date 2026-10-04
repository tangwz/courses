---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "advanced-kubernetes-orchestration"
lesson: "managing-workflows-kubeflow"
sourceId: 6988
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration/managing-workflows-kubeflow"
title: "使用 KubeFlow Pipelines 管理机器学习工作流程"
description: "KubeFlow 在 Kubernetes 上编排和自动化端到端机器学习管道的概述。"
order: 1
plots: []
sourceHash: "e196ba914de6339cd05faf8ea79a54f5a8bf67d7dcb6c85ce865a752b2778483"
sourceCorrections: []
---

运行单个分布式训练任务是一项重要的工程任务，但生产机器学习 (machine learning)需要编排一系列相互关联的任务。从数据引入和验证到模型训练、评估和条件性部署的整个工作流，都必须自动化、可复现且可扩展。手动执行这些步骤不仅效率低下，还容易出错，特别是在多用户环境中。这时，工作流编排器就显得必不可少。

KubeFlow Pipelines 是一个 Kubernetes 原生工作流引擎，专门用于组合、部署和管理端到端机器学习管道。作为 KubeFlow 项目的一部分，其主要作用是提供一个结构化框架，用于定义复杂的任务图，其中每个步骤都作为容器在 Kubernetes 集群上运行。这种做法让您能够直接在您的机器学习工作流定义中运用 Kubernetes 的全部能力进行资源管理、调度和隔离。

### KubeFlow 管道的结构

KubeFlow 管道是一个有向无环图 (DAG)，图中每个节点都代表一个特定任务或 `组件`。

- **组件 (Component):** 管道的基本构成单元。组件是一个自包含、容器化的应用程序，执行一个单一、明确的步骤。这可以是任何操作，例如从对象存储下载数据、使用 `pandas` 转换特征或使用 `PyTorch` 训练模型。因为每个组件都是一个容器，您可以指定其精确的资源需求，例如“此训练步骤需要 4 个 CPU、32Gi 内存和 1 个 NVIDIA A100 GPU。”
- **管道 (Pipeline):** 工作流的正式定义，描述了组件集及其依赖关系。它指定了数据和产物如何从一个组件流向下一个组件。
- **产物 (Artifact):** 组件生成的输出。产物可以是数据集、训练好的模型文件、评估指标或 HTML 可视化内容。KubeFlow 自动管理这些产物在管道步骤之间的存储和传输。

下图展示了一个标准管道结构。数据从一个组件流向下一个组件，并有可能根据前一步骤的输出（如模型评估指标）进行条件性执行。

> 一个典型的定义为 DAG 的机器学习 (machine learning)管道。每个步骤都是一个容器化组件，转换表示产物的流动。训练步骤可以调度到专用的 GPU 节点上运行。

### 定义组件和管道

您可以使用 Python 中的 KubeFlow Pipelines SDK 定义管道。这让您可以使用熟悉的编程结构来构建工作流。组件通常通过装饰 Python 函数来创建。

考虑一个用于数据预处理的简单组件。函数签名使用 `kfp` SDK 的类型注解来声明其输入和输出。这里，它接收一个输入 `Dataset` 产物，并生成一个新的、经过处理的 `Dataset` 产物。

```python
from kfp.dsl import component, Input, Output, Dataset

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas==1.5.3", "scikit-learn==1.2.2"]
)
def preprocess_data(
    raw_data: Input[Dataset],
    processed_data: Output[Dataset]
):
    """加载原始数据，执行清理，并保存结果。"""
    import pandas as pd

    df = pd.read_csv(raw_data.path)
    # 执行清理操作
    df.dropna(inplace=True)
    df = df[df['value'] > 0]
    
    df.to_csv(processed_data.path, index=False)
```

请注意 `@component` 装饰器如何指定容器环境，包括基础镜像和所需的 Python 包。KubeFlow 使用这些信息为该步骤构建容器镜像或使用现有镜像，确保执行环境的隔离性和可复现性。

定义好组件后，您同样可以使用装饰的 Python 函数将它们组合成一个管道。管道函数通过将一个组件的输出作为另一个组件的输入来定义数据流。

```python
from kfp.dsl import pipeline

# 假设 'train_model_op' 是另一个已定义的组件
# 从 training_component 导入 train_model_op

@pipeline(
    name="simple-training-pipeline",
    description="一个用于预处理数据和训练模型的管道。"
)
def my_training_pipeline(data_path: str):
    # 这将是一个从 URL 或存储桶导入数据的组件
    # 为简化起见，我们假设存在一个组件 'import_data_op'
    import_task = import_data_op(source_url=data_path)

    # 使用前面定义的预处理组件
    preprocess_task = preprocess_data(
        raw_data=import_task.outputs["dataset"]
    )

    # 训练组件使用预处理过的数据
    train_task = train_model_op(
        training_data=preprocess_task.outputs["processed_data"]
    )
    
    # 您还可以为特定任务指定资源请求
    train_task.set_gpu_limit(1).set_cpu_limit("8").set_memory_limit("32G")
```

在此管道定义中，`preprocess_task.outputs["processed_data"]` 直接传递给 `train_model_op` 组件。KubeFlow 处理数据传递的底层机制，通常通过将输出产物写入共享对象存储并将其路径提供给下一个容器来完成。最后一行展示了如何声明性地为给定步骤请求特定硬件，这是一项与 Kubernetes 调度器直接集成的功能。关于 GPU 调度下一节将说明集群如何满足此类请求。

### KubeFlow 在编排中的应用

虽然 Apache Airflow 等其他工作流编排器也能管理机器学习 (machine learning)任务，KubeFlow 的优势在于其 Kubernetes 原生架构。在 Airflow 中，在 Kubernetes 上执行任务通常需要使用 `KubernetesPodOperator`，这增加了定义 Pod 和资源请求的配置层。在 KubeFlow 中，整个系统都建立在 Kubernetes 概念之上。这种紧密集成简化了资源感知型管道的定义以及容器化环境中依赖关系的管理。对于已经选择 Kubernetes 进行训练和服务的机器学习团队来说，KubeFlow Pipelines 提供了一个更直接、更统一的编排方案。

通过使用 KubeFlow，您可以将机器学习流程从一系列手动脚本转变为一个版本控制、自动化和资源感知的系统。这为管理生产机器学习中典型的复杂多步骤工作负载提供了基础。接下来的章节将详细说明 Kubernetes 如何满足这些管道提出的资源请求，包括从 GPU 调度到集群本身的动态扩展。

## 参考资料

- [KubeFlow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/) — The KubeFlow Authors (Ongoing)
  KubeFlow Pipelines 的官方指南，涵盖概念、SDK 使用和部署。
- [Kubernetes Documentation](https://kubernetes.io/docs/) — The Kubernetes Authors (2024)
  Publisher: Cloud Native Computing Foundation (CNCF)
  Kubernetes 平台的综合文档，解释了其基础设施。
- [MLOps in Production: A Survey on Challenges and Solutions](https://www.mdpi.com/23/12/5482) — Tobias Gindele, Tobias Klemps, Marc F. Schöfberger, Klaus P. Jantke, and André Bresink (2023)
  Journal: Sensors; Publisher: MDPI; Volume: 23; Pages: 5482; DOI: [10.3390/s23125482](https://doi.org/10.3390/s23125482)
  一项关于 MLOps 挑战和解决方案的调查，为工作流编排器提供了背景。
