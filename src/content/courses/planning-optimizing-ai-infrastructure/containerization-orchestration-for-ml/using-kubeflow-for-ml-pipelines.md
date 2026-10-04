---
course: "planning-optimizing-ai-infrastructure"
chapter: "containerization-orchestration-for-ml"
lesson: "using-kubeflow-for-ml-pipelines"
sourceId: 7017
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/using-kubeflow-for-ml-pipelines"
title: "使用 Kubeflow 构建机器学习管道"
description: "Kubeflow 简介：一个用于在 Kubernetes 上构建、部署和管理可移植、可扩展机器学习工作流的工具。"
order: 6
plots: []
sourceHash: "f0a48702926de40144b01afc137b417018e29ade0684cd40fbb4e1d0027f8677"
sourceCorrections: []
---

虽然 Kubernetes 提供了运行容器的基础能力，但管理多步骤的机器学习 (machine learning)过程需要更结构化的方法。一个典型的机器学习项目并非一个独立的应用程序；它是一个包含不同阶段的工作流程，例如数据摄取、预处理、模型训练、评估和部署。使用 `kubectl` 命令手动编排这个序列会很复杂且容易出错。这正是 Kubeflow 发挥作用的地方。

Kubeflow 是一个专为 Kubernetes 设计的开源机器学习工具集。它不替代 Kubernetes。相反，它在 Kubernetes 之上构建，提供了一系列工具，简化了大规模部署、监控和管理机器学习系统的过程。Kubeflow 的核心目标是让 Kubernetes 上的机器学习工作流实现**可组合**、**可移植**和**可扩展**。

编排这些工作流的核心功能是 **Kubeflow Pipelines**。它提供了一个框架和用户界面，用于构建和部署可复用的机器学习管道。

### 理解 Kubeflow Pipelines

一个 Kubeflow Pipeline 是一个由容器化任务构成的有向无环图（DAG）。图中的每个任务都是一个独立的**组件**。让我们细致了解这些元素。

#### 组件

组件是管道的基本构建单元。它是一个独立的、容器化的应用程序，在您的工作流中执行一个单一的步骤。可以将组件视为一个具有强类型输入和输出的函数。例如，您可以有以下用途的组件：

- 从云存储下载数据集。
- 清洗和转换数据。
- 训练机器学习 (machine learning)模型。
- 评估模型的准确性。

因为每个组件都是一个容器，所以它封装了自身的代码和依赖项。这意味着数据预处理组件可以使用与模型训练组件不同的库集，甚至不同的 Python 版本，从而确保关注点清晰分离。

#### 管道

管道通过连接组件来定义机器学习工作流的结构。您定义一个组件的输出如何成为另一个组件的输入，从而创建一个依赖关系图。例如，处理后的数据路径（您的 `preprocess` 组件的输出）作为输入参数 (parameter)传递给您的 `train` 组件。

这种图结构使 Kubeflow 能够管理执行顺序，确保一个步骤只有在其所有依赖项都满足后才运行。它还允许独立步骤的并行执行，提高了整体效率。

> 一个典型的机器学习管道，定义为组件的图。实线表示直接依赖，而虚线可以表示条件步骤，例如仅当模型评估得分达到特定阈值时才部署模型。

#### 使用 Python 定义管道

Kubeflow Pipelines 最有效的特性之一是其 Python SDK (`kfp`)。它允许您使用熟悉的 Python 代码定义组件和管道，然后将其编译为 Kubernetes 可以理解的静态 YAML 配置。

让我们来看一个简化的例子，说明如何定义一个两步管道。

首先，您定义您的组件。`@dsl.component` 装饰器将 Python 函数转换为可复用的管道组件。您指定其依赖项并定义其输入和输出。

```python
from kfp import dsl
from kfp.compiler import Compiler

# 组件 1: 预处理数据
@dsl.component(
    base_image='python:3.9',
    packages_to_install=['pandas==1.3.5']
)
def preprocess_data(
    raw_data_path: str, 
    processed_data: dsl.OutputPath('Dataset')
):
    """加载原始数据，进行清洗，并保存到输出路径。"""
    import pandas as pd
    df = pd.read_csv(raw_data_path)
    # Perform cleaning operations
    df_cleaned = df.dropna()
    df_cleaned.to_csv(processed_data, index=False)
    print("数据预处理完成。")

# 组件 2: 训练模型
@dsl.component(
    base_image='tensorflow/tensorflow:2.8.0', # 使用不同的镜像
)
def train_model(
    dataset: dsl.InputPath('Dataset'),
    model_output: dsl.Output[dsl.Model]
):
    """加载处理后的数据并训练一个简单模型。"""
    import pandas as pd
    # 训练逻辑的占位符
    # 在实际场景中，您会加载数据并训练一个模型
    df = pd.read_csv(dataset)
    print(f"正在使用来自 {dataset} 的数据训练模型...")
    
    # 保存一个示例模型文件
    with open(model_output.path, 'w') as f:
        f.write("这是一个训练好的模型工件。")
    print(f"模型已保存到 {model_output.path}")
```

接下来，您使用 `@dsl.pipeline` 装饰器定义管道本身。在此函数内部，您实例化您的组件，并通过将一个任务的输出作为输入传递给另一个任务来将它们连接起来。

```python
# 定义管道结构
@dsl.pipeline(
    name='简单训练管道',
    description='一个演示管道，用于预处理数据和训练模型。'
)
def my_first_pipeline(data_url: str):
    # 实例化第一个任务
    preprocess_task = preprocess_data(raw_data_path=data_url)

    # 第二个任务使用第一个任务的输出
    train_task = train_model(
        dataset=preprocess_task.outputs['processed_data']
    )
    
    # 您还可以为特定步骤指定资源请求
    train_task.set_cpu_limit('2').set_memory_limit('4G').add_node_selector_constraint('cloud.google.com/gke-accelerator', 'NVIDIA_TESLA_T4')
```

在此示例中，`train_task` 依赖于 `preprocess_task`，因为它使用了后者的输出。Kubeflow 自动处理此依赖关系。注意训练步骤如何配置为请求特定资源，包括 GPU。这使您能够仅将强大且昂贵的硬件分配给需要它的步骤，从而优化资源利用率和成本。

### 为什么使用 Kubeflow 构建管道？

将 Kubeflow 整合到您的 MLOps 技术栈中，带来多项重要优势：

- **可复现性和实验追踪：** Kubeflow 记录每个管道的执行情况，包括组件版本、输入参数 (parameter)和输出工件。这为每次实验创建了可审计的记录，使结果易于复现。
- **可复用性：** 组件被设计为独立的且可复用的。您可以为数据验证或模型评估等常见任务构建一个共享组件库，供不同团队在多个项目中共同使用。
- **可扩展性和资源管理：** 通过在 Kubernetes 上运行，Kubeflow 管道继承了其可扩展性。您可以并行执行步骤，并精确控制分配给每个组件的计算资源（CPU、内存、GPU），如代码示例所示。
- **可移植性：** 使用 Kubeflow Pipelines SDK 定义的管道可以在任何兼容的 Kubernetes 集群上编译和运行，无论是在本地还是在任何主要云服务商处。这避免了厂商锁定，并提供了一致的开发和部署体验。

通过抽象底层 Kubernetes 对象，Kubeflow Pipelines 让数据科学家和机器学习 (machine learning)工程师能够专注于用 Python 定义他们的工作流逻辑，而平台则处理调度、执行和资源管理等复杂工作。这使其成为为机器学习项目带来操作规范的有效工具。

## 参考资料

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/) — The Kubeflow Authors (2024)
  提供关于 Kubeflow Pipelines 的全面指导，包括概念、组件开发以及使用 Python SDK 定义管道。
- [MLOps Engineering at Scale](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF9ohZzk9d14YfyHvMT2U5pcVucSzQCPf5SbqJonB_eTgoAR0IPQTiZN9vhtxVLtM6BnBVwYab4GxsnIo9ezxAQ7iwGFEgp3RzfX9TcLZuuASc-uIm9lJ8Pi-NJvK7yaOKIzHYOpRShav2-Z-elZxaFPMh88PQ64PrrTaOLY5LgQ==) — Olivier Grisel, Lj Miranda, and Boris Dayma (2022)
  Publisher: O'Reilly Media
  提供实施 MLOps 原则的实用指导，包括 ML 的 CI/CD、实验跟踪以及使用 Kubeflow 等工具进行管道编排。
- [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — Martin Görner, Dale Markowitz, Noah Fiedel, and Carole-Ann Matignon (2020)
  Publisher: Google Cloud
  讨论 MLOps 和机器学习持续交付的原则，为 Kubeflow 管道的实现提供了一个概念性框架。
