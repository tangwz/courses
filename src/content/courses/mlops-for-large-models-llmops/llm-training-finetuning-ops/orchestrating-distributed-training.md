---
course: "mlops-for-large-models-llmops"
chapter: "llm-training-finetuning-ops"
lesson: "orchestrating-distributed-training"
sourceId: 3138
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/orchestrating-distributed-training"
title: "编排分布式训练任务"
description: "管理和调度跨多个节点和加速器的复杂分布式训练任务。"
order: 1
plots: []
sourceHash: "58e5545d5dc18fe0edad435c1eb0c83afe01c3fd9c5baeba6d8492200b3226a3"
sourceCorrections: []
---

训练大型语言模型对计算资源要求极高，通常需要协调数百甚至数千个分布在多台机器（节点）上的GPU或TPU的协作配合。简单地在一台机器上运行训练脚本已不再可行。这就是分布式训练编排变得非常重要的地方。编排指的是对这些高要求的、多节点训练任务中涉及的计算资源和软件组件进行自动化配置、协调和管理。如果没有有效的编排，在长时间的训练运行中，管理大规模任务、处理不可避免的故障以及确保资源高效利用将几乎不可能实现。

### 编排器的作用

可以把编排器想象成您分布式训练交响乐的指挥家。其主要职责包括：

1. **资源分配：** 识别集群中可用的计算节点和加速器（GPU/TPU），并根据任务要求（如节点数量、每节点GPU数量）将它们分配给特定的训练任务。
2. **任务启动：** 在已分配的节点上启动训练过程（通常是容器化的）。这包括分发训练代码、设置必要的环境变量（如进程排名、大小、主地址/端口）以及执行训练脚本。
3. **工作进程协调：** 促进分布式训练过程之间的通信和同步。虽然编排器通常不处理低层次的通信本身（这通常由NCCL或MPI等库通过PyTorch或TensorFlow等框架完成），但它提供必要的信息，使工作进程能够相互发现并建立通信通道。
4. **监控与管理：** 跟踪单个工作进程和整个任务的状态。这包括检测故障、可能重启失败的工作进程，并提供资源使用和任务进展的可见性。
5. **生命周期管理：** 处理任务的整个生命周期，从提交和排队（如果资源繁忙）到执行、完成或终止。

> 一个由编排器管理的典型分布式训练配置。编排器接收任务规范，分配资源（节点和GPU），启动训练过程，并促进工作进程间的通信设置。

### 编排框架与工具

有多种工具和平台可用于编排分布式训练任务，每种都有其优点和典型用途：

- **Kubernetes：** 一个流行的开源容器编排系统。虽然它不是专为机器学习 (machine learning)设计的，但其可扩展性使其适用于管理大型语言模型训练任务。像Kubeflow（特别是其训练操作符）或使用Kubernetes Jobs和StatefulSets构建的定制解决方案，允许您以声明方式定义和管理分布式训练工作负载。Kubernetes擅长管理容器化应用、处理网络，并提供容错机制。然而，为大型语言模型所需的GPU调度和高吞吐、低延迟网络对其进行优化配置，可能需要丰富的经验。
- **Slurm Workload Manager：** 高性能计算（HPC）环境中广泛使用的开源任务调度器。如果您的组织使用传统的HPC集群，那么Slurm很可能是您会接触到的编排器。它在管理大量任务、精确分配资源（节点、核心、GPU、内存）以及处理集群内的优先级和分区方面效率很高。启动分布式任务通常涉及提交批处理脚本（`sbatch`），这些脚本指定资源要求，并使用`srun`等工具在已分配的节点上启动进程。
- **云原生托管服务：** 主要云提供商提供托管服务，旨在简化分布式训练：
  - **AWS SageMaker：** 提供托管训练任务，负责基础设施供应、扩展和任务编排。您配置任务参数 (parameter)（实例类型、数量、训练脚本位置），SageMaker管理执行。
  - **Azure Machine Learning：** 通过其`CommandJob`或`SweepJob`功能提供类似的能力，允许您定义分布式训练配置（例如PyTorchDistribution、MpiDistribution）并在托管计算集群上运行它们。
  - **Google Cloud AI Platform Training：** 支持定制训练任务，您在其中指定机器类型、加速器配置和容器镜像，让Google Cloud处理底层的编排工作。
    这些服务抽象了大部分底层复杂性，但可能不如管理自己的Kubernetes或Slurm集群那样灵活。
- **专用框架调度器：** 一些分布式训练框架，特别是那些源自大型科技公司的，可能带有自己的集成或偏好的调度器，尽管这些调度器通常是建立在或集成于上述系统之上的。

### 实际中的核心编排任务

无论使用何种特定工具，从MLOps的角度来看，编排大型语言模型训练任务通常涉及以下步骤：

1. **定义任务规范：** 您需要清楚地定义训练运行的要求。这包括：

   - 训练代码（通常打包在容器镜像中）。
   - 所需的节点数量。
   - 每节点加速器（GPU/TPU）的数量和类型。
   - 资源要求，如CPU、RAM以及可能的网络带宽。
   - 依赖项（例如，数据集、基础模型检查点）。
   - 分布式设置所需的环境变量（通常由编排器注入）。
   - 用于访问私有容器注册表或数据存储等资源的密钥管理。
2. **提交任务：** 任务规范被提交给编排器（例如，使用Kubernetes的`kubectl apply`、Slurm的`sbatch`，或云服务的SDK调用）。如果资源不可用，编排器会将任务放入队列；否则，立即开始分配过程。
3. **资源供应与启动：** 编排器分配请求的节点和GPU。然后，它通常会将指定的容器镜像拉取到每个节点上并启动训练过程。关键在于，它会注入环境变量，使每个进程都能识别其在分布式设置中的角色。常见的变量包括：

   - `WORLD_SIZE`：任务中的进程总数。
   - `RANK`：当前进程的唯一ID（从0到`WORLD_SIZE - 1`）。
   - `LOCAL_RANK`：*本地*节点上进程的唯一ID。
   - `MASTER_ADDR`：排名为0的进程的IP地址或主机名。
   - `MASTER_PORT`：排名为0的进程用于协调的监听端口号。
4. **执行与监控：** 训练脚本使用注入的环境变量来初始化分布式通信后端（例如PyTorch中的`torch.distributed.init_process_group`）。编排器监控pod/节点的健康状况。与日志和监控系统（第5章讨论）的集成在这里非常重要，以便跟踪所有工作进程的进展、性能指标（如吞吐量 (throughput)和GPU利用率）以及潜在错误。
5. **处理故障：** 长时间的大型语言模型训练运行容易受到硬件故障或瞬时问题的影响。编排器，通常与训练框架的检查点机制（本章稍后讨论）结合使用，需要优雅地处理这些故障。这可能包括自动重新调度失败的工作进程，或提供从上次检查点重启整个任务的机制。*弹性训练*等功能允许任务即使在某些工作进程失败的情况下也能继续，动态调整`WORLD_SIZE`，尽管这会增加复杂性。

### 大型语言模型训练编排中的挑战

编排大型语言模型训练面临独特的挑战：

- **规模和资源异构性：** 管理数百或数千个GPU，可能跨越不同硬件代次或类型，需要复杂的调度和分配逻辑。
- **网络敏感性：** 分布式训练，特别是模型并行，对网络延迟和带宽非常敏感。编排器和底层基础设施必须支持高速互连（如NVLink，InfiniBand）。
- **依赖项管理：** 确保所有节点都具有正确版本的驱动、库（CUDA、cuDNN）、Python包和训练代码容器可能很复杂。
- **成本管理：** GPU成本高昂。需要高效的编排来最大限度地提高利用率，避免为闲置资源付费。这包括有效的排队、可能使用竞价实例（并具备适当的容错能力），以及合理调整资源请求大小。
- **与MLOps工具的集成：** 编排器需要与其他MLOps组件（如实验追踪、工件存储库和监控系统）顺畅集成。

成功编排分布式训练任务是大型模型开发的实际部署的基本要求。它将您的训练代码与复杂的硬件基础设施连接起来，使您能够有效地管理资源，确保可靠的执行，并最终以达到领先性能所需的规模训练模型。在研究这些编排所支持的特定并行策略之前，了解这些编排原理是必要的。

## 参考资料

- [Kubeflow Training Operator Documentation](https://www.kubeflow.org/docs/components/training/) — The Kubeflow Authors (2025)
  关于使用Kubeflow训练算子在Kubernetes集群上管理和运行分布式机器学习训练任务（包括PyTorch和TensorFlow）的官方指南。
- [torch.distributed (Distributed communication package)](https://pytorch.org/docs/stable/distributed.html) — PyTorch Contributors (2017)
  PyTorch的官方文档，详细介绍了其分布式通信包，这对在分布式训练中设置和协调工作进程至关重要。
- [Slurm Workload Manager Documentation](https://slurm.schedmd.com/documentation.html) — SchedMD LLC (2025)
  Publisher: SchedMD LLC
  Slurm工作负载管理器的官方综合文档，描述了其在HPC环境中用于作业调度、资源管理和分布式进程启动的功能。
- [Custom training overview (Google Cloud AI Platform Training)](https://cloud.google.com/ai-platform/training/docs/custom-training-overview) — Google Cloud (2024)
  Publisher: Google Cloud
  Google Cloud自定义模型训练托管服务的概述，展示了云平台如何抽象分布式机器学习任务的基础设施和编排。
