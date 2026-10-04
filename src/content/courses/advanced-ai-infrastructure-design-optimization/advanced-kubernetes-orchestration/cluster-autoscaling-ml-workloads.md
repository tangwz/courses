---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "advanced-kubernetes-orchestration"
lesson: "cluster-autoscaling-ml-workloads"
sourceId: 6992
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration/cluster-autoscaling-ml-workloads"
title: "面向动态机器学习工作负载的集群自动扩缩容"
description: "配置 Kubernetes 集群自动扩缩容器，以有效管理应对变化的训练和推理需求的计算资源。"
order: 3
plots: []
sourceHash: "cf798273f941c1ed04c3baf17d2f6588e2400d720e37eafe474ff33b98a882cc"
sourceCorrections: []
---

机器学习 (machine learning)工作负载本身是动态变化的。一个大规模训练任务可能需要数十个GPU运行数小时或数天，此后这些资源就会闲置。反之，一个新部署的推理 (inference)服务可能突然面临流量激增，需要即刻增加服务能力。在这种环境下管理静态的计算节点集群会导致一个艰难的抉择：过度配置资源并为闲置容量承担高成本，或者配置不足并面临任务在队列中无限期等待的风险。

Kubernetes 集群自动扩缩容器直接应对了这一难题，它能自动调整集群中的节点数量。它监控因资源限制而无法调度的Pod，并添加新节点以容纳它们。它还会整合工作负载并移除未充分利用的节点，提供一种弹性基础设施机制，使资源供给与应用需求匹配。

### 自动扩缩容机制

集群自动扩缩容器采用一个简单而有效的控制循环运行。它扩容的主要触发条件是Pod处于`Pending`（待定）状态，并带有特定的事件原因：`FailedScheduling`（调度失败）。此事件表示Kubernetes调度器未找到任何具有足够可用资源（如CPU、内存或GPU）的现有节点来运行该Pod。

当集群自动扩缩容器检测到此类Pod时，它会评估已配置的节点池。节点池是由云提供商管理（例如 AWS 自动扩缩组、GCP 托管实例组）的具有相同实例类型的节点组。它模拟将待定Pod调度到来自每个适用池的新节点上，并选择一个能够满足Pod资源请求的池。

缩容由利用率驱动。自动扩缩容器定期检查所有节点的资源使用情况。如果一个节点的利用率在持续一段时间内低于配置的阈值（通常约为50%），并且其上运行的所有Pod都可以安全地重新调度到其他地方，则该节点成为移除候选。自动扩缩容器随后会排空节点，优雅地驱逐Pod，并终止底层云实例。

### 配置对GPU感知的自动扩缩容

通用自动扩缩容配置不足以满足机器学习 (machine learning)平台的需求。我们必须使其感知GPU等专用硬件，并确保它能明智地决定应配置哪种类型的节点。

#### 多节点池

标准做法是根据硬件能力将集群划分为多个节点池。例如，您可能拥有：

- 一个`cpu-general`池，包含经济高效、CPU优化的实例，用于通用工作负载和系统组件。
- 一个`gpu-training-a100`池，包含强大的NVIDIA A100 GPU，用于大规模训练。
- 一个`gpu-inference-t4`池，包含NVIDIA T4 GPU，这些GPU针对成本效益和低延迟推理 (inference)进行了优化。

当一个训练Pod请求A100 GPU时，自动扩缩容器将识别出只有`gpu-training-a100`池可以满足此请求，并会扩容该特定池。

> GPU请求Pod的自动扩缩容过程。集群自动扩缩容器识别待定Pod，并从正确的配备GPU的节点池中配置一个新节点。

#### 污点、容忍度与亲和性

为了强制执行这种隔离，并防止非GPU工作负载占用昂贵的GPU节点，我们使用污点和容忍度。您应该为GPU池中的所有节点应用一个污点。

例如，A100池中的节点可以被`nvidia.com/gpu=present:NoSchedule`污点标记 (token)。此污点会阻止任何Pod调度到该节点上，除非该Pod具有匹配的容忍度。

A训练Pod会需要一个A100 GPU，它随后会在其规范中包含此容忍度，同时还有`nodeSelector`或`nodeAffinity`规则来明确请求正确的硬件。

这是一个设计用于在带有污点的A100节点上运行的Pod的清单片段：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: large-model-training
spec:
  containers:
  - name: training-container
    image: my-registry/pytorch-a100:latest
    resources:
      limits:
        nvidia.com/gpu: 1 # 这请求一个GPU
  tolerations:
  - important: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-a100 # GKE的示例
```

这种组合确保：

1. Pod明确请求GPU资源。
2. Pod可以调度到为GPU使用而设置污点的节点上。
3. Pod被引导至具有特定A100加速器类型的节点。

当集群自动扩缩容器看到这个待定Pod时，它知道它必须扩容一个能够提供匹配所有这些条件的节点的节点池。

### 调优和运维考虑

虽然核心机制简单明了，但生产环境需要仔细调优。

- **扩展器（Expanders）**：集群自动扩缩容器可以配置扩展器策略，以决定在有多个选项可用时扩容哪个节点池。常见策略包括`least-waste`（最少浪费），它选择在Pod调度后空闲CPU或内存最少的池；以及`priority`（优先级），它使用用户定义的节点池优先级列表。对于机器学习 (machine learning)，`least-waste`通常是一个很好的起点，可以提高成本效益。
- **扩容延迟**：配置新的云虚拟机，特别是带有GPU的虚拟机，并非瞬时完成。这可能需要几分钟。对于对启动时间敏感的工作负载，您可以使用`cluster-overprovisioner`（集群预留器）。这涉及部署低优先级的“暂停”Pod来预留资源。当高优先级的机器学习Pod到达时，它会抢占一个暂停Pod，该Pod随后被重新调度，从而在后台触发自动扩缩容器添加新节点。这会保持一个“热”容量缓冲区。
- **缩容行为**：`--scale-down-unneeded-time`标志决定一个节点在被考虑终止之前必须未充分利用多长时间。较短的持续时间（例如5分钟）可以优化成本，但如果需求波动迅速，可能导致集群“抖动”。较长的持续时间（例如20分钟）可以提供更高的稳定性，但成本也更高。此外，您必须在缩容期间保护某些Pod不被驱逐。为关键服务使用`PodDisruptionBudget`（PDB，Pod中断预算）可确保自动扩缩容器不会移除节点，如果这样做会违反预算（例如，使服务的可用副本数量低于所需的最低限度）。

通过将集群自动扩缩容器与精心设计的节点池和调度原语结合，您可以构建一个真正弹性的基础设施平台。该系统可以在需要时动态提供昂贵的GPU资源，并在不需要时将其释放，这成为一个经济高效且响应迅速的AI平台的重要组成部分。

## 参考资料

- [Cluster Autoscaler](https://github.com/kubernetes/autoscaler/blob/master/cluster-autoscaler/README.md) — Kubernetes Authors (2023)
  Kubernetes 集群自动扩缩器（Cluster Autoscaler）的官方文档，详细介绍了其架构、配置和操作指南。
- [Running GPU workloads on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) — Google Cloud Documentation (2023)
  提供了在 Google Kubernetes Engine 中配置和自动扩缩支持 GPU 的节点池的实用指导，与本节中关于 GPU 感知自动扩缩的内容直接相关。
