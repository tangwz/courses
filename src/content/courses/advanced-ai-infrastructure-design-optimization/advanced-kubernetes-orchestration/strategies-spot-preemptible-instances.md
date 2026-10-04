---
course: "advanced-ai-infrastructure-design-optimization"
chapter: "advanced-kubernetes-orchestration"
lesson: "strategies-spot-preemptible-instances"
sourceId: 6994
sourceUrl: "https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration/strategies-spot-preemptible-instances"
title: "使用竞价实例和可抢占实例的策略"
description: "将低成本竞价实例整合到机器学习工作流中的最佳实践，同时保持可靠性。"
order: 4
plots: []
sourceHash: "6bfd9ed2385fc5ad488fb04bbc91975ce0cfba0b2a321d076cc2863f8ec97a8b"
sourceCorrections: []
---

训练和大规模推理 (inference)的巨大计算需求使云计算成为任何AI预算中最大的开销项。竞价实例和可抢占实例提供了一种直接的解决方案，与按需付费价格相比，它们能以高达90%的折扣提供对未使用云资源的访问。然而，这种成本优势伴随着一个重要的操作注意事项：云提供商可以在几乎没有预警的情况下收回这些资源。一个平台不仅必须容忍这些中断，还必须设计为能够优雅地处理它们，将潜在的故障转变为例行的操作事件。

### 机器学习 (machine learning)工作流中的抢占应对

抢占是指当云提供商需要收回容量时，强制终止竞价实例。对于 AWS，这称为 Spot 实例中断；对于 GCP，是可抢占式虚拟机终止；对于 Azure，则是逐出。实例在关闭前会收到短暂通知，通常在30秒到两分钟之间。对于长时间运行的训练任务，如果任务进度没有保存，意外终止可能意味着数小时的工作丢失和开销浪费。

有效使用竞价实例的核心是构建能够承受这种不可预测性的系统。目标是使任务中断和重新调度成为一个标准化的自动化流程，而不是灾难性故障。

> 竞价实例上 Pod 的生命周期，展示了由抢占通知触发的优雅停机过程。

### 识别适用的工作负载

并非所有工作负载对中断的容忍度都相同。使用竞价实例的决定必须基于应用程序的容错特性。

**竞价实例的理想选择：**

- **带检查点的分布式训练：** 如第2章所述，设计为频繁保存状态的训练任务非常适合竞价实例。如果节点被抢占，任务可以重新调度并从上一个检查点恢复，从而最大限度地减少工作损失。
- **超参数 (parameter) (hyperparameter)调整：** 这些任务涉及运行许多独立、中短期持续时间的训练任务。单次运行的失败对整个调整过程影响很小，因为它可以简单地重新调度。
- **无状态批量推理 (inference)：** 以独立批次处理数据的任务具有高弹性。如果节点失败，未处理的批次可以轻松地重新分配给其他工作者。
- **机器学习 (machine learning) CI/CD 流水线：** 构建容器、运行集成测试以及其他流水线任务通常是幂等且无状态的，这使它们成为使用竞价实例节省成本的极佳选择。

**不适合竞价实例的选择：**

- **低延迟在线推理：** 实时服务端点有严格的可用性和延迟要求，无法容忍抢占的不可预测性。
- **有状态服务：** 平台的重要组成部分，例如 KubeFlow 控制平面、生产特征存储的在线数据库或制品库，需要按需实例的稳定性。
- **无检查点的长时间运行的单进程任务：** 任何无法保存其进度的任务都不适合，因为抢占将意味着从头开始。

### 弹性的架构模式

为了可靠地在竞价实例上运行工作负载，必须在 Kubernetes 中实施特定的架构模式。

#### 1. 使用生命周期钩子进行优雅停机

Kubernetes 提供了一个 `preStop` 生命周期钩子，它在容器终止之前执行。这是响应抢占通知的主要机制。当云提供商发出即将关闭的信号时，Kubernetes 会终止该节点上的 Pod，这反过来会触发每个容器中的 `preStop` 钩子。

训练应用程序的容器应使用此钩子触发向持久存储（如 S3 或 PVC）进行最终、快速的检查点保存，并干净地退出。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: resilient-training-pod
spec:
  containers:
  - name: trainer
    image: my-pytorch-trainer:1.0
    command: ["python", "train.py", "--resume-from-checkpoint"]
    lifecycle:
      preStop:
        exec:
          # 此脚本将模型状态保存到持久存储
          # 在容器终止之前。
          command: 
          - "/bin/sh"
          - "-c"
          - "python /app/save_checkpoint.py --path /mnt/checkpoints/final_checkpoint.pt"
    volumeMounts:
    - name: checkpoint-storage
      mountPath: /mnt/checkpoints
  volumes:
  - name: checkpoint-storage
    persistentVolumeClaim:
      claimName: training-pvc
```

#### 2. 带有污点和容忍的混合集群

一种常见且有效的策略是运行一个由按需实例和竞价实例节点组组成的混合集群。

- **按需节点组：** 一个小型、稳定的按需实例组托管重要的集群组件。这些节点标有 Kubernetes 污点（例如，`CriticalAddonsOnly=true:NoSchedule`），以阻止常规工作负载调度到它们上面。
- **竞价节点组：** 一个或多个大型、自动扩缩的竞价实例组用于机器学习 (machine learning)工作负载。这些节点被赋予不同的污点（例如，`spot=true:NoExecute`），以表明它们的瞬时性。

工作负载 Pod 必须具有相应的 `toleration` (容忍) 才能调度到竞价节点上。这确保了只有容错应用程序才能在易变、低成本的计算资源上运行。

> 一个 Kubernetes 集群架构，它将重要服务隔离到稳定的按需节点上，同时使用污点和容忍在经济高效的竞价节点组上运行可中断的训练任务。

为实施此策略，可以在 Pod 规格中结合使用污点、容忍和节点亲和性。

```yaml
# Pod 或 Deployment 规格的一部分
spec:
  # 此容忍允许 Pod 调度到带有竞价污点的节点上。
  tolerations:
  - important: "spot"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
  
  affinity:
    nodeAffinity:
      # 这优先选择调度到标记为竞价实例的节点上。
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - lifecycle
            operator: In
            values:
            - spot
      # 这阻止调度到为重要 Pod 保留的节点上。
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - CriticalAddonsOnly
            operator: DoesNotExist
```

#### 3. 实例队列多样化

竞价实例被抢占的概率因实例类型、区域和可用区而异。依赖单一实例类型会使工作负载容易受到该特定类型价格飙升或容量短缺的影响。

现代自动扩缩器如 **Karpenter** 或云原生配置如 **AWS EC2 Fleet** 应配置为从多样化的实例类型列表中选择。例如，如果任务需要一个具有16GB内存的 GPU，可以指定一系列适用的实例类型（例如，`g4dn.xlarge`、`g5.xlarge`、`p3.2xlarge`）。自动扩缩器随后将配置可用且价格最低的类型，显著增加获取容量的可能性并提升弹性。

通过结合优雅停机机制、混合集群设计和智能实例选择，可以利用竞价实例可观的成本节省，同时不影响 MLOps 平台的可靠性。这种运营规范是成熟且经济高效的 AI 基础设施的标志性特点。

## 参考资料

- [Amazon EC2 Spot Instances User Guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  提供有关使用AWS Spot实例的官方文档和最佳实践，包括中断通知和成本优化策略。
- [Configure Pods to Use a Pod-Specific Lifecycle Hook](https://kubernetes.io/docs/tasks/configure-pod-container/attach-handler-lifecycle-event/) — Kubernetes Authors (2023)
  详细说明了如何使用preStop钩子在收到抢占通知等事件时实现容器的优雅终止。
- [Taints and Tolerations / Assigning Pods to Nodes](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/) — Kubernetes Authors (2024)
  解释了Kubernetes将Pod调度到特定节点的机制，从而实现包含按需和Spot实例的混合集群。
- [Karpenter Documentation](https://karpenter.sh/docs/) — AWS, Community (2024)
  Karpenter的官方文档，Karpenter是一个开源的节点自动伸缩器，旨在实现高效且经济的Kubernetes集群供应，包括实例多样化。
