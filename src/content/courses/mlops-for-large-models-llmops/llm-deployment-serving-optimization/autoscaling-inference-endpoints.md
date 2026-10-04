---
course: "mlops-for-large-models-llmops"
chapter: "llm-deployment-serving-optimization"
lesson: "autoscaling-inference-endpoints"
sourceId: 3181
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/autoscaling-inference-endpoints"
title: "自动扩缩容推理端点"
description: "根据负载和性能指标配置大型语言模型推理端点的自动扩缩容。"
order: 7
plots: []
sourceHash: "6dd53d863a2677dcf2040522db913ffcfa94bfcf5ef643ad5d1d3d514cb200b6"
sourceCorrections: []
---

部署大型语言模型不仅仅是让模型可用；它还需要高效应对波动的需求。大型语言模型的推理 (inference)工作负载通常表现出很大的变化性。您可能会在业务高峰时段看到高流量，之后是低谷，或者因批量处理任务或意外用户活动而突然激增。为峰值负载静态配置资源成本高昂，在低谷期导致GPU闲置和预算浪费。反之，资源配置不足则导致用户体验不佳，因为高延迟甚至请求丢失，无法达到服务水平目标（SLO）。

自动扩缩容提供了一个动态方案，根据实时需求自动调整分配给推理端点的计算资源。对于大型语言模型服务，这通常指水平扩缩GPU加速实例或Pod的数量（扩容以添加更多副本，缩容以移除副本）。目标是保持所需的性能水平，同时尽量降低运营成本。

### 大型语言模型推理 (inference)的自动扩缩容要点

推理端点自动扩缩容的基本原理简单明了：监测重要指标并相应调整处理单元的数量。当负载增加或性能下降时，添加更多副本；当负载减少时，移除闲置副本。

- **资源：** 主要扩缩资源是运行大型语言模型推理服务的计算实例或容器Pod，通常配备一个或多个GPU。
- **扩缩维度：** 水平扩缩（调整副本数量）是无状态推理工作负载的标准方法。每个副本独立处理传入请求。垂直扩缩（增加现有实例的CPU/内存/GPU资源）对于实时推理不太常见，因为调整实例大小具有中断性，但在特定情况下可能会考虑。
- **机制：** 自动扩缩控制器持续监测选定指标，将其与预设阈值比较，并与底层平台（如Kubernetes或云提供商的基础设施）交互以修改副本数量。

### 驱动自动扩缩容决策的指标

为大型语言模型端点选择合适的指标或指标组合，对于有效的自动扩缩容很重要。传统网络服务常用的CPU或内存利用率等标准指标，在这里常常不够用。大型语言模型推理 (inference)任务可能在CPU负载高之前就已经使GPU饱和。

- **GPU利用率：** 这通常是GPU密集型推理任务工作负载最直接的指示。监测GPU计算或内存利用率的百分比，清晰显示硬件加速器的工作状态。可以设定目标利用率（例如70-80%）来触发扩缩容操作。需要像NVIDIA数据中心GPU管理器（DCGM）导出器这样的工具，将这些指标暴露给Prometheus等监测系统，然后这些系统可以为自动扩缩器提供数据。
- **推理延迟：** 根据请求延迟（例如p95或p99延迟）进行扩缩容直接针对用户体验和SLO。如果处理请求的时间超过阈值，系统就会扩容。尽管以用户为中心，但延迟可能是一个滞后指标，意味着扩缩容可能在性能已经下降后才发生。
- **请求速率（RPS/QPS）：** 根据每秒传入请求数量进行扩缩容是直接的。然而，大型语言模型请求的复杂度差异很大（例如，提示长度、所需输出长度），这意味着单独的RPS可能无法准确反映GPU的实际负载。短而简单的请求即使RPS很高，其要求可能也低于RPS较低但生成内容复杂冗长的请求。
- **队列深度：** 如果请求在推理工作器处理前排队，则队列长度是过载的有力指示。根据队列深度扩缩容可确保有足够的处理能力及时处理传入工作负载。
- **每秒令牌数：** 对于生成式大型语言模型，令牌生成速率是一个有用的指标。所有副本的每秒聚合令牌数下降可能表示饱和，需要扩容。

通常，指标组合能带来最有效的扩缩容表现。例如，您可以使用GPU利用率作为主要扩缩指标，但也要配置最大延迟阈值，以确保即使平均利用率看起来可以接受，也能满足SLO。

> 典型的自动扩缩容设置包括负载均衡器将请求分发给推理Pod，指标收集器从Pod收集性能数据（如GPU利用率和延迟），以及自动扩缩器根据这些指标调整Pod的数量。

### 自动扩缩容平台和工具

有多种平台和工具可以为大型语言模型端点实现自动扩缩容：

- **Kubernetes水平Pod自动扩缩器（HPA）：** Kubernetes中的标准机制。HPA根据观测到的指标自动调整Deployment、ReplicaSet或StatefulSet中的Pod数量。
  - **v1/v2beta1：** 基于CPU和内存扩缩容（通常不足以满足大型语言模型的需求）。
  - **v2/v2beta2及更高版本：** 支持基于自定义和外部指标的扩缩容（例如，通过Prometheus Adapter暴露的GPU利用率，或延迟指标）。需要指标服务器，并可能需要自定义指标的适配器。
- **KEDA（Kubernetes事件驱动自动扩缩容）：** 一个日益流行的CNCF项目，扩展了Kubernetes的自动扩缩容能力。KEDA可以根据各种事件源和指标提供者扩缩工作负载（例如，Prometheus查询，SQS/Azure队列等云提供商队列，Kafka主题滞后，自定义指标端点）。它擅长根据外部因素和队列长度进行扩缩容，并且还可以处理缩容到零副本的情况。
- **云提供商自动扩缩容：**
  - **AWS：** 自动扩缩组（ASG）可以扩缩EC2实例。您可以根据CloudWatch指标配置扩缩策略，包括从GPU监测工具推送的自定义指标（例如，使用带DCGM输入的CloudWatch代理）。对于EKS（Kubernetes），通常使用HPA或KEDA，可能与集群自动扩缩器结合使用来管理底层EC2节点。
  - **Azure：** 虚拟机规模集（VMSS）根据Azure Monitor指标提供实例级别的扩缩容（如果配置，包括GPU利用率等访客操作系统指标）。对于AKS（Kubernetes），HPA/KEDA很常见，并与Azure集群自动扩缩器集成。
  - **GCP：** 托管实例组（MIG）根据Cloud Monitoring指标（包括GPU指标）扩缩Compute Engine虚拟机。对于GKE（Kubernetes），HPA/KEDA与GKE集群自动扩缩器配合使用。
- **专用服务平台：** KServe（原KFServing）、Ray Serve、BentoML等框架或商业平台通常提供更高级别的抽象，用于部署和管理模型，包括为机器学习 (machine learning)工作负载专门设计、集成且可能更优化的自动扩缩容功能。

### 大型语言模型自动扩缩容的特有挑战

尽管理念简单，大型语言模型自动扩缩容也带来了一些具体挑战：

- **冷启动和模型加载时间：** 大型语言模型很大。将一个拥有数十亿参数 (parameter)的模型加载到GPU上可能需要几十秒到几分钟。扩容时，由于加载时间，新的Pod/实例无法立即处理请求。这种延迟会显著影响系统快速响应突发流量激增的能力。
  - *应对方案：* 保持最小副本数（`minReplicas > 0`）以避免完全冷启动。使用仅在模型加载后才通过的就绪探针。优化模型加载（例如，更快的序列化格式，并行加载）。如果负载模式可预测，考虑预测性自动扩缩容。
- **缩容到零的考量：** 缩容到零副本在闲置期间具有吸引力，尤其对于KEDA等工具。然而，它保证了缩容后的第一个请求可能面临较长的冷启动。这种成本与首个请求延迟之间的权衡必须根据应用需求仔细评估。
- **GPU粒度和成本：** GPU是独立的、昂贵的资源。扩缩容决策是增加或移除整个GPU实例/分配。扩容时超出所需容量可能代价高昂。应通过调整稳定窗口和冷却期来避免抖动（快速扩容和缩容）。
- **调优复杂度：** 找到最佳的自动扩缩容配置需要仔细调优和实践。设置合适的指标阈值（例如，目标GPU利用率，延迟SLO），定义扩缩速度（一次添加/移除多少副本），以及配置冷却/稳定期（防止抖动）是一个迭代过程。部署后密切监测自动扩缩器的行为。
- **请求复杂度不一致：** 如前所述，每个请求的资源成本可能显著不同。为平均负载调优的自动扩缩器可能难以应对突发的高复杂度请求。这再次说明了基于延迟的扩缩容或更精细指标的潜在需求。

### 示例：Kubernetes HPA用于GPU利用率

让我们通过一个针对平均GPU利用率的Kubernetes HPA (v2) 清单来说明。这假设您有一个指标管道（例如，DCGM-Exporter -> Prometheus -> Prometheus Adapter），将GPU指标提供给Kubernetes API。

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
  namespace: llm-serving
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment # 或者 ReplicaSet, StatefulSet
    name: llm-inference-deployment
  minReplicas: 2 # 保持至少2个副本以缓解冷启动
  maxReplicas: 10 # 设置成本控制的上限
  metrics:
  - type: Pods # 使用Pod级别的指标
    pods:
      metric:
        name: dcgm_gpu_utilization # 通过指标适配器暴露的指标名称
      target:
        type: AverageValue # 目标是Pod的平均利用率
        averageValue: 75 # 目标是75%的GPU利用率（根据需要调整）
  behavior: # 可选：微调扩缩容速度和稳定性
    scaleUp:
      stabilizationWindowSeconds: 60 # 上次扩容后等待60秒
      policies:
      - type: Percent
        value: 50 # 增加50%的Pod
        periodSeconds: 30
      - type: Pods
        value: 2 # 或者至少增加2个Pod
        periodSeconds: 30
      selectPolicy: Max # 使用增加更多Pod的策略
    scaleDown:
      stabilizationWindowSeconds: 300 # 上次缩容后等待5分钟
      policies:
      - type: Pods
        value: 1 # 每次移除1个Pod
        periodSeconds: 60
```

> 这个Kubernetes HPA示例清单定义了大型语言模型部署的自动扩缩容规则。它旨在使Pod的平均GPU利用率达到75%，保持2到10个副本。它还包含行为策略，用于控制扩容和缩容的速度和稳定性。

### 总结

自动扩缩容是一项不可或缺的技术，用于高效可靠地运行大型语言模型推理 (inference)端点。通过根据观测到的负载或性能指标（如GPU利用率或延迟）动态调整计算资源，您可以在满足性能SLO和管理GPU基础设施相关的高昂成本之间取得平衡。理解这些挑战，特别是模型加载时间以及选择合适的指标，以及运用Kubernetes HPA、KEDA或云原生服务等工具，是成功进行大型语言模型运维的基本技能。仔细调优和持续监测自动扩缩器的性能和成本影响是必要的，以优化生产环境中的大型语言模型服务。

## 参考资料

- [Horizontal Pod Autoscaler](https://kubernetes.io/docs/tasks/run-applications/horizontal-pod-autoscaling/) — Kubernetes Documentation (2025)
  解释Kubernetes Horizontal Pod Autoscaler核心原理和配置的官方文档，包括对LLM服务至关重要的自定义和外部指标。
- [KEDA Documentation](https://keda.sh/docs/) — KEDA Community (2024)
  KEDA（Kubernetes 事件驱动自动扩缩容）的官方文档，详细介绍了其基于各种事件源进行扩缩容的能力，以及实现零副本扩缩容以优化成本。
- [What is Amazon EC2 Auto Scaling?](https://docs.aws.amazon.com/autoscaling/ec2/userguide/what-is-amazon-ec2-auto-scaling.html) — Amazon Web Services (2024)
  解释主要云计算环境中动态资源调整基本概念的官方文档，适用于管理用于LLM服务的GPU实例。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  本书提供了一份构建生产级机器学习系统的全面指南，包括部署、监控和高效扩展模型的策略。
