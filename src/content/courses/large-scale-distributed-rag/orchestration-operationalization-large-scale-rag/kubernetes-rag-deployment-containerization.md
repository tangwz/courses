---
course: "large-scale-distributed-rag"
chapter: "orchestration-operationalization-large-scale-rag"
lesson: "kubernetes-rag-deployment-containerization"
sourceId: 6218
sourceUrl: "https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag/kubernetes-rag-deployment-containerization"
title: "RAG 部署的容器化与 Kubernetes 应用"
description: "运用 Docker 和 Kubernetes 进行大规模 RAG 应用的打包、部署和管理。"
order: 3
plots: []
sourceHash: "84c83b2268b1bd4096c9c7eb4eebb301682fa97380bbcbab046e06ee6d27ea40"
sourceCorrections: []
---

为了稳妥部署 RAG 组件，容器化和编排变得不可或缺。分布式 RAG 系统，可能由用于检索、生成、数据处理和编排的多个微服务构成，需要一个一致、可扩展且易于管理的运行环境。Docker 用于容器化、Kubernetes 用于编排提供了行业标准方案。使用这些技术，可以将 RAG 组件及其依赖项打包，在不同环境中可靠地部署它们，并根据需求动态扩展，同时与大规模 RAG 的 MLOps 实践集成。

### 为什么要对 RAG 组件进行容器化？

容器化，主要是通过 Docker 实现，为 RAG 系统这类复杂应用提供多项优势：

1. **环境一致性**：每个 RAG 组件（例如，检索器 API、LLM 服务、嵌入 (embedding)生成器）都有其自己的依赖项集合、Python 版本、`transformers` 或 `faiss` 等特定库、系统工具，以及用于 GPU 加速任务的 CUDA 版本。Docker 将每个组件及其依赖项封装成一个便携式镜像。这个镜像在开发者的笔记本电脑、测试服务器或生产 Kubernetes 集群中运行完全相同，消除了“在我机器上能跑”的问题。
2. **隔离性**：不同的 RAG 组件可能需要冲突的库版本。容器将这些组件隔离，使每个组件都能在其特定环境中运行而不受干扰。
3. **微服务支持**：容器是微服务的天然打包单位。一个检索器、一个重排器和一个 LLM 都可以是一个独立的容器，可以独立开发、部署和扩展。
4. **资源效率**：与传统虚拟机相比，容器更轻量，共享宿主操作系统的内核。这导致更快的启动时间和更优的资源使用，使得更多 RAG 服务实例可以在相同的硬件上运行。

### 将 RAG 微服务 Docker 化

为您的 RAG 服务创建 Docker 镜像需要为每个服务编写一个 `Dockerfile`。我们来看几个典型的 RAG 组件：

- **检索器服务**：这可能是一个 FastAPI 或 Flask 应用程序，提供向量 (vector)搜索查询服务。其 `Dockerfile` 将指定一个 Python 基础镜像，复制应用程序代码，安装 `requirements.txt` 中的依赖项（包括向量数据库客户端），并定义启动 API 服务器的命令。
- **LLM 服务端点**：为了部署 LLM，您可以使用 vLLM、Text Generation Inference (TGI) 或 Triton Inference Server 等专用服务框架。这些框架通常提供基础 Docker 镜像或清晰的容器化说明。GPU 访问是一个主要考量；您的 Docker 设置和后续的 Kubernetes 部署需要考虑 NVIDIA 驱动和 Docker 运行时。
- **数据摄入/嵌入 (embedding)工作器**：这些组件可能处理文档、生成嵌入，并更新您的向量数据库。它们可以被打包为容器，设计用于作为批处理作业或消息队列的长期消费者运行。

一个常见做法是在 `Dockerfile` 中使用多阶段构建，通过排除构建时依赖项或中间文件来保持最终镜像的大小小巧和安全。

```dockerfile
# 示例：基于 Python 的检索器服务 Dockerfile（简化版）
# 阶段 1：构建阶段（如果需要编译或资产构建）
FROM python:3.10-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# 如果有必要，在此处添加任何构建步骤

# 阶段 2：最终阶段
FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /app /app
# 为安全起见，确保使用非 root 用户
RUN useradd -ms /bin/bash appuser
USER appuser
EXPOSE 8000
CMD ["python", "main_retriever.py"]
```

### 使用 Kubernetes 编排 RAG

Docker 负责打包您的 RAG 组件，而 Kubernetes 则负责编排它们。对于大型分布式 RAG 系统，Kubernetes 提供：

- **自动化部署与扩展**：定义您的 RAG 服务的期望状态，Kubernetes 会努力保持该状态。它可以自动扩展或缩减服务。
- **服务发现与负载均衡**：Kubernetes 为服务分配稳定的 IP 地址和 DNS 名称，使得 RAG 组件即使在 Pod 被创建或销毁时也能可靠地相互发现和通信。
- **自我修复**：Kubernetes 可以重启失败的容器，替换不健康的 Pod，并将它们重新调度到健康的节点上，从而提高系统的韧性。
- **配置与秘密管理**：将应用程序配置和敏感数据（如 API 密钥）与容器镜像分开管理。
- **资源管理**：为您的 RAG 组件定义 CPU、内存和 GPU 的请求与限制，以确保公平的资源分配和系统稳定性。

用于部署 RAG 系统的 Kubernetes 对象包括：

- **Pods**：最小的可部署单元，通常为单个 RAG 微服务（例如，检索器 Pod、LLM Pod）托管一个容器。
- **Deployments**：管理无状态应用程序，如您的检索器 API 或 LLM 服务。它们处理声明式更新、回滚，并确保指定数量的副本 (Pod) 正在运行。
- **Services**：将您的 RAG 组件作为具有稳定端点的网络服务暴露。例如，用于编排器和检索器之间内部通信的 `ClusterIP` 服务，或用于将您的 RAG API 外部公开的 `LoadBalancer` 服务。
- **ConfigMaps 和 Secrets**：安全地存储配置数据（例如，模型名称、检索参数 (parameter)）和敏感信息（例如，数据库凭据、外部服务的 API 密钥）。
- **HorizontalPodAutoscaler (HPA)**：根据观察到的指标（如 CPU 利用率、GPU 利用率（针对 LLM））或自定义指标（如每秒查询数 (QPS) 或消息队列长度）自动扩展 Deployment 中 Pod 的数量。
- **PersistentVolumes (PVs) 和 PersistentVolumeClaims (PVCs)**：虽然大多数 RAG 组件目标是无状态，但如果任何部分需要持久化数据（例如，本地缓存、用于批处理的临时存储），PVs 和 PVCs 会管理该存储。

下面是典型 RAG 系统在 Kubernetes 上部署的图示：

> Kubernetes 集群中编排的 RAG 组件。用户查询通过 API 网关路由到编排器 Pod，后者再与检索器和 LLM 服务通信。每个服务组件都由 Kubernetes Deployment 管理，并可使用 HPA 进行自动扩展。外部服务（如向量 (vector)数据库和模型存储）由相应的 Pod 访问。

### 在 Kubernetes 中定义 RAG 服务：一个示例

检索器服务的 Kubernetes `Deployment` YAML 文件可能如下所示（简化版）：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retriever-deployment
  labels:
    app: rag-retriever
spec:
  replicas: 3 # Start with 3 replicas, HPA can adjust this
  selector:
    matchLabels:
      app: rag-retriever
  template:
    metadata:
      labels:
        app: rag-retriever
    spec:
      containers:
      - name: retriever
        image: your-repo/rag-retriever-service:v1.2.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        envFrom:
        - configMapRef:
            name: retriever-config
        - secretRef:
            name: vector-db-credentials
---
apiVersion: v1
kind: Service
metadata:
  name: retriever-service
spec:
  selector:
    app: rag-retriever
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP # Internal service
```

此清单定义了检索器的 Deployment，指定了容器镜像、端口、资源请求/限制，以及对 `ConfigMap` 和 `Secret` 的引用以进行配置。相应的 `Service` 使其在集群内可被发现。

### RAG 组件的资源管理

有效的资源管理对于性能和成本效益非常重要：

- **CPU 和内存**：为每个 RAG 组件定义 `requests`（保障资源）和 `limits`（最大可分配资源）。检索器在查询处理或重排过程中可能受限于 CPU。LLM 编排器和 API 网关也需要适当的 CPU/内存。
- **GPU 管理**：LLM 推理 (inference)通常是 GPU 密集型的。Kubernetes 通过设备插件（例如 NVIDIA 设备插件）支持 GPU 调度。您将在 LLM 服务器 Deployment 中指定 GPU `requests` 和 `limits`（通常是 `nvidia.com/gpu: 1`）。

  ```yaml
  # Snippet for GPU resources in an LLM pod spec
  resources:
    limits:
      nvidia.com/gpu: 1
    requests:
      nvidia.com/gpu: 1
  ```
- **节点亲和性与污点/容忍度**：使用这些功能来确保特定的 RAG 组件在适当的硬件上运行。例如，使用节点标签和亲和性规则将 LLM Pod 专门调度到带有 GPU 的节点上。污点可以阻止通用工作负载在昂贵的 GPU 节点上运行。

### 在 Kubernetes 上扩展 RAG

Horizontal Pod Autoscaler (HPA) 对于处理可变负载非常必要：

- **检索器 HPA**：可以根据 CPU 利用率或自定义指标（如每秒查询数 (QPS) 或平均查询延迟）进行扩展。
- **LLM 服务器 HPA**：扩展 LLM 服务器通常涉及 GPU 利用率指标（如果通过自定义指标适配器提供）或 QPS。对于令牌流式 LLM，管理并发连接也是一个考量因素。
- **摄入工作器 HPA**：如果您的数据摄入管道使用 Kubernetes（例如，处理来自 Kafka 消息的工作器），HPA 可以根据队列长度（Kafka 延迟）来扩展它们。

检索器服务的 HPA：

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: retriever-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: retriever-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  # - type: Pods # 自定义指标示例（QPS）
  #   pods:
  #     metric:
  #       name: qps_per_pod
  #     target:
  #       type: AverageValue
  #       averageValue: "10" # 目标每个 Pod 10 QPS
```

除了 Pod 自动扩展之外，**集群自动扩展器**（云提供商特有的组件）可以根据整体资源需求从您的集群中添加或删除节点，确保您的 RAG 系统拥有所需的底层基础设施。

### 滚动更新与健康检查

Kubernetes Deployment 支持滚动更新，让您可以零停机地将 RAG 组件更新到新版本。通过逐步替换旧 Pod 为新 Pod 并监控它们的健康状况，您可以最大程度地减少服务中断。为您的 RAG 组件 Pod 配置 `readinessProbes` 和 `livenessProbes`。

- **活跃度探测**：Kubernetes 定期检查此探测。如果它失败，Kubernetes 会重启容器。这有助于从死锁或无响应状态中恢复。
- **就绪度探测**：Kubernetes 检查此探测以确定 Pod 是否准备好接收流量。新 Pod 仅在其就绪度探测成功后才会接收流量。这在滚动更新期间很重要。

### Kubernetes 上 RAG 的高阶考量

对于高度成熟的 RAG 部署，您可以考虑：

- **服务网格（例如 Istio, Linkerd）**：这些在您的 RAG 微服务之间提供高级流量管理（例如，金丝雀部署、A/B 测试路由）、增强的可观测性（详细指标、跟踪）和安全（例如，相互 TLS）。
- **自定义 Kubernetes Operator**：对于复杂的有状态 RAG 组件或自动化特定的 RAG 操作任务，开发 Kubernetes Operator 可以封装特定领域的知识。
- **Knative**：用于 RAG 组件的无服务器式扩展，特别是对于管道中的事件驱动部分或需要扩展到零的请求-响应服务。

通过对 RAG 组件进行容器化并使用 Kubernetes 进行编排，您可以构建一个有韧性、可扩展且易于管理的体系。这种方式不仅简化了部署，还与更广泛的 MLOps 工具（用于监控、日志记录和 CI/CD）配合，这对于在生产环境中高效运行大规模 RAG 系统非常必要。

## 参考资料

- [Docker Documentation](https://docs.docker.com/) — Docker, Inc. (2024)
  Docker 的官方综合指南，涵盖核心概念、Dockerfile 最佳实践和容器管理。
- [Kubernetes Documentation](https://kubernetes.io/docs/) — The Kubernetes Authors (2024)
  Kubernetes 的官方完整参考资料，详细说明了架构、对象类型和操作实践。
- [Kubernetes in Action](https://www.manning.com/books/kubernetes-in-action) — Marko Lukša (2017)
  Publisher: Manning Publications
  一本广受认可的书籍，提供了对 Kubernetes 概念及其在生产环境中应用的实用和深入理解。
- [Practical MLOps: How to Take Machine Learning Models to Production](https://www.oreilly.com/library/view/practical-mlops/9781098103014/) — Noah Gift and Alfredo Deza (2021)
  Publisher: O'Reilly Media
  涵盖 MLOps 的端到端生命周期，包括将机器学习模型部署到生产环境的容器化和编排策略。
