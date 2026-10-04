# 动手实践：在Kubernetes上部署RAG并进行监控

来源：[原文](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag/practice-deploy-rag-kubernetes-monitoring)

[返回章节目录](README.md) · [返回课程目录](../README.md)

大规模检索增强生成 (RAG) 系统的部署架构考量包括工作流编排、微服务设计和MLOps实践。动手演示如何在Kubernetes上部署一个简化版RAG系统并配置基本监控。该演示呈现了这些组件如何在实际运行环境中协同工作。

### 前提条件

开始之前，请确保您已安装并配置以下工具：

- `kubectl` 命令行工具，已配置为与Kubernetes集群通信。您可以使用Minikube、Kind、k3s，或云服务商提供的托管Kubernetes服务（例如EKS、GKE、AKS）。
- Helm，Kubernetes包管理器。
- Docker，用于构建和管理容器镜像（不过本次实践中，我们将假定使用预构建镜像，或侧重于部署清单）。
- 对Kubernetes基本知识（Pods、Deployments、Services、ConfigMaps、Namespaces）有基本了解。
- 对前面讨论的RAG组件（检索器、生成器、向量 (vector)存储）的理解。

我们将部署一个RAG系统，它包含：

1. **检索器API**：一个微服务，接收查询，将其转换为嵌入 (embedding)，并搜索向量数据库。
2. **生成器API**：一个微服务，接收查询和检索到的上下文 (context)，并使用LLM生成答案。
3. **向量数据库**：为简化本次练习，我们将使用Qdrant，通过其Helm chart部署。
4. **监控栈**：Prometheus用于指标收集，Grafana用于可视化。

### 步骤1：设置命名空间

将您的应用程序组件部署到专用Kubernetes命名空间是一个好的做法。

```bash
kubectl create namespace rag-system
```

本次实践中，所有后续的 `kubectl` 命令都应使用 `-n rag-system` 标志运行，或者您可以设置当前上下文 (context)的默认命名空间：`kubectl config set-context --current --namespace=rag-system`。

### 步骤2：部署向量 (vector)数据库 (Qdrant)

我们将使用Helm部署Qdrant。这大大简化了设置。

1. 添加Qdrant Helm仓库：

   ```bash
   helm repo add qdrant https://qdrant.github.io/qdrant-helm
   helm repo update
   ```
2. 安装Qdrant：

   ```bash
   helm install qdrant qdrant/qdrant -n rag-system \
     --set persistence.enabled=false \
     --set replicas=1 \
     --set service.http.servicePort=6333 \
     --set service.grpc.servicePort=6334
   ```

   为简化本次实践，我们禁用持久性 (`persistence.enabled=false`) 并运行单个副本。在生产环境中，您会配置持久性并可能增加更多副本。服务端口6333 (HTTP) 和 6334 (gRPC) 是Qdrant的标准端口。

   验证Qdrant是否正在运行：

   ```bash
   kubectl get pods -n rag-system -l app.kubernetes.io/name=qdrant
   ```

   您应该会看到一个处于 `Running` 状态的Qdrant Pod。服务 `qdrant` 将在集群内部 `qdrant.rag-system.svc.cluster.local:6333` 可用。

### 步骤3：将RAG组件容器化（示例）

在实际项目中，您的检索器和生成器API会有各自的Dockerfile。例如，一个使用FastAPI的Python检索器API可能会有如下Dockerfile：

```dockerfile
# 检索器API的示例Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./retriever_app /app/retriever_app

# 假定QDRANT_HOST和QDRANT_PORT通过环境变量配置
# CMD ["uvicorn", "retriever_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

而一个生成器API，也许使用Hugging Face的Text Generation Inference (TGI)：

```dockerfile
# 生成器API的示例Dockerfile（如果未使用预构建的TGI镜像）
# 这会更复杂，涉及模型下载和设置。
# 对于本次实践，我们可能会假定使用预构建的TGI镜像或更简单的自定义LLM服务。
```

本次动手实践中，我们将侧重于Kubernetes清单，并假定您在注册表（例如Docker Hub、GCR、ECR）中已有检索器和生成器服务的容器镜像。我们假定使用 `your-repo/retriever-api:latest` 和 `your-repo/generator-api:latest`。对于生成器，如果配置得当，您也可以使用公共TGI镜像，如 `ghcr.io/huggingface/text-generation-inference:latest`。

### 步骤4：部署检索器API

创建一个名为 `retriever-deployment.yaml` 的文件：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retriever-api
  namespace: rag-system
  labels:
    app: retriever-api
spec:
  replicas: 2 # 从2个副本开始
  selector:
    matchLabels:
      app: retriever-api
  template:
    metadata:
      labels:
        app: retriever-api
      annotations:
        prometheus.io/scrape: "true" # 启用Prometheus抓取
        prometheus.io/port: "8000"   # 应用暴露指标的端口
        prometheus.io/path: "/metrics" # 指标端点的路径
    spec:
      containers:
      - name: retriever-api
        image: your-repo/retriever-api:latest # 替换为您的实际镜像
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_HOST
          value: "qdrant.rag-system.svc.cluster.local"
        - name: QDRANT_PORT
          value: "6333"
        # 为部署添加 readiness 和 liveness 探针
        readinessProbe:
          httpGet:
            path: /health # 假定一个 /health 端点
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 20
        resources: # 定义资源请求和限制
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: retriever-api-svc
  namespace: rag-system
  labels:
    app: retriever-api
spec:
  selector:
    app: retriever-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP # 内部服务
```

此清单定义了：

- 检索器API的 `Deployment`，它指定了副本数、容器镜像、Qdrant服务地址的环境变量，以及基本的健康探针。
- Prometheus的注解，用于发现并抓取此服务的指标。您的应用程序需要在 `/metrics` 路径上以Prometheus格式暴露指标。
- 一个 `ClusterIP` 类型的 `Service`，用于在Kubernetes集群内部暴露检索器API。

应用它：

```bash
kubectl apply -f retriever-deployment.yaml -n rag-system
```

### 步骤5：部署生成器API

创建一个名为 `generator-deployment.yaml` 的文件。本例假定您正在部署类似TGI的服务，它通常需要特定参数 (parameter)来加载模型。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: generator-api
  namespace: rag-system
  labels:
    app: generator-api
spec:
  replicas: 1 # LLM可能资源密集；根据您的模型和负载调整副本数
  selector:
    matchLabels:
      app: generator-api
  template:
    metadata:
      labels:
        app: generator-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "80" # TGI默认指标端口通常是80，请检查您的LLM服务器
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: generator-api
        # 使用通用TGI镜像的示例。根据您的LLM服务设置进行调整。
        image: ghcr.io/huggingface/text-generation-inference:latest # 替换或配置
        args:
          - "--model-id"
          - "mistralai/Mistral-7B-v0.1" # 示例模型，测试时选择一个小的
          - "--port"
          - "8080" # 应用程序端口，如果未通过端口指定，TGI默认使用80作为API端口
          # 根据需要添加其他必要的TGI参数，例如分片、量化等。
        ports:
        - name: http # API端口
          containerPort: 8080 # 确保此端口与TGI监听的端口匹配
        - name: metrics # Prometheus指标端口（TGI可能使用不同端口或需要特定配置）
          containerPort: 80 # TGI通常默认在80端口暴露指标。请验证。
        # 添加 readiness 和 liveness 探针。对于TGI，这可以是 /health 端点。
        readinessProbe:
          httpGet:
            path: /health
            port: 8080 # TGI用于健康检查的端口
          initialDelaySeconds: 60 # 模型加载可能需要时间
          periodSeconds: 15
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 30
        resources: # LLM是资源密集型。生产环境中需大幅调整这些设置。
          requests:
            memory: "8Gi" # 示例，高度依赖模型
            cpu: "2"      # 示例
          limits:
            memory: "16Gi" # 示例
            cpu: "4"       # 示例
        # 如果您的LLM需要GPU，您需要配置节点选择器和GPU资源请求。
        # 示例：
        # resources:
        #   limits:
        #     nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: generator-api-svc
  namespace: rag-system
  labels:
    app: generator-api
spec:
  selector:
    app: generator-api
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 8080 # TGI监听API请求的端口
  type: ClusterIP
```

此清单部署了生成器API。主要考量点：

- **镜像和模型**：将 `ghcr.io/huggingface/text-generation-inference:latest` 和 `mistralai/Mistral-7B-v0.1` 替换为您选择的LLM服务方案和模型。确保 `args` 与您的设置一致。
- **资源分配**：LLM对资源要求高。`resources` 部分需要仔细调整。如果使用GPU，请确保您的Kubernetes节点已启用GPU并且您已指定GPU资源。
- **Prometheus注解**：如果您的生成器服务在不同端口暴露指标，请更新 `prometheus.io/port`。TGI通常在80端口暴露指标。
- **健康探针**：`/health` 端点和端口应与您的LLM服务器配置匹配。模型加载可能需要时间，因此 `initialDelaySeconds` 可能需要设置得宽松些。

应用它：

```bash
kubectl apply -f generator-deployment.yaml -n rag-system
```

表示Kubernetes内部署的RAG组件的图示：

> 在Kubernetes上部署的RAG系统组件的高层架构，包括与向量 (vector)数据库的联系，以及用户请求和监控的潜在连接。

### 步骤6：部署Prometheus和Grafana进行监控

我们将使用 `kube-prometheus-stack` Helm chart，它方便地捆绑了Prometheus、Grafana、Alertmanager和各种导出器。

1. 添加Prometheus社区Helm仓库：

   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo update
   ```
2. 安装 `kube-prometheus-stack`：

   ```bash
   helm install prometheus prometheus-community/kube-prometheus-stack -n rag-system \
     --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
     --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false
   ```

   `serviceMonitorSelectorNilUsesHelmValues=false` 和 `podMonitorSelectorNilUsesHelmValues=false` 设置允许Prometheus在未明确限制的情况下发现所有命名空间中的 `ServiceMonitor` 和 `PodMonitor` 资源。为了在生产环境中进行更精细的控制，您可能希望限制此项。

   此安装可能需要几分钟。验证Pod状态：

   ```bash
   kubectl get pods -n rag-system -l "release=prometheus"
   ```

   您应该会看到Prometheus、Grafana、Alertmanager和node-exporter的Pod正在运行。

   我们添加到 `retriever-api` 和 `generator-api` 的注解（`prometheus.io/scrape: "true"` 等）是Prometheus发现抓取目标的一种方式。另一种更可靠的方式是使用 `kube-prometheus-stack`，即创建 `ServiceMonitor` 资源。

   检索器API的 `ServiceMonitor` 示例（保存为 `retriever-servicemonitor.yaml`）：

   ```yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: retriever-api-sm
     namespace: rag-system
     labels:
       release: prometheus # 与kube-prometheus-stack的Helm发布名称匹配
   spec:
     selector:
       matchLabels:
         app: retriever-api # 选择 retriever-api-svc
     namespaceSelector:
       matchNames:
       - rag-system
     endpoints:
     - port: http # Service定义中端口的名称 (retriever-api-svc)
       path: /metrics # 指标暴露的路径
       interval: 15s
   ```

   应用它：`kubectl apply -f retriever-servicemonitor.yaml -n rag-system`。您会为 `generator-api-svc` 创建类似的 `ServiceMonitor`。这会告知由 `kube-prometheus-stack` 部署的Prometheus从匹配这些标签的服务中抓取指标。

### 步骤7：访问Grafana并构建基本仪表盘

1. 访问Grafana。`kube-prometheus-stack` 通常会创建一个Grafana服务。查找其类型和访问方式：

   ```bash
   kubectl get svc -n rag-system prometheus-grafana
   ```

   如果它是 `ClusterIP` 类型，您可以进行端口转发：

   ```bash
   kubectl port-forward svc/prometheus-grafana 3000:80 -n rag-system
   ```

   然后通过 `http://localhost:3000` 访问Grafana。
2. 此chart部署的Grafana默认登录凭据通常是 `admin` / `prom-operator`。如果这些无效，请查阅chart的文档以获取当前默认值。
3. 创建一个新仪表盘：

   - 点击左侧边栏的 `+` 图标，然后点击 `Dashboard`。
   - 点击 `Add new panel`。
   - 在 `Query` 选项卡中，选择 `Prometheus` 作为数据源（它应该已预配置）。
   - 输入PromQL查询。例如，查看对检索器API的HTTP请求速率（假定您的指标命名适当，例如 `http_requests_total`）：

     ```promql
     sum(rate(http_requests_total{job="rag-system/retriever-api-svc", handler!="/metrics"}[5m])) by (handler, method, status_code)
     ```

     根据Prometheus发现服务的方式调整 `job` 标签。如果使用 `ServiceMonitor`，`job` 标签可能是 `retriever-api-sm` 或类似名称。使用Grafana中的“指标浏览器”查找可用指标。
   - 一个更简单的检索器Pod CPU使用率查询：

     ```promql
     sum(rate(container_cpu_usage_seconds_total{namespace="rag-system", pod=~"retriever-api-.*", container="retriever-api"}[5m])) by (pod)
     ```
   - 前往右侧的 `Visualization` 设置并选择图表类型（例如，时序图）。
   - 为您的面板命名（例如，“检索器API请求速率”或“检索器CPU使用率”）。
   - 保存面板和仪表盘。

以下是一个Plotly JSON配置示例，它可以在仪表盘中表示一个简单的时间序列图表，显示API随时间变化的延迟。



[交互图表：RAG API 延迟 (P95)](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-5-orchestration-operationalization-large-scale-rag/practice-deploy-rag-kubernetes-monitoring#plot-1btx70r)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "检索器 API P95 延迟",
      "x": [
        "2023-10-26 10:00",
        "2023-10-26 10:05",
        "2023-10-26 10:10",
        "2023-10-26 10:15",
        "2023-10-26 10:20"
      ],
      "y": [
        120,
        125,
        118,
        130,
        122
      ],
      "line": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "生成器 API P95 延迟",
      "x": [
        "2023-10-26 10:00",
        "2023-10-26 10:05",
        "2023-10-26 10:10",
        "2023-10-26 10:15",
        "2023-10-26 10:20"
      ],
      "y": [
        850,
        860,
        845,
        870,
        855
      ],
      "line": {
        "color": "#51cf66"
      }
    }
  ],
  "layout": {
    "title": "RAG API 延迟 (P95)",
    "xaxis": {
      "title": "时间"
    },
    "yaxis": {
      "title": "延迟 (ms)"
    },
    "paper_bgcolor": "#f8f9fa",
    "plot_bgcolor": "#e9ecef"
  }
}
```

</details>



> 仪表盘面板的示例数据，显示检索器和生成器API在短时间内的P95延迟。

### 步骤8：测试已部署的RAG系统

要测试端到端系统，通常会有一个入口点，例如API网关或一个简单的前端应用程序，它负责编排对 `retriever-api-svc` 和 `generator-api-svc` 的调用。本次实践中，我们未部署此类组件，以保持重点。

但是，您可以使用端口转发来测试单个服务：

1. 转发 `retriever-api-svc` 端口：

   ```bash
   kubectl port-forward svc/retriever-api-svc 8081:80 -n rag-system
   ```

   现在您可以向 `http://localhost:8081` 发送请求。例如，如果您的检索器API有一个 `/search` 端点：

   ```bash
   # 假定检索器需要一个包含“query”字段的JSON负载
   curl -X POST http://localhost:8081/search \
        -H "Content-Type: application/json" \
        -d '{"query": "What are the principles of distributed RAG?"}'
   ```
2. 同样地，转发 `generator-api-svc` 端口并测试它：

   ```bash
   kubectl port-forward svc/generator-api-svc 8082:80 -n rag-system
   ```

   然后，向生成器的端点发送请求（例如，TGI的 `/generate` 或 `/v1/generate`）：

   ```bash
   # 类似TGI端点的示例（根据需要调整负载和端点）
   curl -X POST http://localhost:8082/generate \
        -H "Content-Type: application/json" \
        -d '{"inputs": "Query: What are distributed RAG principles?\nContext: Distributed RAG involves...", "parameters": {"max_new_tokens": 100}}'
   ```

发送一些测试请求后，回到您的Grafana仪表盘。您应该会看到指标开始填充您创建的面板，反映活动情况。例如，请求计数应该会增加，延迟图表应该会显示数据点。

### 其他考量

本次实践提供了一个基础部署。对于生产级别的系统，您会在此基础上完善以下方面：

- **实现API网关**：提供单一入口点、身份验证、速率限制等。
- **数据摄取**：这里的Qdrant实例是空的。您需要一个摄取管道（可能是Kubernetes Jobs或CronJobs）来填充和更新向量 (vector)数据库。
- **高级监控和日志记录**：集成分布式追踪（例如Jaeger、OpenTelemetry）和集中式日志记录（例如ELK stack、Loki）。
- **自动扩缩**：为您的API部署配置Horizontal Pod Autoscalers (HPAs)，基于CPU、内存或自定义指标。
- **CI/CD 流水线**：自动化RAG组件的构建、测试和部署。
- **安全**：实施网络策略，安全地管理密钥（例如HashiCorp Vault、带加密的Kubernetes Secrets），并加固容器镜像。
- **成本优化**：选择合适的机器类型，在适用情况下利用竞价实例，并密切监控资源使用情况。

本次动手练习演示了在Kubernetes上部署和监控RAG系统的主要机制。通过在此基础上发展，您可以很好地实现复杂的、大规模RAG解决方案的运行。

## 参考资料

- [Kubernetes Documentation](https://kubernetes.io/docs/) — Kubernetes Authors (2024)
  理解和实现健壮的Kubernetes部署（包括部署、服务和命名空间等概念）的基础资源。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍检索增强生成（RAG）框架的开创性论文，提供了对其架构和优势的基础认识。
- [Effective MLOps: Building, Deploying, and Maintaining Production Machine Learning Systems](https://learning.oreilly.com/library/view/effective-mlops/9781098104868/) — Luigi Patruno, Catriona Campbell (2023)
  Publisher: O'Reilly Media; DOI: [10.5220/000002](https://doi.org/10.5220/000002)
  提供了关于MLOps实践的全面指导，涵盖CI/CD、监控以及生产环境中机器学习系统的扩容。
- [Qdrant Documentation](https://qdrant.tech/documentation/) — Qdrant Team (2024)
  提供了部署、配置和使用Qdrant作为向量数据库的官方指南，包括与Kubernetes的集成。

---

[上一节](07-%E4%BA%91%E7%AB%AFRAG%E7%9A%84%E6%88%90%E6%9C%AC%E4%BC%98%E5%8C%96%E7%AD%96%E7%95%A5.md) · [下一节](../06-%E9%AB%98%E7%BA%A7%20RAG%20%E6%9E%B6%E6%9E%84%E4%B8%8E%E6%8A%80%E6%9C%AF/01-%E5%A4%9A%E8%B7%B3%E4%B8%8E%E8%BF%AD%E4%BB%A3%E5%BC%8FRAG%E7%9A%84%E8%A7%84%E6%A8%A1%E5%8C%96%E5%BA%94%E7%94%A8.md)
