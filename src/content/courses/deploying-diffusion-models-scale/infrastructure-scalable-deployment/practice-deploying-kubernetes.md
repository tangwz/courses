---
course: "deploying-diffusion-models-scale"
chapter: "infrastructure-scalable-deployment"
lesson: "practice-deploying-kubernetes"
sourceId: 5429
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment/practice-deploying-kubernetes"
title: "动手实践：在 Kubernetes 上部署"
description: "在支持 GPU 的 Kubernetes 集群上部署容器化的扩散模型服务。"
order: 8
plots: []
sourceHash: "bbf03b4e38721bc6ff48ef34ee6e64493e1f09ea5abc6b09042a7e780ef24db1"
sourceCorrections: []
---

将容器化的扩散模型应用部署到配置了 GPU 支持的 Kubernetes 集群上，展示了如何实现可扩缩基础设施。本次实践练习将详细介绍已构建并推送到容器注册表的应用的部署过程。它模拟了一个典型的部署场景，整合了容器化、编排和硬件加速。

### 前提条件

在继续之前，请确保您已准备好以下各项：

1. **Kubernetes 集群访问权限：** 您需要访问一个 Kubernetes 集群（例如 GKE、EKS、AKS，或适当配置的本地环境，如 Minikube/Kind）。
2. **GPU 节点：** 集群必须有配备与您的模型和容器环境兼容的 GPU 的节点。
3. **GPU 设备插件：** NVIDIA 设备插件（或适用于其他硬件的等效插件）必须已安装并在集群上运行。这使 Kubernetes 能够发现和调度 GPU 资源。您通常可以通过检查 `kube-system` 命名空间中与设备插件相关的 daemonset 来验证这一点。
4. **`kubectl`：** Kubernetes 命令行工具，配置为与您的集群通信。
5. **容器镜像：** 包含您的扩散模型应用的 Docker 镜像必须可从您的 Kubernetes 集群访问（例如，已推送到 Docker Hub、Google Container Registry、AWS ECR 等）。请记下完整的镜像 URL（例如，`your-registry/your-diffusion-app:v1.0`）。

### 第 1 步：定义 Kubernetes 部署

Kubernetes Deployment 管理无状态应用，确保指定数量的副本（Pod）在运行。我们需要定义一个 Deployment，告诉 Kubernetes 如何运行我们的扩散模型容器，特别是请求 GPU 资源。

创建一个名为 `diffusion-deployment.yaml` 的文件，内容如下。请记住将 `your-registry/your-diffusion-app:v1.0` 替换为您的容器镜像的实际 URL。

```yaml
# diffusion-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diffusion-deployment
  labels:
    app: diffusion-service
spec:
  replicas: 1 # 从一个副本开始
  selector:
    matchLabels:
      app: diffusion-service
  template:
    metadata:
      labels:
        app: diffusion-service
    spec:
      containers:
      - name: diffusion-container
        image: your-registry/your-diffusion-app:v1.0 # <-- 替换为您的镜像 URL
        ports:
        - containerPort: 8080 # 您的应用在容器内部监听的端口
        resources:
          limits:
            nvidia.com/gpu: 1 # 请求精确 1 个 GPU
          requests:
            nvidia.com/gpu: 1 # 请求精确 1 个 GPU
        # 可选：如果您的应用需要，可添加环境变量
        # env:
        # - name: MODEL_PATH
        #   value: "/models/stable-diffusion-v1-5"
      # 可选：如果您的 GPU 节点具有特定的污点，可添加容忍度
      # tolerations:
      # - important: "nvidia.com/gpu"
      #   operator: "Exists"
      #   effect: "NoSchedule"
```

**此清单中的要点：**

- **`kind: Deployment`**：指定对象类型。
- **`metadata.name`**：为 Deployment 资源命名。
- **`spec.replicas`**：定义所需运行的 Pod 数量。我们从 1 个开始。自动扩缩（之前已讨论）稍后可以动态调整此值。
- **`spec.selector.matchLabels`**：使用标签将 Deployment 与其管理的 Pod 连接起来。
- **`spec.template.metadata.labels`**：为此 Deployment 创建的 Pod 分配标签。
- **`spec.template.spec.containers`**：定义将在 Pod 内运行的容器。
  - **`image`**：要拉取的容器镜像。**请确保此项正确。**
  - **`ports.containerPort`**：您的应用在容器*内部*监听的端口。这尚未将其暴露到外部。
  - **`resources.limits`** 和 **`resources.requests`**：这是我们请求 GPU 资源的地方。`nvidia.com/gpu: 1` 告诉 Kubernetes 仅在至少有一个可用 NVIDIA GPU 的节点上调度此 Pod，并为该容器分配一个 GPU。精确的资源名称（`nvidia.com/gpu`）取决于设备插件的安装。
- **`tolerations`**（可选）：如果您的 GPU 节点被“污点”以阻止非 GPU 工作负载调度到其上，您可能需要在 Pod 规范中添加容忍度，以便它们*可以*调度到那里。

### 第 2 步：定义 Kubernetes 服务

Deployment 运行您的 Pod，但我们需要一种可靠的方式来访问它们，特别是当 Pod 被重新创建或扩缩时。Kubernetes Service 为访问由 Deployment 管理的 Pod 提供一个稳定的网络端点（IP 地址和 DNS 名称）。

创建一个名为 `diffusion-service.yaml` 的文件：

```yaml
# diffusion-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: diffusion-loadbalancer
spec:
  selector:
    app: diffusion-service # 必须与 Deployment 中定义的 Pod 标签匹配
  ports:
    - protocol: TCP
      port: 80       # 服务将在外部（或内部）可访问的端口
      targetPort: 8080 # 容器监听的端口（来自 deployment.yaml）
  type: LoadBalancer # 对云提供商使用 LoadBalancer（或对本地/手动设置使用 NodePort）
```

**此清单中的要点：**

- **`kind: Service`**：指定对象类型。
- **`metadata.name`**：为 Service 资源命名。
- **`spec.selector`**：根据 Pod 的标签选择此服务将路由流量到的 Pod。它必须与我们 Deployment 的 Pod 模板中的 `app: diffusion-service` 标签匹配。
- **`spec.ports`**：定义端口映射。到达服务 `port: 80` 的流量将被转发到容器的 `targetPort: 8080`。
- **`spec.type: LoadBalancer`**：此类型常用于云提供商（AWS、GCP、Azure）。它会自动预配一个云负载均衡器，将外部流量引导到服务。如果您在本地运行（例如 Minikube）或需要手动外部配置，您可以使用 `type: NodePort` 代替，它会在每个节点上的特定端口暴露服务。

### 第 3 步：应用清单

现在，使用 `kubectl` 将这些配置文件应用到您的集群：

```bash
# 应用 Deployment 配置
kubectl apply -f diffusion-deployment.yaml

# 应用 Service 配置
kubectl apply -f diffusion-service.yaml
```

Kubernetes 将开始创建 YAML 文件中定义的资源。

### 第 4 步：验证部署

检查您的 Deployment 和 Pod 的状态：

```bash
# 检查 Deployment 是否正在进行中
kubectl get deployments diffusion-deployment

# 预期输出（可能需要一些时间）：
# NAME                   READY   UP-TO-DATE   AVAILABLE   AGE
# diffusion-deployment   1/1     1            1           30s

# 检查 Pod 状态（它需要拉取镜像并启动）
kubectl get pods -l app=diffusion-service

# 预期输出（STATUS 应变为 Running）：
# NAME                                    READY   STATUS    RESTARTS   AGE
# diffusion-deployment-xxxxxxxxxx-yyyyy   1/1     Running   0          1m

# 如果 Pod 停滞在 Pending 状态或出现错误，请检查：
# kubectl describe pod <pod-name-from-above>
# 查找与调度（GPU 可用性）或镜像拉取相关的事件。
```

检查您的 Service 状态并查找其外部 IP（如果使用 `LoadBalancer`）：

```bash
# 检查服务状态
kubectl get services diffusion-loadbalancer

# 预期输出（EXTERNAL-IP 最初可能为 <pending>）：
# NAME                     TYPE           CLUSTER-IP     EXTERNAL-IP      PORT(S)        AGE
# diffusion-loadbalancer   LoadBalancer   10.x.x.x       <EXTERNAL_IP>    80:3xxxx/TCP   2m
```

云提供商预配负载均衡器并分配外部 IP 地址可能需要几分钟。请持续检查直到 `EXTERNAL-IP` 出现。如果您使用 `type: NodePort`，`PORT(S)` 列将显示 `80:<NodePort>/TCP` 这样的映射。

下图展示了所创建的资源：

> 该图显示了 Kubernetes Deployment 管理一个 Pod，该 Pod 运行使用 GPU 的容器化扩散模型应用。LoadBalancer 服务将应用对外暴露，将用户请求路由到 Pod。

### 第 5 步：测试推理 (inference)端点

一旦您的 `diffusion-loadbalancer` 服务的 `EXTERNAL-IP` 可用（且 Pod 处于 `Running` 状态），您就可以发送推理请求。假设您的容器内应用在 8080 端口暴露了一个接受通过 POST 请求发送 JSON 提示的 `/generate` 端点，您可以使用 `curl` 进行测试（将 `<EXTERNAL_IP>` 替换为从 `kubectl get services` 获取的实际 IP 地址）：

```bash
# 将 <EXTERNAL_IP> 替换为服务的实际外部 IP
export SERVICE_IP=<EXTERNAL_IP>

# 发送一个示例请求（根据需要调整端点和 JSON 数据）
curl -X POST http://${SERVICE_IP}:80/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "a serene scene with a flowing river", "steps": 30}' \
     --output generated_image.png # 如果直接返回图片，则保存输出

# 或者如果返回包含图片 URL/数据的 JSON：
# curl -X POST http://${SERVICE_IP}:80/generate -H "Content-Type: application/json" -d '{"prompt": "..."}'
```

监视响应。成功的请求应返回生成的图片或相关数据，具体取决于您的 API 设计。如果遇到错误，请检查 Pod 日志：

```bash
# 首先获取 Pod 名称
kubectl get pods -l app=diffusion-service

# 跟踪 Pod 日志
kubectl logs -f <pod-name-from-above>
```

### 第 6 步：验证 GPU 使用情况（可选）

为确认应用正在 Pod 内部使用 GPU，您可以在容器内执行 `nvidia-smi`：

```bash
# 获取 Pod 名称
POD_NAME=$(kubectl get pods -l app=diffusion-service -o jsonpath='{.items[0].metadata.name}')

# 在 Pod 的容器内执行 nvidia-smi
kubectl exec $POD_NAME -- nvidia-smi
```

如果命令成功运行并显示 GPU 使用详情（特别是在处理请求时），您的部署已正确配置为使用硬件加速。

### 第 7 步：清理资源

为避免产生费用，在您完成试验后删除 Kubernetes 资源：

```bash
# 删除服务（这将取消预配云负载均衡器）
kubectl delete service diffusion-loadbalancer

# 删除部署（这将终止 Pod）
kubectl delete deployment diffusion-deployment

# 验证删除
kubectl get deployments,services,pods -l app=diffusion-service
# 应返回 "No resources found"
```

本次实践练习演示了在 Kubernetes 上部署 GPU 加速扩散模型服务的核心步骤。您创建了 Deployment 和 Service 清单，请求了特定的 GPU 资源，应用了配置，验证了设置，并测试了端点。这为构建更复杂、可扩缩且更具弹性的部署架构提供了条件，我们将在后续章节中查看。请记住，生产系统通常需要额外的配置，用于监控、日志记录、自动扩缩和强大的错误处理。

## 参考资料

- [Kubernetes Concepts Documentation](https://kubernetes.io/docs/concepts/) — Kubernetes Authors (2020)
  官方文档，概述了Pods、Deployments、Services和资源管理等Kubernetes基本对象，对于理解实际部署步骤至关重要。
- [NVIDIA device plugin for Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/kubernetes/device-plugin.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  官方NVIDIA文档，描述了设备插件的安装和功能，使Kubernetes能够管理和调度容器化应用程序的GPU资源。
- [Deploying and Scaling Machine Learning Models with Kubernetes](https://www.packtpub.com/product/deploying-and-scaling-machine-learning-models-with-kubernetes/9781800561582) — Paul Brebner (2021)
  Publisher: Packt Publishing
  一本关于使用Kubernetes部署和扩展机器学习模型的实践指南，涵盖了GPU调度和暴露服务等相关主题。
