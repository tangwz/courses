# 实践：配置GPU感知型自动扩缩组

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration/practice-gpu-autoscaling-group)

[返回章节目录](README.md) · [返回课程目录](../README.md)

您将配置一个支持GPU的节点池，该节点池可从零扩缩，确保仅在实际需要时才供应这些昂贵的资源。这种设置是构建经济高效的共享ML平台的基础。

我们将结合Kubernetes的污点（taints）、容忍度（tolerations）和资源请求，以创建一个系统，使集群自动扩缩器（Cluster Autoscaler）能够做出明智的、了解GPU的扩缩决策。您将部署一个明确请求GPU的Pod，并观察整个自动化过程，从Pod最初的`Pending`状态，到新GPU节点的供应，以及最终工作负载的调度。

### 前提条件

开始之前，请确保您的环境已具备以下条件：

- 一个在支持GPU实例的云服务商（如GKE、EKS或AKS）上正常运行的Kubernetes集群。
- `kubectl`已配置为与您的集群通信。
- 相应的云命令行工具（例如`gcloud`、`aws`、`az`）已通过认证。
- Kubernetes集群自动扩缩器（Cluster Autoscaler）必须已部署并在您的集群中运行。我们假设已存在一个基线安装，如前一节所述。本次练习侧重于调整其行为以适应GPU工作负载。

### GPU自动扩缩机制

了解GPU的自动扩缩逻辑不依赖于集群自动扩缩器中的特殊“GPU模式”。相反，它是标准Kubernetes调度原语共同运作的结果。该过程如下进行：

1. **节点池污点化：** 我们为GPU创建一个专用节点池，并对其应用一个污点，例如`nvidia.com/gpu=present:NoSchedule`。此污点阻止任何Pod调度到这些节点上，除非它明确具有匹配的容忍度。这将我们昂贵的GPU资源圈定给真正需要它们的工作负载使用。
2. **资源请求：** Pod清单被定义为通过`resources.limits`请求GPU，并包含对前一步骤中应用的污点的`toleration`（容忍度）。
3. **触发扩容：** 当提交此Pod时，Kubernetes调度器会尝试放置它。它发现没有可用的、Pod可以调度的GPU节点。Pod进入`Pending`（待处理）状态。
4. **自动扩缩器行动：** 集群自动扩缩器检测到`Pending`状态的Pod。它分析Pod的需求（GPU资源、容忍度），并确定从污点化的GPU节点池添加一个节点将允许该Pod被调度。
5. **资源供应：** 集群自动扩缩器调用云服务商的API，将GPU节点池的所需节点数量增加一个。
6. **调度：** 一旦新节点加入集群并报告其为`Ready`（就绪）状态，Kubernetes调度器将待处理的Pod调度到该节点上。

以下图表说明了这一系列事件的完整流程。

> 从Pod提交到成功调度到新供应的GPU节点上的自动扩缩流程。

### 步骤1：创建一个污点化且支持自动扩缩的节点池

首先，我们定义一个符合自动扩缩条件的节点池，且专门用于GPU工作负载。两个最重要的参数 (parameter)是将最小规模设置为`0`并应用`NoSchedule`污点。将`min-nodes`设置为零是我们节约成本策略的核心。

以下是在Google Kubernetes Engine (GKE)中创建此类节点池的示例命令。对于EKS或AKS，原理是相同的，尽管具体参数会有所不同。

```sh
# Example for GKE
gcloud container node-pools create gpu-pool \
  --project "<your-gcp-project>" \
  --cluster "<your-cluster-name>" \
  --zone "<your-cluster-zone>" \
  --machine-type "n1-standard-4" \
  --accelerator "type=nvidia-tesla-t4,count=1" \
  --enable-autoscaling \
  --min-nodes "0" \
  --max-nodes "5" \
  --node-taints "nvidia.com/gpu=present:NoSchedule" \
  --node-labels "app-type=gpu-workloads"
```

运行此命令后，您会得到一个名为`gpu-pool`的节点池。它目前没有节点，但已准备好扩容至最多五个配备T4的节点。在此池中创建的任何节点都将自动被污点化，从而排斥没有正确容忍度的Pod。

### 步骤2：安装NVIDIA GPU设备插件

Kubernetes本身不原生理解GPU是什么。它需要一个设备插件来发现并公开节点上的GPU硬件。NVIDIA GPU Operator是管理此过程的推荐方式，因为它会自动处理驱动安装、设备插件注册和监控组件。

如果您尚未安装，请添加NVIDIA Helm仓库并安装该操作符。

```sh
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

helm install gpu-operator nvidia/gpu-operator \
  --wait \
  --namespace gpu-operator \
  --create-namespace
```

一旦操作符的Pod开始运行，任何带有NVIDIA GPU的节点都将自动被标记 (token)为`nvidia.com/gpu=true`，并且其GPU容量将作为名为`nvidia.com/gpu`的可分配资源对Kubernetes调度器可见。

### 步骤3：定义一个Pod以触发扩容

现在，创建将触发扩容的工作负载。以下YAML清单定义了一个简单的Pod，它除了占用一个GPU外不执行任何操作。请注意两个重要部分：`resources.limits`用于请求GPU，以及`tolerations`用于允许它调度到我们污点化的节点上。

将此内容保存为`gpu-test-pod.yaml`。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-pod
spec:
  restartPolicy: Never
  containers:
  - name: cuda-container
    image: nvidia/cuda:11.4.2-base-ubuntu20.04
    command: ["/bin/bash", "-c", "sleep 3600"] # 暂停1小时
    resources:
      limits:
        nvidia.com/gpu: 1
  tolerations:
  - important: "nvidia.com/gpu"
    operator: "Equal"
    value: "present"
    effect: "NoSchedule"
```

### 步骤4：部署Pod并观察系统

所有部分准备就绪后，应用清单并观察自动化过程。

1. **部署Pod：**

   ```sh
   kubectl apply -f gpu-test-pod.yaml
   ```
2. **观察Pod状态：**
   立即检查其状态。您将看到它处于`Pending`（待处理）状态。

   ```sh
   kubectl get pods -w

   # 初始输出将类似如下
   # NAME           READY   STATUS    RESTARTS   AGE
   # gpu-test-pod   0/1     Pending   0          2s
   ```
3. **检查Pod事件：**
   为了理解它为何待处理，请描述该Pod。在`Events`（事件）部分，您将看到调度器发出的消息，指示它找不到合适的节点。

   ```sh
   kubectl describe pod gpu-test-pod
   ```

   您应该看到类似这样的事件：`Warning FailedScheduling ... 0/X nodes are available: X node(s) had taints that the pod didn't tolerate.` 这是预期的，也是集群自动扩缩器的触发条件。
4. **检查集群自动扩缩器日志：**
   在另一个终端中，查看集群自动扩缩器部署的日志。您将看到它识别出待处理的Pod并触发`gpu-pool`的扩容。

   ```sh
   # 查找自动扩缩器Pod
   kubectl get pods -n kube-system | grep cluster-autoscaler

   # 查看其日志
   kubectl logs -f <cluster-autoscaler-pod-name> -n kube-system
   ```

   查找包含`Scale-up`、`pod gpu-test-pod triggered scale-up`和`... expanding node group .../gpu-pool from 0 to 1`的行。
5. **观察新节点出现：**
   在自动扩缩器日志运行的同时，观察您集群中的节点。`gpu-pool`中的一个新节点将会出现，最初处于`NotReady`（未就绪）状态，几分钟后会转换为`Ready`（就绪）状态。

   ```sh
   kubectl get nodes -w
   ```
6. **确认Pod调度：**
   一旦新节点就绪，Kubernetes调度器将自动把`gpu-test-pod`放置到该节点上。Pod的状态将从`Pending`（待处理）变为`ContainerCreating`（容器创建中），最终变为`Running`（运行中）。

### 步骤5：缩容与清理

该系统还处理缩容以节约成本。完成GPU工作负载后，请删除Pod。

```sh
kubectl delete -f gpu-test-pod.yaml
```

Pod删除后，新的GPU节点现在处于空闲状态。集群自动扩缩器将识别到这种资源利用不足的情况。在其配置的超时时间（通常为10分钟）过后，它将终止该节点并将`gpu-pool`缩容回零。您已成功创建了一个完全弹性、按需的GPU资源池。为避免本次练习产生任何额外费用，您还可以删除节点池本身。

```sh
# Example for GKE
gcloud container node-pools delete gpu-pool \
  --cluster "<your-cluster-name>" \
  --zone "<your-cluster-zone>"
```

## 参考资料

- [Cluster Autoscaler](https://kubernetes.io/docs/concepts/cluster-administration/node-autoscaling/) — Kubernetes Authors (2024)
  Publisher: The Linux Foundation
  Kubernetes 官方文档，解释了集群自动扩缩器的目的、架构和配置。
- [Taints and Tolerations](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/) — Kubernetes Authors (Current)
  使用污点和容忍度进行节点隔离和选择性 Pod 调度的官方指南。
- [Resource Management for Pods and Containers](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/) — Kubernetes Authors (Current)
  Publisher: The Kubernetes Project
  描述了如何为 Pod 指定资源请求和限制，这对于 GPU 分配至关重要。
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html) — NVIDIA (2024)
  Publisher: NVIDIA
  使用 GPU Operator 在 Kubernetes 中部署和管理 NVIDIA GPU 的官方文档。
- [Running GPUs on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus) — Google Cloud (Current)
  Publisher: Google Cloud
  在 Google Kubernetes Engine 上配置和使用 GPU 节点及工作负载的指南。

---

[上一节](05-%E9%80%9A%E8%BF%87%E5%91%BD%E5%90%8D%E7%A9%BA%E9%97%B4%E3%80%81%E9%85%8D%E9%A2%9D%E5%92%8C%E4%BC%98%E5%85%88%E7%BA%A7%E7%B1%BB%E5%AE%9E%E7%8E%B0%E5%A4%9A%E7%A7%9F%E6%88%B7.md) · [下一节](../04-%E7%AC%AC4%E7%AB%A0%EF%BC%9A%E9%AB%98%E6%80%A7%E8%83%BD%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86%E4%B8%8E%E6%9C%8D%E5%8A%A1/01-%E9%92%88%E5%AF%B9%E5%BB%B6%E8%BF%9F%E5%92%8C%E5%90%9E%E5%90%90%E9%87%8F%E7%9A%84%E6%8E%A8%E7%90%86%E6%9C%8D%E5%8A%A1%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1.md)
