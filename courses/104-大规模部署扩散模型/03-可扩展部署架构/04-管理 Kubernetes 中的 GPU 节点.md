# 管理 Kubernetes 中的 GPU 节点

来源：[原文](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment/managing-gpu-nodes-kubernetes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 Kubernetes 上部署扩散模型，在处理标准 CPU 工作负载之外，会带来一些特定要求。去噪过程的计算密集型特性要求使用专用硬件，主要是图形处理单元（GPU）。仅仅在集群节点中拥有 GPU 硬件是不够的；Kubernetes 需要一种方式来识别、管理并将这些资源分配给需要它们的 Pod。

### 启用 GPU 感知：设备插件框架

原生的 Kubernetes 不自带感知供应商专用硬件，如 GPU。为了解决此问题，Kubernetes 提供了一个设备插件框架。此框架允许硬件供应商（如 NVIDIA 或 AMD）或第三方开发插件，这些插件运行在每个节点上，检测特定硬件资源（例如 GPU），向 Kubelet（主要节点代理）报告其可用性，并管理其对容器的分配。

对于 NVIDIA GPU，最常用的方案是适用于 Kubernetes 的 NVIDIA 设备插件。此插件自动识别节点上可用的 NVIDIA GPU 数量和类型，并将其作为 Kubernetes 集群内的可调度资源暴露。

### 安装 NVIDIA 设备插件

安装通常涉及部署一个 DaemonSet。DaemonSet 确保设备插件 Pod 的一个副本在集群中的每个（或选定子集）节点上运行。这使得每个能够运行 GPU 工作负载的节点都能报告其 GPU 资源。

安装方式会因您的集群环境（如 EKS、GKE、AKS 等托管式云 Kubernetes，或自管理集群）而略有差异，但通常是应用 NVIDIA 提供的 YAML 清单。

例如，使用 `kubectl`：

```yaml
# 示例命令，请参考 NVIDIA 官方文档获取最新的清单
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.1/nvidia-device-plugin.yml
```

在安装插件之前，请确保您的 GPU 节点主机操作系统上已安装相应的 NVIDIA 驱动程序。设备插件依赖于这些驱动程序与硬件通信。主机操作系统、容器镜像中使用的 CUDA 工具包版本以及 GPU 硬件本身的驱动程序兼容性是一个重要方面。

### 在 Pod 规范中请求 GPU 资源

一旦设备插件运行并已向 Kubernetes API 服务器注册了 GPU 资源，您就可以在 Pod 规范中请求 GPU，这很像请求 CPU 或内存。资源名称通常是供应商专有的，对于 NVIDIA GPU，它是 `nvidia.com/gpu`。

这里是一个 Deployment 清单的示例片段，它为其容器请求一个 NVIDIA GPU：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diffusion-worker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: diffusion-worker
  template:
    metadata:
      labels:
        app: diffusion-worker
    spec:
      containers:
      - name: diffusion-container
        image: your-registry/your-diffusion-app:latest
        resources:
          limits:
            nvidia.com/gpu: 1 # 请求 1 个 NVIDIA GPU
          requests:
            nvidia.com/gpu: 1 # 可选，但如果设置了 limits，则建议也设置 requests
        # ... 其他容器配置 (端口、环境变量等)
```

当此 Pod 被调度时，Kubernetes 调度器将只考虑拥有至少一个 `nvidia.com/gpu` 资源且该资源可用且未分配的节点。所选节点上的 Kubelet 通过与设备插件协作，会将一个特定的 GPU 设备分配给容器，并使其在容器环境中可用（通常通过设备挂载）。

### 隔离 GPU 工作负载：标签、污点和容忍度

在许多生产环境中，将特定节点专门用于 GPU 工作负载是一种期望的做法，以提升资源利用率并降低成本，并阻止非 GPU 工作负载占用昂贵的 GPU 实例上的资源。Kubernetes 为此提供了多种方式：

1. **节点标签：** 您可以为您的 GPU 节点应用自定义标签。例如：`kubectl label node <您的 GPU 节点名称> hardware-type=nvidia-gpu`。
2. **节点选择器/亲和性：** 在您的 Pod 规范中，您可以使用 `nodeSelector` 或更复杂的 `nodeAffinity` 规则，以确保您的扩散模型 Pod 只被调度到带有 `hardware-type=nvidia-gpu` 标签的节点上。

   ```yaml
   # 使用 nodeSelector 的示例
   spec:
     nodeSelector:
       hardware-type: nvidia-gpu
     containers:
       # ... 容器规范如上
   ```
3. **污点和容忍度：** 为避免非 GPU 工作负载被调度到昂贵的 GPU 节点上，您可以为 GPU 节点“标记 (token)污点”。污点会排斥 Pod，除非 Pod 具有匹配的“容忍度”。

   - 标记 GPU 节点污点：`kubectl taint nodes <您的 GPU 节点名称> nvidia.com/gpu=present:NoSchedule`
   - 为您的 GPU Pod 规范添加容忍度：

     ```yaml
     spec:
       tolerations:
       - "nvidia.com/gpu"
         operator: "Exists"
         effect: "NoSchedule"
       containers:
         # ... 容器规范如上，包括 GPU 请求
     ```

这种组合确保只有明确请求 GPU *并* 容忍 `nvidia.com/gpu` 污点的 Pod 才会被调度到 GPU 节点上。

> 此图描绘了 Kubernetes 调度器如何根据 GPU 需求、节点标签、污点和容忍度分配 Pod，并由 NVIDIA 设备插件提供支持。

### GPU 节点管理的注意事项

- **驱动一致性：** 请确保主机节点上的 NVIDIA 驱动程序与容器内使用的 CUDA 版本以及设备插件版本兼容。不匹配是导致问题的原因之一。
- **资源监控：** 使用与设备插件兼容的工具（例如，使用 `dcgm-exporter` 的 Prometheus）监控节点上的 GPU 利用率、内存使用情况和温度。这对于识别性能瓶颈和支持自动扩缩容决策很有帮助。
- **多实例 GPU (MIG)：** 对于较新的 NVIDIA 架构（如 A100），MIG 支持将单个物理 GPU 划分为多个更小、完全隔离的 GPU 实例。NVIDIA 设备插件支持 MIG，从而实现更细粒度的资源分配，适用于不需要整个 GPU 的工作负载，有望提高利用率并降低成本。配置时需要特定的节点设置和 Pod 规范中相应的资源请求（例如，`nvidia.com/mig-1g.5gb: 1`）。
- **更新与维护：** 更新 GPU 驱动程序或设备插件通常需要仔细协调，可能涉及排空节点，以减少对正在运行的推理 (inference)服务的影响。

在 Kubernetes 中有效管理 GPU 节点，对于可靠且高效地部署扩散模型来说非常重要。通过使用设备插件、资源请求、标签、污点和容忍度，您可以对调度进行精确控制，以确保您的计算密集型生成工作负载能够获取所需硬件资源，同时优化集群利用率。这为构建可扩展且高效益的推理基础设施创造了条件。

## 参考资料

- [Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) — Kubernetes Authors (2024)
  Kubernetes官方文档，解释了设备插件框架，该框架使Kubernetes能够发现和管理GPU等特定厂商硬件。
- [NVIDIA device plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin) — NVIDIA Corporation (2024)
  NVIDIA Kubernetes设备插件的官方GitHub仓库，提供安装说明、使用详情和源代码。
- [Multi-Instance GPU (MIG) User Guide for Kubernetes](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA官方用户指南，说明了如何在Kubernetes中配置和使用多实例GPU（MIG）与NVIDIA设备插件，从而实现更精细的GPU资源分配。

---

[上一节](03-%E4%BD%BF%E7%94%A8%20Kubernetes%20%E8%BF%9B%E8%A1%8C%E7%BC%96%E6%8E%92.md) · [下一节](05-%E6%8E%A8%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E7%9A%84%E8%87%AA%E5%8A%A8%E6%89%A9%E7%BC%A9%E5%AE%B9%E7%AD%96%E7%95%A5.md)
