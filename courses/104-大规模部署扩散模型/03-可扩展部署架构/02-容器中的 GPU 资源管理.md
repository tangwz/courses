# 容器中的 GPU 资源管理

来源：[原文](https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-3-infrastructure-scalable-deployment/gpu-resource-management-containers)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Docker 将扩散模型应用程序容器化是部署过程中的一个主要步骤。在此之后，一个主要考虑事项是确保应用程序能够有效获取并利用所需的专用硬件，特别是 GPU。默认情况下，标准 Docker 容器与主机的硬件设备（包括 GPU）是隔离的。这种隔离通常提供安全性和可移植性，但对于扩散模型推理 (inference)这类计算密集型任务，直接使用 GPU 加速对性能来说至关重要。

### 弥合差距：NVIDIA 容器工具包

为了让容器能够使用 NVIDIA GPU，您需要一种机制来连接容器的隔离环境与主机系统的 NVIDIA 驱动和硬件。这正是 **NVIDIA 容器工具包** 的作用。

该工具包原名 NVIDIA Docker，它通过添加必要组件来扩展标准容器运行时（如 Docker Engine 或 containerd），从而使得 NVIDIA GPU 可以在容器内部使用。它通过以下组合来达到目的：

1. **运行时集成：** 它将钩子程序安装到容器运行时的生命周期中。当启动一个请求 GPU 访问的容器时，这些钩子会拦截该过程。
2. **驱动挂载：** 该工具包自动识别主机系统上所需的 NVIDIA 驱动库和必需的设备节点（例如 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`）。
3. **命名空间注入：** 它有选择地将这些所需的驱动文件和设备节点挂载到容器的文件系统和设备命名空间中。

重要的一点是，这意味着您**无需**将 NVIDIA 驱动程序打包*到*容器镜像中。容器使用主机上安装的驱动程序。这种方法使容器镜像更小，并确保与主机硬件设置的兼容性，但它也引入了一个依赖：容器依赖于运行它的主机上存在兼容的驱动版本。

### 使用 Docker 请求 GPU

直接使用 Docker 运行容器时，您可以通过 `--gpus` 标志来请求 GPU 访问。该标志会指示 NVIDIA 容器工具包（通过 Docker 守护进程）执行必需的设置。

```bash
# 示例：运行一个请求访问所有可用 GPU 的容器
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 示例：按索引请求 2 个特定 GPU
docker run --rm --gpus '"device=0,1"' my-diffusion-model-image:latest python infer.py --prompt "A cat wearing a hat"

# 示例：根据功能请求 GPU（例如，计算能力）
# docker run --rm --gpus 'all,capabilities=compute,utility' ...
```

在容器内部运行的 `nvidia-smi` 命令应列出可用的 GPU，确认工具包已成功暴露硬件。

### Kubernetes 中的 GPU 管理

大规模部署容器化应用程序时，像 Kubernetes 这样的编排平台变得不可或缺。Kubernetes 需要自己的机制来理解和管理跨节点集群的专用硬件，如 GPU。这由**设备插件**处理。

Kubernetes 的 NVIDIA 设备插件是部署在集群中每个支持 GPU 的节点上的组件（通常作为 DaemonSet）。它的职责包括：

1. **发现：** 它检测节点上可用的 NVIDIA GPU 数量和类型。
2. **健康检查：** 它监控 GPU 的健康状况。
3. **资源报告：** 它使用特定的资源名称（通常是 `nvidia.com/gpu`）将 GPU 注册为 Kubelet（节点代理）的可分配资源。
4. **服务提供：** 它提供一个 gRPC 服务，当请求 GPU 的容器（Pod）需要在节点上调度时，Kubelet 会调用该服务，指定应分配哪些设备。

设备插件运行后，您可以在 Kubernetes Pod 规范中使用标准的 `resources.requests` 和 `resources.limits` 字段来请求 GPU。

```yaml
# 示例 Kubernetes Pod 规范请求一个 GPU
apiVersion: v1
kind: Pod
metadata:
  name: diffusion-inference-pod
spec:
  containers:
  - name: diffusion-worker
    image: my-diffusion-model-image:latest
    command: ["python", "worker.py"]
    resources:
      limits:
        nvidia.com/gpu: 1 # 请求 1 个 GPU
      requests:
        nvidia.com/gpu: 1 # 请求 1 个 GPU (可选，通常与限制相同)
  # 如果 GPU 节点需要，添加节点选择器或容忍度
  # nodeSelector:
  #   cloud.google.com/gke-accelerator: nvidia-tesla-t4
```

当提交此 Pod 时，Kubernetes 调度器会查找至少有一个 `nvidia.com/gpu` 资源可用的节点（由设备插件报告）。一旦调度完成，选定节点上的 Kubelet 会与设备插件和容器运行时（通过 NVIDIA 容器工具包）交互，在容器启动前将正确的 GPU 设备挂载到容器中。

> 该图说明了使主机 GPU 在容器内可用所涉及的组件，包括通过 Docker 直接访问以及通过 Kubernetes 使用设备插件进行编排。

### 重要考量

- **驱动兼容性：** 最常见的问题是编译到您的应用程序中（或包含在您的基础容器镜像中，例如 `nvidia/cuda:<version>`）的 CUDA 工具包版本与主机上安装的 NVIDIA 驱动程序版本不匹配。确保主机驱动程序足够新，以支持容器内使用的 CUDA 版本。NVIDIA 提供兼容性矩阵供参考。
- **资源可见性和监控：** 尽管容器内的 `nvidia-smi` 会显示已分配的 GPU，但监控整个集群的 GPU 总体使用率、内存使用量和温度通常需要将节点级指标（通常由设备插件或 `dcgm-exporter` 等工具暴露）集成到您的监控堆栈（例如 Prometheus、Grafana）中。这对于自动扩缩和性能调整非常重要，这将在稍后介绍。
- **资源限制与请求：** 在 Kubernetes 中，对于专用 GPU 工作负载，将 `nvidia.com/gpu` 的 `requests` 和 `limits` 都设置为相同的值（通常是 `1`）很常见。这保证了 Pod 的资源。与 CPU/内存不同，GPU 通常不被视为可压缩或以相同方式超额订阅的资源。
- **GPU 共享（高级）：** 尽管标准的 Kubernetes 方法是分配整个 GPU，但 NVIDIA 在较新的硬件（例如 A100、H100）上提供 Multi-Instance GPU (MIG) 等技术，可以将单个物理 GPU 分割成多个更小、完全隔离的 GPU 实例。存在 Kubernetes 设备插件来将这些 MIG 实例作为不同的资源进行管理。时间切片是另一种隔离性较低的机制，多个容器共享 GPU 计算时间，由特定的设备插件配置管理。这些是高级技术，通常用于当单个工作负载无法完全占用一个 GPU 时，优化其使用率。

有效管理容器中的 GPU 资源对于可靠且高效地部署扩散模型是根本。通过使用 NVIDIA 容器工具包和 Kubernetes 设备插件，您可以将 GPU 加速集成到您的容器化工作流程中，使得优化后的模型能够以所需的规模和速度执行推理 (inference)任务。这种基础设施为构建可扩展且具备韧性的推理服务奠定了基础。

## 参考资料

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  描述了 NVIDIA Container Toolkit 的架构和使用方法，以实现在 Docker 和其他容器运行时中访问 GPU。
- [NVIDIA Device Plugin for Kubernetes](https://github.com/NVIDIA/k8s-device-plugin) — NVIDIA Corporation (2024)
  描述了如何安装和使用 NVIDIA Device Plugin，以便在 Kubernetes 集群中将 GPU 作为可分配资源公开。
- [CUDA Compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver-compatibility) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  提供了 CUDA Toolkit 版本与 NVIDIA GPU 驱动之间的兼容性信息，对于系统设置很重要。
- [Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) — Kubernetes Authors (2024)
  介绍了 Kubernetes 设备插件的通用信息及其在管理专用硬件资源方面的作用。

---

[上一节](01-%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8BDocker%E5%AE%B9%E5%99%A8%E5%8C%96.md) · [下一节](03-%E4%BD%BF%E7%94%A8%20Kubernetes%20%E8%BF%9B%E8%A1%8C%E7%BC%96%E6%8E%92.md)
