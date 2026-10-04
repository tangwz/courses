# 高级GPU调度与共享

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration/advanced-gpu-scheduling-sharing)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管Kubernetes可以将Pod调度到带有GPU的节点，但其默认行为将每个加速器视为一个不可分割的单元。这种“全有或全无”的分配方式经常导致显著的资源浪费，因为从交互式开发到轻量级推理 (inference)的许多ML任务都不需要现代GPU的全部算力 (compute)。为了构建一个经济高效且运行顺畅的ML平台，您必须采用更精密的调度和共享机制。

在这一点上，适用于Kubernetes的NVIDIA GPU Operator变得不可或缺。它自动管理所有必要的NVIDIA软件组件，包括驱动程序、容器工具包和设备插件，这些组件将GPU公开为可调度的资源。我们将在**此之上**实行两种高级共享策略：基于软件的时间切片和基于硬件的多实例GPU (MIG)。

### NVIDIA GPU Operator：生产环境的基准

在划分资源之前，您需要一种管理它们的机制。NVIDIA GPU Operator是生产环境的标准。它使用Kubernetes中的Operator模式来管理集群中每个节点上GPU相关软件的生命周期。一旦安装，Kubernetes调度器就能识别`nvidia.com/gpu`资源，允许您在Pod配置中请求GPU。

一个请求完整GPU的基本Pod如下所示：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: full-gpu-training-pod
spec:
  containers:
  - name: cuda-container
    image: nvidia/cuda:11.8.0-base-ubuntu22.04
    command: ["/bin/bash", "-c", "--"]
    args: [ "while true; do nvidia-smi; sleep 30; done;" ]
    resources:
      limits:
        nvidia.com/gpu: 1 # 请求一个完整、独占的GPU
```

这对于大型训练任务运作良好，但对小型任务而言效率不高。

### 通过时间切片共享GPU

时间切片允许多个容器共享一个物理GPU。GPU的调度机制在不同容器的进程之间快速切换上下文 (context)，给每个进程分配一部分GPU执行时间。这是一种软件层面的方案，不提供内存隔离，但对于提高非计算密集型或对延迟不敏感的工作负载的利用率非常有效，例如：

- 用于模型开发和检查的Jupyter Notebook。
- 低流量推理 (inference)服务。
- 并行运行多个小型实验。

您可以通过向GPU Operator应用配置来启用时间切片。此配置定义GPU应如何划分。例如，您可以指定将一个GPU分成三个“份额”。然后，设备插件会向Kubernetes调度器宣告此分数资源。

> 时间切片允许多个Pod共享单个物理GPU。GPU的执行上下文在进程之间切换，这对于具有间歇性计算需求的工作负载很有效，但由于上下文切换会引入性能开销。

一个请求某个份额的Pod将使用如下所示的资源限制：

```yaml
# 假设节点已配置为每个GPU提供3个时间切片
# 并宣告资源 'nvidia.com/gpu.shared'。
apiVersion: v1
kind: Pod
metadata:
  name: dev-notebook-pod
spec:
  containers:
  - name: jupyter-container
    image: jupyter/tensorflow-notebook
    resources:
      limits:
        nvidia.com/gpu.shared: 1 # 请求一个GPU份额
```

时间切片的主要权衡是性能。上下文切换会引入延迟，并且由于内存不隔离，一个“吵闹的邻居”Pod可能会耗尽GPU内存，导致共享相同GPU的其他Pod出现CUDA内存错误。

### 通过多实例GPU (MIG) 实现硬件划分

对于需要严格性能保障和安全隔离的工作负载，多实例GPU (MIG) 是更好的方案。在NVIDIA Ampere架构GPU（如A100和H100）及更新版本上可用，MIG将单个GPU划分成最多七个独立的、硬件隔离的GPU实例。

每个MIG实例都有自己的：

- 专用流式多处理器 (SM)。
- 专用内存和内存控制器。
- 专用L2缓存。

这种硬件级别的划分提供可预测的性能和强大的故障隔离。如果一个Pod的内核在其MIG实例上失败，它不会影响在同一物理GPU上不同实例上运行的其他Pod。这使得MIG成为安全多租户以及在单个GPU上托管具有严格服务质量 (QoS) 要求的多推理 (inference)模型的一种优秀技术。

> 多实例GPU (MIG) 将单个物理GPU划分为多个完全隔离的GPU实例。每个实例都有自己的专用内存、缓存和计算资源，确保可预测的性能和安全。

当节点上启用MIG时，设备插件将可用的MIG配置宣告为不同的资源类型。Pod在其资源限制中请求特定的MIG实例配置。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mig-inference-pod
spec:
  containers:
  - name: triton-server
    image: nvcr.io/nvidia/tritonserver:23.10-py3
    resources:
      limits:
        # 请求特定的MIG配置：1 GPC，10GB内存
        nvidia.com/mig-1g.10gb: 1
```

Kubernetes仅会将此Pod调度到具有可用`1g.10gb` MIG实例的节点上。

### 选择合适的调度策略

默认分配、时间切片和MIG之间的选择完全取决于您的工作负载对性能、隔离性和成本的要求。

| 特性 | 最佳适用场景 | 隔离性 | 性能 | 硬件 |
| --- | --- | --- | --- | --- |
| **默认（1 Pod/GPU）** | 重型训练、HPC | 进程级 | 最大、专用 | 任何NVIDIA GPU |
| **时间切片** | 开发笔记本、低流量API | 无（共享内存） | 可变，有开销 | 任何NVIDIA GPU |
| **MIG** | 多租户推理 (inference)、严格QoS | 强（硬件） | 可预测、已划分 | Ampere架构及更新版本 |

掌握这些高级调度技术，您可以将由昂贵GPU组成集群，从一组单一资源转换为一个灵活、精细且高效的平台。这使得您能够在相同的共享基础设施上支持更多种类的ML工作负载，从开发到生产，直接提高资源利用率并降低运营成本。这些精细的资源定义还为集群自动扩缩器提供了必要的信号，使其做出更智能的决策，这是我们接下来会讨论的话题。

## 参考资料

- [NVIDIA GPU Operator Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html) — NVIDIA Corporation (2024)
  提供 NVIDIA GPU Operator 的安装与配置指南，该 Operator 自动化管理 GPU 软件组件并在 Kubernetes 中暴露 GPU 资源。
- [Multi-Instance GPU (MIG) User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  详细介绍 MIG 架构、设置以及适用于 Ampere 及更新 GPU 的硬件隔离 GPU 分区资源分配。
- [Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) — Kubernetes Authors (2024)
  介绍 Kubernetes 设备插件框架，硬件供应商（如 NVIDIA）使用该框架向调度器公布专用硬件资源。
- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  记录 NVIDIA Container Toolkit，其支持 GPU 加速应用程序在 Docker 和其他 OCI 兼容容器中运行。

---

[上一节](01-%E4%BD%BF%E7%94%A8%20KubeFlow%20Pipelines%20%E7%AE%A1%E7%90%86%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.md) · [下一节](03-%E9%9D%A2%E5%90%91%E5%8A%A8%E6%80%81%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E7%9A%84%E9%9B%86%E7%BE%A4%E8%87%AA%E5%8A%A8%E6%89%A9%E7%BC%A9%E5%AE%B9.md)
