# 在Kubernetes集群中管理GPU资源

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-4-containerization-orchestration-for-ml/managing-gpu-resources-kubernetes)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然标准的Kubernetes集群在管理CPU和内存方面表现出色，但它默认不识别GPU。要运行加速的机器学习 (machine learning)工作负载，您必须明确启用集群来检测、管理并将这些专用硬件资源分配给您的容器。这可以通过Kubernetes设备插件框架来实现，该系统允许第三方硬件供应商向Kubernetes调度器公布其资源。

对于机器学习，最常见的实现是NVIDIA设备插件。该插件在集群中每个配备GPU的节点上运行，检测可用的NVIDIA GPU，并将其作为可调度资源报告给`kubelet`（Kubernetes的主要节点代理）。一旦控制平面了解这些资源，您就可以在Pod规范中直接请求它们，就像请求CPU或内存一样。

### 设备插件工作流程

暴露和调度GPU的过程涉及多个相互协作的组件。设备插件充当物理硬件和Kubernetes控制平面之间的桥梁。

> Kubernetes组件之间进行GPU调度的交互。设备插件向Kubelet公布GPU，允许调度器将请求GPU资源的Pod放置到正确的节点上。

### 节点设置

在插件工作之前，每个GPU节点都必须在宿主操作系统上安装两个必要组件：

1. **NVIDIA 驱动程序**：插件不安装驱动程序。它依赖于宿主系统拥有正确的专有NVIDIA驱动程序才能与硬件进行交互。
2. **NVIDIA Container Toolkit**：此工具包使Docker或containerd等容器运行时能够在容器内部访问GPU。它处理将宿主系统上的必要驱动文件和设备节点挂载到容器隔离环境中的底层工作。

如果没有这些前提条件，设备插件可能可以运行，但您的Pod将无法启动或无法在运行时访问GPU。

### 部署NVIDIA设备插件

部署NVIDIA设备插件的标准方法是使用Kubernetes `DaemonSet`。DaemonSet确保插件的Pod副本在集群中的每个节点（或您指定的部分节点）上运行。当一个带有GPU的新节点加入集群时，DaemonSet会自动将插件Pod部署到该节点上。

以下是部署NVIDIA设备插件的简化YAML清单：

```yaml
# nvidia-device-plugin-ds.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
          # 如果控制平面节点有GPU，允许插件在其上运行
      tolerations:
      - node-role.kubernetes.io/control-plane
        operator: Exists
        effect: NoSchedule
      - "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.14.1
        name: nvidia-device-plugin-ctr
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop: ["ALL"]
            # 挂载必要的宿主路径，以便插件与驱动和kubelet通信
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
```

您可以使用 `kubectl apply -f nvidia-device-plugin-ds.yaml` 命令应用此清单。部署后，您可以通过 `kubectl get pods -n kube-system -o wide` 命令验证插件是否在您的启用了GPU的节点上运行。

### 在Pod中请求GPU

随着插件的运行，您的集群现在将GPU视为由 `nvidia.com/gpu` 标识的有限资源类型。要调度使用GPU的Pod，您必须在容器规范的 `resources` 部分中请求它。

我们来看一个需要一个GPU的PyTorch应用的Pod清单：

```yaml
# pytorch-pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-gpu-pod
spec:
  containers:
  - name: pytorch-container
    image: pytorch/pytorch:latest-cuda
        # 保持容器运行并测试GPU访问的命令
    command: ["/bin/sh", "-c"]
    args: ["python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'; sleep 3600"]
    resources:
      limits:
            # 这是对一个NVIDIA GPU的资源请求
        nvidia.com/gpu: 1
```

当您应用此清单时，Kubernetes调度器将只考虑至少有一个可分配 `nvidia.com/gpu` 资源的节点。如果找到合适的节点，调度器就会将Pod分配给它。该节点上的`kubelet`随后会在启动容器之前，为其分配一个特定的GPU。

### 验证GPU分配和使用情况

您可以通过运行 `kubectl describe node <your-gpu-node-name>` 命令来确认Kubernetes是否了解您的节点的GPU。在 `容量` 和 `可分配` 部分下查找 `nvidia.com/gpu` ：

```text
容量:
  cpu:                96
  ephemeral-storage:  99264Mi
  memory:             527699Mi
  nvidia.com/gpu:     4
  pods:               110
可分配:
  cpu:                96
  ephemeral-storage:  91497Mi
  memory:             517200Mi
  nvidia.com/gpu:     4
  pods:               110
```

部署Pod后，您可以通过在运行中的容器内部执行 `nvidia-smi` 命令来确认它正在使用GPU。

```bash
# 在Pod内部执行 nvidia-smi
$ kubectl exec -it pytorch-gpu-pod -- nvidia-smi

# 预期输出显示GPU和一个正在运行的进程
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    驱动版本: 515.65.01    CUDA 版本: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  名称        持久模式| 总线ID        显示应用程序 | 可变非纠错ECC |
| 风扇  温度  性能状态  功耗:使用/上限|         内存使用率 | GPU利用率  计算模式 |
|                               |                      |               MIG模式 |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:07:00.0 Off |                    0 |
| N/A   32C    P0    56W / 400W |      1MiB / 40960MiB |      0%      默认 |
|                               |                      |             禁用 |
+-------------------------------+----------------------+----------------------+
...
```

此输出的存在确认容器已成功获得GPU的访问权限，弥合了您的容器化机器学习 (machine learning)应用与底层硬件之间的差距。这种设置对于在Kubernetes上构建可扩展且高效的训练和推理 (inference)系统来说非常重要。

## 参考资料

- [Extend Kubernetes with Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/) — Kubernetes Authors (2024)
  Kubernetes官方文档，详细介绍了设备插件框架、其架构以及它如何将专用硬件与Kubernetes集成。
- [NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA容器工具包的官方安装和配置指南，它对于在容器内部启用GPU访问至关重要。
- [NVIDIA GPU Operator Overview](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html) — NVIDIA (2024)
  Publisher: NVIDIA Corporation
  NVIDIA关于在Kubernetes中部署和管理所有NVIDIA软件组件（包括设备插件）以配置GPU的全面文档。

---

[上一节](04-Kubernetes%20%E7%BB%84%E4%BB%B6%EF%BC%9APod%E3%80%81Service%E3%80%81Deployment.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20Kubeflow%20%E6%9E%84%E5%BB%BA%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AE%A1%E9%81%93.md)
