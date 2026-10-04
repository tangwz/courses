---
course: "getting-started-with-kubernetes"
chapter: "kubernetes-fundamentals"
lesson: "setting-up-local-kubernetes-cluster"
sourceId: 7300
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-1-kubernetes-fundamentals/setting-up-local-kubernetes-cluster"
title: "搭建本地 Kubernetes 集群"
description: "使用 Minikube 或 Kind 等工具搭建本地 Kubernetes 开发环境的分步指南。"
order: 6
plots: []
sourceHash: "bf5f66e6939e989206f4bdf47c9cb0954285c4e2951eb5587e570b2cd40f1106"
sourceCorrections: []
---

在开发机器上搭建轻量级的本地 Kubernetes 集群是学习、测试和应用开发的理想选择。本地集群提供了一个功能齐全、隔离的 Kubernetes 环境，你可以通过简单的命令创建和销毁它，且不产生任何费用。虽然生产环境中的 Kubernetes 集群运行在云提供商或本地服务器上，但与真实的集群进行交互是理解 Kubernetes 行为和功能的有效方式。

### 选择本地集群工具

有几种出色的开源工具可用于在本地运行 Kubernetes。在本课程中，我们将侧重于两个最受欢迎的选项：Minikube 和 Kind。两者都能在你的机器上提供完整的 Kubernetes 集群，但实现方式不同。

- **Minikube** 在本地机器的虚拟机 (VM) 或容器内运行单节点 Kubernetes 集群。它是本地 Kubernetes 开发中最成熟的项目之一，提供了一个稳定且功能丰富的环境，能够很好地模拟大型设置。
- **Kind** (Kubernetes IN Docker) 使用 Docker 容器作为“节点”来运行多节点 Kubernetes 集群。它以启动速度快而著称，特别适用于测试集群配置或在 CI/CD 流水线中运行集成测试。

两者之间的选择通常取决于个人偏好和使用场景。由于使用虚拟机，Minikube 提供的环境隔离性稍好，而 Kind 的创建和销毁速度通常更快。对于本课程而言，使用哪种工具都非常合适。

> 本地 Kubernetes 集群可以使用不同的底层技术进行配置。Minikube 通常使用专用的虚拟机或容器，而 Kind 则在宿主机上将集群节点配置为 Docker 容器。

### 前提条件

在开始之前，请确保你的机器上已安装以下软件：

1. **容器运行时：** 需要 Docker Desktop、Colima 或 Podman 等工具。Minikube 和 Kind 都依赖容器运行时才能工作。
2. **`kubectl`：** Kubernetes 命令行工具。如果你尚未安装，请参考 Kubernetes 官方文档中针对你操作系统的说明进行安装。

### 选项 1：使用 Minikube 进行搭建

Minikube 是一个简单易用的入门工具。

1. **安装 Minikube：** 按照 Minikube 官网上的安装指南在系统中安装命令行界面 (CLI)。
2. **启动集群：** 打开终端并运行 `minikube start` 命令。

   ```bash
   minikube start
   ```

   此命令会执行以下操作：下载必要的 Kubernetes 组件镜像，创建用于托管集群的容器或虚拟机，并在其中配置完整的单节点 Kubernetes 控制平面和工作节点。首次运行时，此过程可能需要几分钟。

   > **注意：** Minikube 会尝试自动检测最适合你系统的驱动程序（如 Docker、Hyper-V、KVM）。你也可以使用 `--driver` 参数 (parameter)明确指定，例如：`minikube start --driver=docker`。
3. **检查状态：** 命令完成后，你可以验证集群是否正在运行。

   ```bash
   minikube status
   ```

   你应该会看到表示主机 (host)、kubelet 和 API 服务器 (apiserver) 都在运行的输出。

   ```text
   minikube
   type: Control Plane
   host: Running
   kubelet: Running
   apiserver: Running
   kubeconfig: Configured
   ```

   Minikube 会自动配置 `kubectl` 与这个新的本地集群通信，因此不需要额外的配置。

### 选项 2：使用 Kind 进行搭建

Kind 是另一个出色的替代方案，如果你已经安装了 Docker Desktop，则更为方便。

1. **安装 Kind：** 按照 Kind 官网上的安装说明针对你的操作系统进行安装。
2. **创建集群：** 安装 Kind 后，只需一个命令即可创建集群。

   ```bash
   kind create cluster
   ```

   此命令会拉取一个预构建的节点镜像，其中包含了所有必要的 Kubernetes 组件。然后它会启动充当集群节点的 Docker 容器。默认情况下，它会创建一个运行控制平面组件的单节点集群。
3. **列出集群：** 你可以通过列出所有可用的 Kind 集群来验证集群是否创建成功。

   ```bash
   kind get clusters
   ```

   输出将显示默认的集群名称 `kind`。

   ```text
   kind
   ```

   与 Minikube 一样，Kind 会自动更新你的 `kubeconfig` 文件，以便 `kubectl` 准备好与新集群进行交互。

### 验证集群连接

无论选择哪种工具，最后一步都是使用 `kubectl` 来确认你可以与集群的 API 服务器通信。这是确保一切配置正确的基本检查。

运行 `kubectl cluster-info` 获取控制平面及其服务的地址。

```bash
kubectl cluster-info
```

输出将显示 Kubernetes 控制平面主节点和 CoreDNS（用于集群内 DNS 服务的组件）的 URL。

```text
Kubernetes control plane is running at https://127.0.0.1:54163
CoreDNS is running at https://127.0.0.1:54163/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy
```

接下来，检查集群中的节点。由于你创建的是单节点集群，你应该会看到列出了一个节点。

```bash
kubectl get nodes
```

输出将显示一个处于 `Ready` 状态的节点，该节点同时承担控制平面的角色。

```text
NAME                 STATUS   ROLES           AGE   VERSION
kind-control-plane   Ready    control-plane   5m12s v1.29.1
```

如果你看到类似的输出，说明你的本地 Kubernetes 集群已运行并可以使用。你现在拥有了一个可以部署应用和研究 Kubernetes API 的完整环境，这正是你在接下来的实践环节中要做的。

### 清理工作

完成开发任务后，你可以停止或删除本地集群以释放系统资源。

- **对于 Minikube：**

  - 停止集群但不删除它：`minikube stop`
  - 彻底删除集群及其数据：`minikube delete`
- **对于 Kind：**

  - 删除默认集群：`kind delete cluster`

## 参考资料

- [Minikube Documentation](https://minikube.sigs.k8s.io/docs/) — The Kubernetes Authors (2025)
  Minikube 的官方文档，涵盖了安装、使用以及运行本地 Kubernetes 集群的各种驱动程序。
- [Kind (Kubernetes IN Docker) Documentation](https://kind.sigs.k8s.io/docs/) — The Kubernetes Authors (2025)
  Kind 的官方文档，提供了在 Docker 容器中安装和管理本地多节点 Kubernetes 集群的说明。
- [Install and Set Up kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) — The Kubernetes Authors (2023)
  Kubernetes 官方关于安装和配置 `kubectl` 的指南，`kubectl` 是用于控制 Kubernetes 集群的命令行界面。
- [Kubernetes Up & Running: Dive into the Future of Infrastructure](https://www.oreilly.com/library/view/kubernetes-up-running/9781492046522/) — Kelsey Hightower, Brendan Burns, and Joe Beda (2019)
  Publisher: O'Reilly Media
  一本广受认可的书籍，提供了对 Kubernetes 及其核心组件和操作实践的坚实介绍，有助于理解本地集群的背景。
