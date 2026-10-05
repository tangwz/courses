# Kubernetes 入门教程

来源：[Kubernetes 入门教程](https://apxml.com/zh/courses/getting-started-with-kubernetes)

本课程面向已有容器化常识的工程师与开发人员，提供 Kubernetes 技术入门指导。你将学习如何操作该系统的核心组件来部署、管理并扩缩应用。内容结合命令行实战案例，讲解 Pod、Deployment、Service 及配置管理。完成学习后，你将能熟练处理容器化工作负载，并理解其作为现代应用部署规范的架构逻辑。

预计学时：8 小时

先修要求：具备容器化常识

## 课程目录

### 1. [Kubernetes 基础知识](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/README.md)

- 1. [什么是容器编排](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/01-%E4%BB%80%E4%B9%88%E6%98%AF%E5%AE%B9%E5%99%A8%E7%BC%96%E6%8E%92.md)
- 2. [Kubernetes 架构：控制平面](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/02-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E6%8E%A7%E5%88%B6%E5%B9%B3%E9%9D%A2.md)
- 3. [Kubernetes 架构：工作节点](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/03-Kubernetes%20%E6%9E%B6%E6%9E%84%EF%BC%9A%E5%B7%A5%E4%BD%9C%E8%8A%82%E7%82%B9.md)
- 4. [kubectl 的作用](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/04-kubectl%20%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 5. [声明式与指令式管理](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/05-%E5%A3%B0%E6%98%8E%E5%BC%8F%E4%B8%8E%E6%8C%87%E4%BB%A4%E5%BC%8F%E7%AE%A1%E7%90%86.md)
- 6. [搭建本地 Kubernetes 集群](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/06-%E6%90%AD%E5%BB%BA%E6%9C%AC%E5%9C%B0%20Kubernetes%20%E9%9B%86%E7%BE%A4.md)
- 7. [动手实践：集群检查](01-Kubernetes%20%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%9B%86%E7%BE%A4%E6%A3%80%E6%9F%A5.md)

### 2. [管理 Pod](02-%E7%AE%A1%E7%90%86%20Pod/README.md)

- 1. [Pod 抽象层](02-%E7%AE%A1%E7%90%86%20Pod/01-Pod%20%E6%8A%BD%E8%B1%A1%E5%B1%82.md)
- 2. [在 YAML 中编写 Pod 清单](02-%E7%AE%A1%E7%90%86%20Pod/02-%E5%9C%A8%20YAML%20%E4%B8%AD%E7%BC%96%E5%86%99%20Pod%20%E6%B8%85%E5%8D%95.md)
- 3. [单容器与多容器 Pod](02-%E7%AE%A1%E7%90%86%20Pod/03-%E5%8D%95%E5%AE%B9%E5%99%A8%E4%B8%8E%E5%A4%9A%E5%AE%B9%E5%99%A8%20Pod.md)
- 4. [Pod 的生命周期](02-%E7%AE%A1%E7%90%86%20Pod/04-Pod%20%E7%9A%84%E7%94%9F%E5%91%BD%E5%91%A8%E6%9C%9F.md)
- 5. [使用 kubectl 进行 Pod 操作](02-%E7%AE%A1%E7%90%86%20Pod/05-%E4%BD%BF%E7%94%A8%20kubectl%20%E8%BF%9B%E8%A1%8C%20Pod%20%E6%93%8D%E4%BD%9C.md)
- 6. [健康检查：存活探针与就绪探针](02-%E7%AE%A1%E7%90%86%20Pod/06-%E5%81%A5%E5%BA%B7%E6%A3%80%E6%9F%A5%EF%BC%9A%E5%AD%98%E6%B4%BB%E6%8E%A2%E9%92%88%E4%B8%8E%E5%B0%B1%E7%BB%AA%E6%8E%A2%E9%92%88.md)
- 7. [练习：部署与检查 Pod](02-%E7%AE%A1%E7%90%86%20Pod/07-%E7%BB%83%E4%B9%A0%EF%BC%9A%E9%83%A8%E7%BD%B2%E4%B8%8E%E6%A3%80%E6%9F%A5%20Pod.md)

### 3. [使用控制器管理工作负载](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/README.md)

- 1. [Kubernetes 控制器简介](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/01-Kubernetes%20%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%80%E4%BB%8B.md)
- 2. [使用 ReplicaSet 确保 Pod 可用性](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/02-%E4%BD%BF%E7%94%A8%20ReplicaSet%20%E7%A1%AE%E4%BF%9D%20Pod%20%E5%8F%AF%E7%94%A8%E6%80%A7.md)
- 3. [使用 Deployment 进行应用发布](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/03-%E4%BD%BF%E7%94%A8%20Deployment%20%E8%BF%9B%E8%A1%8C%E5%BA%94%E7%94%A8%E5%8F%91%E5%B8%83.md)
- 4. [定义 Deployment 清单](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/04-%E5%AE%9A%E4%B9%89%20Deployment%20%E6%B8%85%E5%8D%95.md)
- 5. [执行 Deployment 更新与回滚](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/05-%E6%89%A7%E8%A1%8C%20Deployment%20%E6%9B%B4%E6%96%B0%E4%B8%8E%E5%9B%9E%E6%BB%9A.md)
- 6. [检查 Deployment 状态](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/06-%E6%A3%80%E6%9F%A5%20Deployment%20%E7%8A%B6%E6%80%81.md)
- 7. [动手实践：创建与更新 Deployment](03-%E4%BD%BF%E7%94%A8%E6%8E%A7%E5%88%B6%E5%99%A8%E7%AE%A1%E7%90%86%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%88%9B%E5%BB%BA%E4%B8%8E%E6%9B%B4%E6%96%B0%20Deployment.md)

### 4. [Kubernetes 网络](04-Kubernetes%20%E7%BD%91%E7%BB%9C/README.md)

- 1. [Kubernetes 网络模型](04-Kubernetes%20%E7%BD%91%E7%BB%9C/01-Kubernetes%20%E7%BD%91%E7%BB%9C%E6%A8%A1%E5%9E%8B.md)
- 2. [Service 简介](04-Kubernetes%20%E7%BD%91%E7%BB%9C/02-Service%20%E7%AE%80%E4%BB%8B.md)
- 3. [Service 类型：ClusterIP、NodePort 与 LoadBalancer](04-Kubernetes%20%E7%BD%91%E7%BB%9C/03-Service%20%E7%B1%BB%E5%9E%8B%EF%BC%9AClusterIP%E3%80%81NodePort%20%E4%B8%8E%20LoadBalancer.md)
- 4. [定义 Service 清单](04-Kubernetes%20%E7%BD%91%E7%BB%9C/04-%E5%AE%9A%E4%B9%89%20Service%20%E6%B8%85%E5%8D%95.md)
- 5. [集群内的服务发现](04-Kubernetes%20%E7%BD%91%E7%BB%9C/05-%E9%9B%86%E7%BE%A4%E5%86%85%E7%9A%84%E6%9C%8D%E5%8A%A1%E5%8F%91%E7%8E%B0.md)
- 6. [通过 Ingress 暴露服务](04-Kubernetes%20%E7%BD%91%E7%BB%9C/06-%E9%80%9A%E8%BF%87%20Ingress%20%E6%9A%B4%E9%9C%B2%E6%9C%8D%E5%8A%A1.md)
- 7. [动手实践：暴露应用程序](04-Kubernetes%20%E7%BD%91%E7%BB%9C/07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9A%B4%E9%9C%B2%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md)

### 5. [配置与持久化存储](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/README.md)

- 1. [使用 ConfigMap 管理配置](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/01-%E4%BD%BF%E7%94%A8%20ConfigMap%20%E7%AE%A1%E7%90%86%E9%85%8D%E7%BD%AE.md)
- 2. [使用 Secret 处理凭据](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/02-%E4%BD%BF%E7%94%A8%20Secret%20%E5%A4%84%E7%90%86%E5%87%AD%E6%8D%AE.md)
- 3. [Kubernetes 存储概念](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/03-Kubernetes%20%E5%AD%98%E5%82%A8%E6%A6%82%E5%BF%B5.md)
- 4. [PersistentVolumes 和 PersistentVolumeClaims](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/04-PersistentVolumes%20%E5%92%8C%20PersistentVolumeClaims.md)
- 5. [将存储卷挂载到 Pod](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/05-%E5%B0%86%E5%AD%98%E5%82%A8%E5%8D%B7%E6%8C%82%E8%BD%BD%E5%88%B0%20Pod.md)
- 6. [实践：注入配置并挂载卷](05-%E9%85%8D%E7%BD%AE%E4%B8%8E%E6%8C%81%E4%B9%85%E5%8C%96%E5%AD%98%E5%82%A8/06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%B3%A8%E5%85%A5%E9%85%8D%E7%BD%AE%E5%B9%B6%E6%8C%82%E8%BD%BD%E5%8D%B7.md)

## 学习目标

- **Kubernetes 架构**：说明 Kubernetes 控制平面与工作节点的组成部分。
- **Pod 管理**：使用 YAML 清单定义、部署并查看单容器或多容器 Pod。
- **应用部署**：使用 Deployment 和 ReplicaSet 管理应用生命周期、扩缩容及版本更新。
- **服务发现与网络**：通过 Service 和 Ingress 在集群内外公开应用访问入口。
- **配置与存储**：管理应用配置与机密数据，并为有状态应用配置持久化存储。
