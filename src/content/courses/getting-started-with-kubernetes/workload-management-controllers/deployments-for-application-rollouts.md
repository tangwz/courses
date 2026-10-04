---
course: "getting-started-with-kubernetes"
chapter: "workload-management-controllers"
lesson: "deployments-for-application-rollouts"
sourceId: 7311
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-kubernetes/chapter-3-workload-management-controllers/deployments-for-application-rollouts"
title: "使用 Deployment 进行应用发布"
description: "了解 Deployment 控制器，它是用于管理 ReplicaSet 并为应用提供声明式更新的高级抽象。"
order: 3
plots: []
sourceHash: "893cd5654ab4f3e7006bfb56e5a90df8d60ee165ddced9710c90fe941c675057"
sourceCorrections: []
---

Kubernetes 中的 ReplicaSet 用于维持一组稳定运行的 Pod，通过确保指定数量的相同 Pod 始终处于活动状态来保证可用性。尽管有这种能力，ReplicaSet 却有一个明显的局限：它缺乏更新这些 Pod 的内置自动化机制。在发布应用新版本时，该过程需要手动编排，例如创建一个新的 ReplicaSet 并谨慎地缩减旧的 ReplicaSet。这种手动方法容易出错，并存在停机的风险。

为了解决这个问题，Kubernetes 提供了一个更高级别的控制器，称为 **Deployment**。Deployment 管理应用的整个生命周期，包括初始创建、更新和扩缩容。它将 ReplicaSet 作为构建模块来实现这些高级功能。你只需与 Deployment 对象交互，而 Deployment 控制器会反过来为你管理底层的 ReplicaSet。这种抽象是在 Kubernetes 中运行无状态应用的标准且推荐的方式。

### Deployment 与 ReplicaSet 的关系

Deployment 的主要职能是为 Pod 提供声明式更新。你无需直接管理 ReplicaSet，只需在 Deployment 清单中声明应用的期望状态即可。该状态包括容器镜像、副本数量以及应用更新的策略等详细信息。

当你对 Deployment 的 Pod 模板进行更改时，Deployment 控制器会启动一次发布（rollout）。其工作原理如下：

1. Deployment 控制器使用更新后的 Pod 规范创建一个新的 ReplicaSet。
2. 它开始逐渐扩容新的 ReplicaSet，创建运行新版本应用的新 Pod。
3. 与此同时，它缩减旧的 ReplicaSet，终止旧的 Pod。
4. 此过程持续进行，直到所有旧 Pod 都被新 Pod 替换，应用完成全面更新。

这个受控的过程称为**滚动更新（rolling update）**，它确保应用在整个更新期间保持可用。Deployment 还会保留旧的 ReplicaSet（将其副本数缩减为零），以便在新版本出现问题时，你可以轻松回滚到以前的版本。

> 此图展示了处于发布过程中的 Deployment。Deployment 控制器正在缩减旧的 ReplicaSet（运行 `nginx:1.20.2`），同时扩容新的 ReplicaSet（`nginx:1.21.6`），以达到新版本运行三个副本的期望状态。

### 控制发布策略

Deployment 提供了对更新过程的精细控制。默认策略是 `RollingUpdate`，它通过增量更新 Pod 来最大限度地减少停机时间。你可以通过 Deployment 规范中的两个参数 (parameter)来调整其行为：

- `maxUnavailable`：指定在更新过程中可以处于不可用状态的 Pod 的最大数量。它可以是一个绝对数值（如 `1`）或期望副本数的百分比（如 `25%`）。较低的数值能确保更高的可用性，但可能会减慢发布速度。
- `maxSurge`：定义可以创建的超过期望副本数量的 Pod 最大数量。与 `maxUnavailable` 类似，它可以是绝对数值或百分比。这允许系统在终止旧 Pod 之前先创建新 Pod，从而加快更新速度。

例如，如果你有一个包含 10 个副本的 Deployment，并将 `maxUnavailable` 设置为 `25%`，`maxSurge` 设置为 `25%`，Kubernetes 将确保至少有 7 个 Pod（10 - 2.5，向下取整）始终可用。它还会确保在发布过程中的任何时刻，Pod 总数（旧的和新的总和）不超过 12 个（10 + 2.5，向上取整）。

### 重建（Recreate）策略

除了 `RollingUpdate`，Deployment 还支持一种更简单的策略，称为 `Recreate`。使用此策略时，Deployment 控制器会在创建任何新版本 Pod 之前，先终止旧版本的所有 Pod。

这种方法会导致应用停机，因为会有一段时间没有任何 Pod 在运行。虽然通常不适用于生产环境，但 `Recreate` 策略在特定场景下很有用，例如：

- 可以接受短暂停机的开发环境。
- 由于数据架构迁移或其他不兼容的更改，无法容忍旧版本和新版本同时运行的应用。

通过声明式更新来管理应用生命周期，Deployment 提供了现代软件交付所需的自动化、韧性和控制力。它们是在 Kubernetes 上实现 CI/CD 流水线和 GitOps 工作流的基石性组件。

## 参考资料

- [Kubernetes in Action, Second Edition](https://www.manning.com/books/kubernetes-in-action-second-edition) — Marko Lukša, Kevin Conner (2026)
  Publisher: Manning Publications; Pages: Chapter 6
  一本关于 Kubernetes 核心概念的综合书籍，其中有专门的章节深入解释了 Deployment、ReplicaSet 和各种更新策略。
- [Production Kubernetes](https://www.oreilly.com/library/view/production-kubernetes/9781492092308/) — Josh Rosso, Rich Lander, Alexander Brand, John Harris (2021)
  Publisher: O'Reilly Media; Pages: Chapter 3: Deploying Applications
  提供在生产环境中运行 Kubernetes 的实用指导，包含与高级 Deployment 策略、可靠性和关键应用程序生命周期管理相关的内容。
