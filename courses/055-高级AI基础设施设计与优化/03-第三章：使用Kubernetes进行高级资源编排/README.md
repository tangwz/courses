# 第 3 章：第三章：使用Kubernetes进行高级资源编排

来源：[原章节](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-3-advanced-kubernetes-orchestration)

[返回课程目录](../README.md)

尽管上一章侧重于单个分布式训练任务的运行方式，但生产系统必须并发高效地管理许多此类任务。直接在云虚拟机上操作，在调度、故障容错和资源利用方面存在难题。容器编排提供了必要的抽象层来解决这些问题。

本章将详细说明如何使用Kubernetes编排大规模机器学习工作负载。您将配置生产级功能，以在共享计算集群上管理模型的整个生命周期。我们首先使用 KubeFlow 定义并自动化机器学习流水线。接着，您将学习通过高级GPU调度来管理专用硬件，这包括时间切片和多实例GPU（MIG）配置。随后，我们将通过设置集群自动扩缩来使计算资源供应与工作负载需求匹配，并通过制定使用低成本竞价实例的策略来提升运行效率。本章最后介绍实现多租户的方法，使多个团队能够安全地共享基础设施并拥有明确的资源边界。

## 小节

- 1. [使用 KubeFlow Pipelines 管理机器学习工作流程](01-%E4%BD%BF%E7%94%A8%20KubeFlow%20Pipelines%20%E7%AE%A1%E7%90%86%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E6%B5%81%E7%A8%8B.md)
- 2. [高级GPU调度与共享](02-%E9%AB%98%E7%BA%A7GPU%E8%B0%83%E5%BA%A6%E4%B8%8E%E5%85%B1%E4%BA%AB.md)
- 3. [面向动态机器学习工作负载的集群自动扩缩容](03-%E9%9D%A2%E5%90%91%E5%8A%A8%E6%80%81%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%B7%A5%E4%BD%9C%E8%B4%9F%E8%BD%BD%E7%9A%84%E9%9B%86%E7%BE%A4%E8%87%AA%E5%8A%A8%E6%89%A9%E7%BC%A9%E5%AE%B9.md)
- 4. [使用竞价实例和可抢占实例的策略](04-%E4%BD%BF%E7%94%A8%E7%AB%9E%E4%BB%B7%E5%AE%9E%E4%BE%8B%E5%92%8C%E5%8F%AF%E6%8A%A2%E5%8D%A0%E5%AE%9E%E4%BE%8B%E7%9A%84%E7%AD%96%E7%95%A5.md)
- 5. [通过命名空间、配额和优先级类实现多租户](05-%E9%80%9A%E8%BF%87%E5%91%BD%E5%90%8D%E7%A9%BA%E9%97%B4%E3%80%81%E9%85%8D%E9%A2%9D%E5%92%8C%E4%BC%98%E5%85%88%E7%BA%A7%E7%B1%BB%E5%AE%9E%E7%8E%B0%E5%A4%9A%E7%A7%9F%E6%88%B7.md)
- 6. [实践：配置GPU感知型自动扩缩组](06-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E9%85%8D%E7%BD%AEGPU%E6%84%9F%E7%9F%A5%E5%9E%8B%E8%87%AA%E5%8A%A8%E6%89%A9%E7%BC%A9%E7%BB%84.md)
