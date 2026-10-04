---
course: "planning-optimizing-ai-infrastructure"
chapter: "cost-management-and-optimization"
lesson: "right-sizing-infrastructure-for-workloads"
sourceId: 7042
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-6-cost-management-and-optimization/right-sizing-infrastructure-for-workloads"
title: "工作负载的适度基础设施配置"
description: "学习分析性能指标的过程，以选择最适合大小（和价格）的基础设施，用于您特定的AI工作负载。"
order: 6
plots: ["plots/7042-0.json"]
sourceHash: "1850a1372ee39bad029b97a3568b2b2b5e5c73141b2fa5c20cafebdaef8b706a"
sourceCorrections: []
---

选择可用的最强大GPU实例可能感觉是最保险的选择，但这往往是快速耗尽预算的最快方法。适度配置是一个有条理的过程，用于根据您的AI工作负载的具体需求，匹配CPU、GPU和内存等基础设施资源。目标是配置足够的容量以达到性能目标，同时避免为闲置资源付费。这种做法使您从“以防万一”的配置模式转向“恰好够用”的模式，这对于成本控制来说是必要的。

挑战在于避免两种常见的失败情况：配置不足和配置过度。配置不足，即没有分配足够的资源，会导致性能不佳。训练任务可能运行过慢，延长开发周期，或者推理 (inference)端点可能无法满足延迟要求，导致用户体验不佳。配置过度是成本管理中更常出现的问题，意味着您正在为您应用程序从不使用的容量付费。一个昂贵的GPU如果持续利用率只有10%，就是纯粹的财务浪费。

### 基于数据的方法进行配置

有效的适度配置不是基于猜测；它基于测量。在您选择正确的实例之前，您必须首先了解您的工作负载的资源概况。这涉及使用我们在第五章中介绍的监控和分析工具，收集代表性工作负载的数据。

首先在基准实例上运行您的训练脚本或推理 (inference)服务，并收集以下主要指标：

- **GPU利用率（%）：** 这显示了GPU处理核心的繁忙程度。在大量计算阶段持续低利用率（例如，低于50%）通常指向其他地方的瓶颈，通常是数据输入流水线或CPU处理。
- **GPU内存利用率（GB）：** 该指标告诉您GPU的显存 (VRAM)有多少用于存储您的模型、数据批次和中间激活。如果您在24 GB显存的GPU上只使用了8 GB显存，您可能正在为您不需要的内存付费。
- **CPU利用率（%）：** 注意CPU。如果CPU利用率固定在100%而GPU利用率较低，您的数据预处理或加载步骤正在使GPU处于“饥饿”状态。
- **系统内存（RAM）使用量（GB）：** 这与GPU内存不同。RAM不足可能导致操作系统使用较慢的磁盘交换空间，严重降低性能。

一旦您拥有这些数据，您就可以开始做出明智的决定。例如，如果您的分析显示GPU利用率低但CPU核心已满载，直接的解决方案不是更大的GPU。正确的做法是优化您的数据加载流水线或选择具有更高CPU与GPU比率的实例。反之，如果您的工作负载因GPU上出现“内存不足”错误而失败，您就有了选择具有更多显存的实例的明确信号。

### 训练工作负载的适度配置

训练任务通常是长时间运行的批处理任务，主要目标是在可接受的时间内以最小的总成本完成训练。成本不仅是实例的小时费率，而是整个任务的总成本。


$$
\text{总任务成本} = \text{实例小时费率} \times \text{完成所需小时数}
$$


一个更快、更昂贵的GPU可能如此快地完成任务，以至于其总成本低于运行时间更长、更便宜、更慢的GPU。为了找到最佳平衡点，您应该进行小型实验，在几种不同的实例类型上训练几个周期，并推算出总时间和成本。



![训练任务的总成本分析](plots/7042-0.json)



> 三种不同GPU实例的总任务成本。虽然A100实例每小时费率最高，但它完成任务最快，从而导致总成本最低。

### 推理 (inference)工作负载的适度配置

推理工作负载具有不同特点。它们通常期望持续运行，以低延迟服务用户请求，并处理可变流量负载。在这里，主要指标是每美元性能，通常以每美元每秒推理数衡量。

对于推理而言，大型GPU经常是过度配置。量化 (quantization)和模型编译等技术可以使模型足够小巧和高效，以便在CPU或专用推理加速器上运行，这些平台能提供低得多的成本。推理的适度配置过程包括：

1. **负载测试：** 使用负载测试工具向部署在不同实例类型上的模型发送模拟流量。
2. **衡量性能：** 记录每个实例在延迟超过您的服务级别目标（SLO）之前可以处理的最大吞吐量 (throughput)（每秒请求数）。
3. **计算成本效益：** 将吞吐量除以实例小时成本，以找到最具成本效益的选项。
4. **实施自动伸缩：** 对于生产系统，非常重要地将您选择的实例类型置于自动伸缩组中。这使得基础设施能够根据实时流量自动添加或删除实例，确保您只为您在任何给定时刻所需的容量付费。

> 一个用于推理服务的自动伸缩组。负载均衡器分发流量，而监控服务根据需求调整活跃实例的数量。

### 一个迭代过程

适度配置不是一次性设置。它是一个持续的循环。您的模型会改变，您的代码会更新，您的用户流量模式会变化。三个月前是最佳的决定，今天可能效率低下。您应该将定期重新分析和重新评估纳入您的MLOps生命周期，以确保您的基础设施与工作负载的实际需求保持一致，保持高性能和低成本。

## 参考资料

- [Cloud FinOps: Collaborative, Real-Time Cloud Financial Management](https://book.cloudfinops.io/) — J. R. R. Allen, R. B. H. Kirk, J. W. K. Walker (2022)
  Publisher: O'Reilly Media
  提供了一个全面的云财务管理框架，其原则直接适用于优化人工智能基础设施成本。
- [NVIDIA Nsight Systems User's Guide](https://docs.nvidia.com/nsight-systems/) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  领先的分析工具的官方文档，对于理解和优化GPU加速的深度学习工作负载至关重要。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  涵盖构建生产级机器学习系统的各个方面，包括基础设施考量、资源管理以及训练和推理的成本优化。
- [Amazon SageMaker Endpoint Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html) — Amazon Web Services (AWS) (2024)
  官方文档，说明如何根据流量自动调整推理实例的数量，这是针对可变推理工作负载进行资源优化管理的关键组成部分。
