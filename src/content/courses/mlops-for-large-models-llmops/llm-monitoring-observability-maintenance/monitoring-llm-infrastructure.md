---
course: "mlops-for-large-models-llmops"
chapter: "llm-monitoring-observability-maintenance"
lesson: "monitoring-llm-infrastructure"
sourceId: 3195
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/monitoring-llm-infrastructure"
title: "监控基础设施使用情况（GPU、内存）"
description: "追踪GPU/TPU使用率、内存使用情况和网络I/O，适用于训练和推理工作负载。"
order: 2
plots: ["plots/3195-0.json", "plots/3195-1.json"]
sourceHash: "c21903428925301e8e27a203a8ec0e284f43c73316f2f553d8694a17843c0e12"
sourceCorrections: []
---

追踪LLM特有的性能指标（如延迟和吞吐量 (throughput)）能帮助我们了解用户体验，而掌握底层基础设施的运行状态对于诊断问题、优化性能和控制成本同样重要。大型语言模型对硬件有极高要求，尤其是GPU和内存。资源使用效率低下会直接导致运营成本增加和潜在的性能瓶颈。因此，密切监控基础设施不是可选项；它是有效LLMOps的基本要点。

### 为什么基础设施监控对LLM重要

LLM工作负载，无论是用于训练还是推理 (inference)，都根本上受限于可用的计算和内存资源。

- **训练：** 针对数十亿参数 (parameter)模型的分布式训练任务需要使用多个GPU节点。计算使用、内存分配或节点间通信的效率低下会大幅增加训练时间和成本。基础设施中的一个瓶颈可能导致昂贵的加速器闲置。
- **推理：** LLM服务需要大量GPU内存来存储模型权重 (weight)并在生成过程中管理动态KV缓存。密集的矩阵乘法需要GPU计算。资源不足会导致高延迟、低吞吐量 (throughput)或内存不足（OOM）错误，从而降低用户体验并可能引起服务中断。

监控基础设施使用情况有助于辨明：

- **瓶颈：** GPU是否被充分利用，还是在等待CPU、数据加载或网络？
- **效率低下：** 您是否为大部分时间闲置的昂贵GPU资源买单？
- **容量限制：** GPU内存是否接近其上限，有OOM错误的风险？
- **成本驱动因素：** 基础设施的哪些部分对运营成本贡献最大？

### 重要基础设施指标

将监控工作集中在最可能影响LLM性能和成本的资源上。

#### GPU使用率

该指标通常衡量在一个特定时间样本内，一个或多个内核在GPU上执行的时间百分比。高使用率（在活动处理期间理想情况下接近100%）表明GPU的计算资源得到了有效利用。

- **低使用率：** 推理 (inference)期间GPU使用率持续偏低通常指向其他地方的瓶颈。常见原因包括：
  - 受CPU限制的预处理或后处理步骤。
  - 请求批处理效率低下（GPU在等待足够的请求以形成批次）。
  - I/O延迟（例如，在RAG系统中等待数据）。
  - 次优的推理服务器配置。
    在训练期间，低使用率可能源于数据加载问题、并行度不足或分布式设置中的同步开销。
- **监控工具：** NVIDIA系统管理界面（`nvidia-smi`）是一个提供实时使用情况数据的标准命令行工具。云服务提供商（AWS CloudWatch、Azure Monitor、Google Cloud Monitoring）提供服务来追踪GPU随时间的使用情况，专业的监控平台通常与NVIDIA驱动程序或DCGM（数据中心GPU管理器）集成，以获取更精细的数据。

```bash
# 示例：使用nvidia-smi检查GPU使用率
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
```



![推理批处理时的GPU使用率](plots/3195-0.json)



> GPU使用率通常在活跃批处理期间飙升，并在空闲时段或批处理效率低下时在请求之间下降。

#### GPU内存使用情况

这衡量了GPU专用内存（VRAM）当前已分配的量。LLM需要大量VRAM来存储模型参数 (parameter)、中间激活，以及对推理来说非常重要的、随序列长度和批次大小增长的键值（KV）缓存。

- **重要性：** 超过可用GPU内存会导致OOM错误，造成任务失败（训练）或请求丢失（推理）。监控内存使用情况对稳定性来说必不可少。
- **因素：** 模型大小是主要因素。量化 (quantization)技术减少了这种占用。在推理期间，由于KV缓存，批次大小和最大序列长度显著影响内存需求。分页注意力等技术有助于更有效地管理内存，但仍需要仔细监控。
- **监控工具：** `nvidia-smi`提供内存使用统计信息。性能分析工具（如PyTorch Profiler或TensorFlow Profiler）可以提供按操作更详细的内存分配明细。云监控服务也追踪GPU内存使用情况。

```bash
# 示例：使用nvidia-smi检查GPU内存使用情况
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
```



![推理期间的GPU内存使用情况](plots/3195-1.json)



> 随着请求的处理和KV缓存的增长，GPU内存使用量会增加，如果有效地使用分页注意力或连续批处理等技术，可能会趋于平稳。接近总可用内存时，需要进行调查或考虑扩容。

#### CPU使用率

尽管GPU处理主要工作，CPU在LLM工作流中仍很活跃。它们管理数据加载和预处理、协调GPU操作、处理网络通信，并执行部分应用逻辑（例如，提示处理、后处理）。CPU瓶颈可能导致GPU“饥饿”，从而使得GPU使用率偏低。应监控CPU使用情况，尤其是在训练的数据密集阶段或推理期间请求率很高时。

#### 系统内存（RAM）使用情况

标准系统RAM用于操作系统进程、应用程序代码、在数据传输到GPU之前进行缓冲，并可能存储不适合VRAM的数据结构。尽管对于LLM而言，它不像VRAM那样经常成为主要瓶颈，但RAM不足可能导致过度交换或性能下降。

#### 网络I/O

在以下几种情况下，网络性能很重要：

- **分布式训练：** 高带宽和低延迟对于高效的梯度同步和节点间数据传输是必不可少的。网络瓶颈会严重限制扩展能力。
- **推理：** 处理传入请求和传出响应需要网络带宽。对于RAG系统，从向量 (vector)数据库或其他源获取上下文 (context)的网络延迟会直接影响整体响应时间。
  使用标准系统工具（`netstat`、`iperf`）或云服务提供商指标监控网络吞吐量 (throughput)（字节/秒）和延迟（毫秒）。

#### 磁盘I/O

磁盘I/O主要影响数据集和模型检查点的加载。尽管LLM通常在可能的情况下受益于将数据加载到RAM或直接加载到GPU内存，但缓慢的磁盘I/O会在训练的初始阶段或在为推理将大型模型加载到内存时造成瓶颈。如果数据加载显得缓慢，请监控磁盘读/写速度和队列长度。

### 监控工具与集成

借助多种工具进行全面的基础设施监控：

- **系统级工具：** `nvidia-smi`、`dcgm-exporter`（用于Prometheus）、`htop`、`iotop`、`iftop`提供低级别指标的快照或连续流。
- **云服务提供商工具：** AWS CloudWatch、Azure Monitor和Google Cloud Monitoring为计算实例提供集成监控，包括GPU指标、网络和磁盘I/O。在这些平台内配置仪表盘和告警。
- **编排/容器平台：** Kubernetes通过Metrics Server等工具提供资源监控机制，并与Prometheus和Grafana等系统集成，通常使用`dcgm-exporter`等导出器获取GPU指标。
- **可观测性平台：** Datadog、Grafana、New Relic和Splunk等工具可以从各种来源（云服务提供商、系统代理、应用程序日志）摄取指标，提供统一视图。在单个仪表盘中关联基础设施指标（例如，GPU使用率）与应用程序指标（例如，推理 (inference)延迟）对诊断非常有价值。

### 设置基线和告警

有效的监控不仅仅是收集数据，更在于解读数据。为您的特定模型和工作负载在典型条件下建立基线性能特点。基于这些基线，配置有意义的告警：

- **GPU使用率：** 对持续的*低*使用率（例如，在预期峰值负载期间低于30%并持续超过5分钟）或*过高*的使用率（例如，持续高于98%，表明存在潜在瓶颈）发出告警。
- **GPU内存：** 当使用量超过高水位线（例如，总VRAM的85-90%）时发出告警，以预防OOM错误。
- **网络/磁盘：** 基于预期工作负载模式，对高延迟或饱和发出告警。
- **CPU：** 如果在预期应受GPU限制的实例上，CPU使用率持续很高（例如，高于90%）时发出告警。

避免过于敏感的告警，以免造成告警疲劳。关注那些预示着真实的性能下降、即将发生的故障或明显的成本效率低下的告警。

监控基础设施使用情况是一个持续的过程。所获得的了解直接指导优化工作，例如调整批次大小、选择不同硬件、实施量化 (quantization)或改进分布式训练策略，最终带来更稳定、更高性能和更具成本效益的LLM部署。

## 参考资料

- [NVIDIA Data Center GPU Manager (DCGM) User Guide](https://docs.nvidia.com/datacenter/dcgm/latest/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  提供了监控 NVIDIA GPU 的全面文档，包括利用率、内存和其他与 LLM 工作负载相关的性能指标。
