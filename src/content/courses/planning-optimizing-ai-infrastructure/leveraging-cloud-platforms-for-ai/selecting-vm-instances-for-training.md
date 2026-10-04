---
course: "planning-optimizing-ai-infrastructure"
chapter: "leveraging-cloud-platforms-for-ai"
lesson: "selecting-vm-instances-for-training"
sourceId: 6997
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-3-leveraging-cloud-platforms-for-ai/selecting-vm-instances-for-training"
title: "为训练选择虚拟机实例"
description: "了解如何根据性能和成本，挑选适合模型训练的基于GPU和CPU的虚拟机实例。"
order: 3
plots: []
sourceHash: "c5b2749cc5d51c4832ac555f7824794f4936ee6c6be73f47ee9d20abf9acab7d"
sourceCorrections: []
---

在云中训练AI模型时，选择合适的虚拟机（VM）是一个主要的决定。此选择直接影响训练时间、成本，甚至您项目的可行性。这需要在GPU的计算能力、CPU的数据处理容量以及系统内存大小之间进行权衡。

### GPU：模型训练的核心

对于大多数深度学习 (deep learning)工作负载而言，图形处理器（GPU）是最重要的组成部分。然而，并非所有GPU都相同。云服务商提供一系列选项，每种都针对不同类型的工作而设计。您的选择决定了浮点计算的原始速度，这是训练神经网络 (neural network)的基础。

#### GPU等级及其用途

云GPU产品可分为几个等级：

- **高性能训练GPU（例如：NVIDIA H100, A100）：** 这些是专门为大规模训练而设计的顶级加速器。它们拥有大量高带宽内存（HBM2e或HBM3）、用于多GPU通信的先进互连（NVLink），以及张量核心等专用硬件，可大幅加速混合精度计算。将它们用于从头训练大型语言模型（LLM）、扩散模型或复杂计算机视觉模型。它们的高成本可以通过其将训练时间从数周缩短到数天甚至数小时的能力来证明是合理的。
- **通用型与推理 (inference)GPU（例如：NVIDIA A10G, L4, T4）：** 此等级代表了性能与成本之间的平衡。虽然它们可用于训练较小模型或对现有模型进行微调 (fine-tuning)，但其主要优势在于高性价比的推理。它们功耗更低，且租用成本远低于高性能型号。如果您的训练任务时间较短或模型并非过大，这些实例会是务实的选择。
- **前代GPU（例如：NVIDIA V100, P100）：** 这些较旧的加速器仍广泛可用，并且非常高性价比。例如，V100仍然是许多常见训练任务的强大GPU。对于预算紧张或工作负载不需要最新技术的团队而言，这些实例能提供可观的价值。

#### GPU内存（VRAM）：决定性因素

一个常见错误是仅关注GPU的计算速度（以TFLOPS衡量）。芯片内存，即VRAM，往往是更直接的限制。VRAM决定了模型和数据批次一次性可加载到GPU上的最大尺寸。

例如，使用Adam等标准优化器训练一个70亿参数 (parameter)的模型需要大量内存。每个参数可能需要4字节用于权重 (weight)，4字节用于梯度，以及8字节用于优化器状态（动量和方差），总计每个参数16字节。


$$
\text{每个参数的内存} = 4 (\text{权重}) + 4 (\text{梯度}) + 8 (\text{优化器}) = 16 \text{ 字节}
$$


$$
\text{总内存} = 7 \times 10^9 \text{ 参数} \times 16 \text{ 字节/参数} \approx 112 \text{ GB}
$$


这项计算甚至没有将激活所需的内存计算在内，而这取决于批次大小。一个拥有80 GB HBM的NVIDIA A100 GPU适合此任务，而一个仅有16 GB VRAM的GPU，如果不使用高级内存节省技术，将完全无法处理。务必在选择实例前预估您的内存需求。

### 平衡系统：vCPU和RAM

如果强大的GPU不断等待数据，则效率低下。实例的虚拟CPU（vCPU）和系统RAM扮演着辅助角色，负责准备数据并将其供给GPU。

- **vCPU：** CPU在训练期间的主要任务是数据加载和预处理流程。这包括从存储读取数据、执行数据增强（例如旋转或裁剪图像）以及批处理张量。如果您的数据流程涉及大量即时转换，您将需要更多vCPU以保持GPU饱和。训练期间GPU利用率持续低于90%通常指向CPU瓶颈。
- **系统内存（RAM）：** 与GPU VRAM不同，系统RAM用于存放数据集（或其大部分）、缓冲数据批次，以及运行操作系统和其他软件。如果您的数据集大于可用RAM，您的实例将不得不重复从较慢的磁盘存储读取数据，从而产生I/O瓶颈，导致GPU饥饿。

云服务商通常提供具有预配置GPU-vCPU-RAM比例的实例。例如，一个计算优化实例可能为每个NVIDIA T4 GPU提供8个vCPU和61 GB RAM，这个比例非常适合多种任务。

### 解读实例名称

云服务商使用的实例名称可能看起来很神秘，但它们遵循一种逻辑，显示了实例的能力。了解这种模式有助于您快速找到合适的选项。

| 云服务商 | 实例名称示例 | 细分说明 |
| --- | --- | --- |
| **AWS** | `p4d.24xlarge` | **p**：加速计算（GPU）。**4**：代。**d**：包含本地NVMe SSD。**24xlarge**：尺寸（96个vCPU）。 |
| **GCP** | `a2-highgpu-1g` | **a2**：加速器优化（A100 GPU）。**highgpu**：GPU与CPU比例。**1g**：GPU数量（1个）。 |
| **Azure** | `Standard_ND96asr_v4` | **ND**：GPU系列（用于AI/HPC）。**96**：vCPU数量。**a**：AMD CPU。**s**：高级存储。**r**：RDMA。**v4**：版本。 |

此表格显示了名称的不同部分如何表示硬件系列、代、尺寸以及本地存储或高速网络等特殊功能。

> 选择云虚拟机的一个决策过程。GPU的选择受模型大小影响，而所需的CPU和RAM由数据管道的复杂性决定。

### 网络和存储考量

最后，考虑实例的网络和存储能力，特别是对于较大的任务。

- **网络：** 如果您计划跨多个虚拟机（节点）运行分布式训练，它们之间的网络带宽非常重要。关注那些提供高速、低延迟网络的实例系列，例如AWS的弹性构造适配器（EFA）或Azure的RDMA兼容实例。
- **本地存储：** 一些实例配备了本地高速NVMe SSD。如果您的训练需要极快速地访问首先从对象存储下载的部分数据，这会很有益。使用本地存储可以比直接从网络附加磁盘读取更快，进一步防止I/O瓶颈。

最终，选择训练实例是系统设计中的一项实践。您的目标是配置一个平衡的系统，使GPU、CPU、RAM和I/O协调运作。过强的GPU与过弱的CPU搭配会导致预算的低效使用。通过首先分析您的工作负载需求，您可以选择一个能以其价格提供最佳性能的实例，使您离优化的AI基础设施更近一步。

## 参考资料

- [Amazon EC2 Accelerated Computing Instances](https://aws.amazon.com/ec2/instance-types/accelerated-computing/) — Amazon Web Services (2024)
  Publisher: Amazon Web Services
  详细介绍了AWS GPU实例产品系列的官方文档，包括各类GPU层级、规格、EFA等网络选项以及实例命名规范。
- [NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) — NVIDIA Corporation (2022)
  Publisher: NVIDIA Corporation
  解释了高性能NVIDIA GPU架构的技术白皮书，涵盖了HBM3、NVLink和Tensor Cores等特性，这些对大规模AI训练至关重要。
- [GPU platforms for Compute Engine](https://cloud.google.com/compute/docs/gpus/gpu-platforms) — Google Cloud (2024)
  Publisher: Google Cloud
  介绍了Google Cloud Compute Engine GPU产品系列的官方文档，包括支持的GPU型号、配置以及它们与a2-highgpu等实例类型的关系。
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本提供构建ML产品系统级考虑因素的书籍，其中有相关章节讨论了选择硬件、管理计算资源和优化深度学习工作负载的内存使用。
