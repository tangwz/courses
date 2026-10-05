# 算力选择：CPU、GPU 和 TPU 架构

来源：[原文](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-1-architectural-patterns-ai-platforms/compute-selection-cpu-gpu-tpu)

[返回章节目录](README.md) · [返回课程目录](../README.md)

选择合适的计算硬件是一个重要的优化问题，它不只局限于原始性能。CPU、GPU 和 TPU 之间的架构差异为不同的机器学习 (machine learning)任务带来了独特的性能表现和成本效益。不正确的选择可能导致严重的资源未充分利用、训练瓶颈以及高昂的云账单。分析各种架构的权衡有助于为您的工作负载选择最合适的处理器。

### 中央处理器 (CPU)：通用计算基础

CPU 设计用于通用性，并能低延迟执行多种任务。其架构包含少量强大复杂的内核，每个内核都能独立执行指令并处理复杂分支逻辑。现代服务器级 CPU 采用单指令多数据 (SIMD) 向量 (vector)指令，例如 Intel 的 AVX（高级向量扩展），以同时对多个数据点执行相同操作。

对于机器学习 (machine learning)，CPU 在以下几个重要方面仍然不可或缺：

- **数据预处理与增强：** 读取存储、解析数据格式（例如 JSON、CSV）和应用转换等任务通常涉及复杂串行逻辑，不易并行化。CPU 的高单线程性能非常适合这些数据管道阶段。
- **传统机器学习：** 不依赖大规模矩阵运算的算法，例如梯度提升树（XGBoost, LightGBM）或支持向量机，可在 CPU 上高效运行。
- **小型模型的低延迟推理 (inference)：** 当以尽可能低的延迟提供单个推理请求是主要目标时，CPU 有时可能比 GPU 表现更好。GPU 启动内核的开销会抵消其对小批次大小（$批大小 = 1$）的并行处理优势，特别是当模型本身不是计算密集型时。

然而，对于定义深度学习 (deep learning)的大规模矩阵和张量运算，CPU 的架构成为了限制因素。与专用加速器相比，有限的核心数量和较低的内存带宽会形成主要瓶颈，使得它们不适合训练大型神经网络 (neural network)。

### 图形处理器 (GPU)：并行处理引擎

GPU 从图形渲染硬件演变而来，成为深度学习 (deep learning)训练的事实标准。其架构与 CPU 根本不同。GPU 包含数千个更小、更简单的核心，这些核心优化用于在大量数据阵列上并行执行相同指令。这种设计与神经网络 (neural network)核心的张量代数是理想匹配。

现代数据中心 GPU 的两个架构特点对于 AI 工作负载尤其重要：

1. **张量核心 (Tensor Cores)：** 这些是集成到 NVIDIA GPU 架构（自 Volta 架构以来）中的专用硬件单元。张量核心旨在加速矩阵乘累加 (MMA) 运算，这是深度学习中的主要计算。它们在混合精度计算方面非常高效。例如，张量核心可以在一次操作中完成两个 16 位浮点矩阵（$FP16$ 或 $BF16$）的乘法，并将结果加到一个 32 位浮点（$FP32$）矩阵中。这显著提升了吞吐量 (throughput)（以 FLOPs 衡量），并允许训练更大的模型或缩短训练时间。
2. **高带宽内存 (HBM)：** 为防止数千个核心出现数据匮乏，数据中心 GPU 配备了 HBM。这种堆叠式 DRAM 技术提供的内存带宽比 CPU 使用的 DDR 内存高出一个数量级。例如，NVIDIA A100 GPU 提供超过 1.5 TB/s 的内存带宽，而高端 CPU 的内存系统可能提供约 200 GB/s。这种高带宽对于训练的前向和后向传播期间向计算单元传输数据非常重要。

大规模并行、专用张量核心和高带宽内存的结合，使得 GPU 成为深度学习模型训练和高吞吐量推理 (inference)最灵活且广泛使用的加速器。

### 张量处理器 (TPU)：专用神经网络 (neural network)加速器

张量处理器 (TPU) 是 Google 定制设计的专用集成电路 (ASIC)，为单一目的而构建：大规模加速神经网络计算。GPU 是一个可编程并行处理器，而 TPU 是一个专用矩阵处理器。

TPU 的核心是其**脉动阵列**。脉动阵列是由简单数据处理单元组成的网络，在 Google 的架构中称为矩阵乘法单元 (MXU)。数据被加载到处理器中，然后以有节奏的波浪式在阵列中“泵送”。每个处理单元执行一次计算，并将其结果传递给相邻单元。这种设计最大化了处理器内部内存的数据复用，显著减少了访问主内存 (DRAM) 的需求，并最大限度地降低了特定计算的功耗。

TPU 的主要优势源于这种专用设计：

- **训练的超强扩展性：** TPU 通过定制高速的二维环面互连，被设计成可互连成称为“Pod”的大型多节点配置。这使得在数千个 TPU 芯片上训练巨型模型时，可以实现接近线性的性能扩展。
- **矩阵运算的峰值性能：** 对于大型矩阵乘法，脉动阵列架构非常高效，在大型语言和视觉模型等特定工作负载中，通常比 GPU 提供卓越的每瓦性能。
- **高吞吐量 (throughput)推理 (inference)：** 相同的架构优势使得 TPU 成为大批次推理场景的有力选择，其目标是每秒最大化推理数量。

TPU 的专业化也带来了局限。它们的灵活性不如 GPU，最适合表示为大型密集矩阵乘法的工作负载。涉及自定义 CUDA 内核、大量分支或稀疏数据访问模式的运算，可能在 GPU 上表现更好。此外，它们的可用性主要限于 Google Cloud Platform。

### 做出正确选择

在 CPU、GPU 和 TPU 之间做出选择，需要根据每个处理器的架构优势来分析您的特定工作负载。决策过程通常平衡性能、成本和开发灵活性。

> 基于工作负载特点选择计算硬件的决策流程。最佳选择取决于任务是训练、推理 (inference)还是数据处理，以及对规模和性能的具体要求。

最终，最佳的基础设施设计通常会组合使用这些处理器。一个典型的 MLOps 管道可能会在数据摄取和预处理步骤中使用 CPU，使用大型 GPU 或 TPU 集群进行分布式训练，并使用一组 GPU 甚至 CPU 来服务生成模型，每个组件的选择旨在最大化其特定任务的性能和成本效益。

## 参考资料

- [In-Datacenter Performance Analysis of a Tensor Processing Unit](https://dl.acm.org/doi/10.1145/3079893.3080246) — Norman P. Jouppi, Cliff Young, David Patil, Dustin Patterson, Gaurav Agrawal, Haixin Liao, Kevin Schardl, Mike Smith, Dave Washington, Zhifeng Chen, Yuan Cao, Branden Holt, Luke Taylor, Alan Knaggs, Rohan Kumar, Sanjay Nigam, Guang Gao, Al Gleason, Robert Horton, Daniel Kini, Danny Kwon, David Leyda, Francois Obermeyer, Rajkumar Rathod, Kevin Ren, Aditya Taraporevala, Grant Vert, Xinan Long, Niels Dekker, Andy Hölzle, Quoc Le, Chengyu Sun (2017)
  Journal: Proceedings of the 44th Annual International Symposium on Computer Architecture (ISCA '17); Publisher: ACM; Pages: 1-12; DOI: [10.1145/3079893.3080246](https://doi.org/10.1145/3079893.3080246)
  描述了谷歌第一代张量处理单元（TPU）的架构和性能特征。
- [NVIDIA Volta GV100 GPU Architecture](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHHJvQ9mmakcb5I0ZseQkklO5mAqRILNRCyZujX62AttfzkFAEmfwgXkNMysdrOqn7W8peHwsyegCPv8iejn31jKJq-ssXYfl1zv9Vh_7sg0M71CNzcdXN8CPw-2UluQ4z5JsDX2Cz56niWxpXL4WPodByD2YhriuXLl3FlJHrIAul3b62k4rCORpd-aaNgERs82ksmMg==) — David B. Kirk, John W. Nickolls (2017)
  Publisher: NVIDIA Corporation
  提供Volta GPU的规格和架构细节，包括Tensor Cores的引入。

---

[上一节](01-%E8%A7%84%E6%A8%A1%E5%8C%96MLOps%E5%8E%9F%E5%88%99.md) · [下一节](03-%E5%88%86%E5%B8%83%E5%BC%8F%E7%B3%BB%E7%BB%9F%E7%9A%84%E9%AB%98%E5%B8%A6%E5%AE%BD%E4%BA%92%E8%BF%9E.md)
