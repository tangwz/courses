---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-ml-runtime-systems"
lesson: "runtime-heterogeneous-scheduling"
sourceId: 4551
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems/runtime-heterogeneous-scheduling"
title: "异构系统调度"
description: "设计能将任务最佳分配至CPU、GPU和其他加速器的调度器。"
order: 5
plots: []
sourceHash: "79b9103c84bcf3bab3ff6664686679c817f8a3db544d0b45286be912cd176d36"
sourceCorrections: []
---

现代机器学习 (machine learning)工作负载经常在由不同处理单元组成的系统上运行，这些单元包括多核CPU、高性能GPU和专用AI加速器（如TPU、NPU或定制ASIC）。有效利用这些资源的综合性能，需要精密的运行时调度策略。目标是协调计算任务在这些不同设备上的执行，以最小化总执行时间，同时最大化硬件利用率。这是一个复杂的优化问题，需要仔细考虑计算成本、数据局部性、设备间通信开销、任务依赖性以及每个硬件组件的特定能力。

### 计算资源特性描述

智能调度的一个先决条件是对目标硬件能力有清晰的认识。运行时调度器通常依赖于一个性能模型，该模型可以是静态推导的，也可以通过运行时分析改进，它根据以下指标描述每个设备：

- **计算吞吐量 (throughput)：** 以FLOPS（每秒浮点运算次数）或TOPS（每秒万亿次运算）衡量，通常区分不同精度（例如FP32、FP16、INT8）。
- **内存带宽：** 数据从设备本地内存（例如HBM、GDDR6、DDR4）读写速率。
- **内存容量：** 设备上可用内存总量。
- **专用单元：** 特定操作硬件加速器（例如NVIDIA Tensor Core、AMD Matrix Core）的存在及其性能。
- **通信接口：** 设备与主机内存之间连接的带宽和延迟特性（例如PCIe代、NVLink、Infinity Fabric）。

准确的特性描述使调度器能够对何时何地执行特定计算内核做出明智决策。

### 任务表示和依赖

机器学习 (machine learning)计算通常表示为有向无环图（DAG），其中节点代表操作（内核），边代表数据依赖（张量）。调度器在该图上运行，决定任务的放置和执行顺序。

> 一个简化的ML推理 (inference)图，显示了任务的潜在设备放置，并突出了主机（CPU）与设备（GPU）内存之间的数据传输。

DAG中任务的粒度对调度有显著影响。细粒度任务（例如，单个算术操作）提供最大灵活性，但会产生高调度和同步开销。粗粒度任务（例如，融合操作符序列）减少开销，但限制了跨设备并行执行和负载平衡的机会。许多运行时在中间粒度上操作，通常对应于单个ML框架操作或优化内核。

### 调度目标和权衡

尽管最小化端到端延迟通常是主要目标，但其他目标也会影响调度决策：

- **吞吐量 (throughput)：** 最大化单位时间内完成的推理 (inference)或训练步数。
- **资源利用率：** 让昂贵的加速器保持忙碌以证明其成本。
- **能源效率：** 最小化功耗，这在移动或边缘部署中尤为重要。
- **服务质量（QoS）：** 在多租户环境中满足特定延迟期限或公平性标准。

这些目标可能相互冲突。例如，最大化吞吐量可能涉及批处理请求，这可能会增加单个请求的延迟。节能策略可能涉及关闭设备或以较低频率运行它们，影响峰值性能。调度器通常必须根据系统策略或应用需求来平衡这些相互竞争的需求。

### 核心调度策略

运行时调度器采用多种策略将任务分配给设备并确定它们的执行顺序：

1. **静态调度：** 调度决策在运行时之前（AOT）做出，通常在编译阶段。编译器分析DAG，根据设备配置文件估计任务执行时间和通信成本，并生成固定的执行计划。经常使用异构系统最早完成时间（HEFT）或关键路径分析等算法，这些算法已适应异构成本。

   - *优点：* 最少的运行时开销，可预测的性能（假设成本模型准确）。
   - *缺点：* 对运行时变化（例如，动态数据大小，来自其他进程的竞争）不灵活，高度依赖静态成本模型的准确性。
2. **动态调度：** 调度决策在运行时做出，就在任务准备执行之前。当任务完成时，调度器检查就绪任务（那些依赖已满足的任务），并根据当前系统状态和调度策略将它们分配给可用设备。

   - *优点：* 适应实时条件、动态输入和实际任务持续时间；可以实现更好的负载平衡。
   - *缺点：* 引入调度决策的运行时开销；需要快速启发式算法；由于局部决策可能导致非最优的全局调度。常见技术包括优先级队列（优先处理关键路径上的任务或需要特定加速器的任务）和工作窃取队列（空闲处理器“窃取”繁忙处理器的任务，适用于平衡类似设备之间的负载）。
3. **混合调度：** 这种方法结合了静态规划和动态调整。大型计算块或关键路径可能被静态调度，而较小的任务或基于运行时条件（例如，队列长度，实际执行时间）的调整则动态处理。这旨在平衡低开销和适应性。

### 数据局部性的核心作用

异构调度中，管理数据移动也许是最重要的因素。在主机CPU主内存和加速器本地内存之间传输张量（例如，通过PCIe）通常比计算或设备内内存访问慢几个数量级。有效的调度器优先考虑最小化或隐藏这种通信延迟：

- **任务协同放置：** 将消耗/生成相同张量的连续操作调度到同一设备上，以避免不必要的数据传输。
- **通信隐藏：** 使用异步内存复制操作和专用硬件复制引擎（例如，使用CUDA流或HIP流）将数据传输与其他设备上的计算重叠。调度器明确管理计算内核和复制操作之间的依赖关系。
- **内存管理感知：** 考虑设备内存容量限制。如果工作集超出设备内存，调度器可能需要协调张量的逐出和预取，并与运行时的内存管理器密切协作。使用固定主机内存进行更快的DMA传输或统一虚拟内存（UVM）等技术可以简化编程，但需要调度器进行仔细的性能调整。

### 设备亲和性与放置启发式

决定哪种类型的设备（CPU、GPU、其他加速器）最适合给定任务涉及基于以下因素的启发式方法：

- **操作类型：** 大型、可并行化的操作，如矩阵乘法或卷积，通常分配给GPU或加速器。控制流、不规则内存访问模式或加速器不支持的操作可能在CPU上运行。
- **数据大小：** 非常小的操作在加速器上启动和传输数据可能比直接在CPU上执行产生更多开销。
- **硬件专业性：** 利用特定数据类型（例如INT8、FP8）或操作（例如矩阵乘法）的任务应调度到具有专用硬件单元（如Tensor Core）的设备上以获得最大性能。
- **当前负载：** 动态调度器考虑每个设备当前的利用率和任务队列来平衡负载。

### 同步机制

在多个设备上执行DAG需要同步。当设备B上的任务依赖设备A上任务的输出时，调度器必须确保任务A在任务B开始前完成。这通常通过硬件/驱动程序API提供的轻量级同步原语（例如CUDA事件、HIP事件、围栏）来管理。调度器插入并等待这些事件。跨设备同步会引入延迟，因此最小化同步点是另一个优化目标，通常通过仔细的任务分组和调度实现。

### 高级调度考量

在这些核心原理之上，高级运行时引入了更多复杂性：

- **多设备调度：** 扩展到包含多个GPU或加速器的系统会引入拓扑感知。调度器在分配通信任务时必须考虑设备间链接（例如NVLink与PCIe）的带宽和延迟。
- **功耗感知调度：** 将功耗纳入成本模型，可以实现能源效率调度，可能通过为非关键任务选择功耗较低的设备或调整设备频率（DVFS）来实现。
- **干扰与操作系统交互：** 运行时调度器必须感知来自其他应用或系统进程的潜在干扰，并可能需要与操作系统级别的调度决策交互或适应其变化。
- **预测建模：** 一些运行时采用基于执行轨迹训练的机器学习 (machine learning)模型，以比静态启发式方法更准确地预测任务持续时间和通信成本，从而做出更好的动态调度决策。

为异构系统设计一个有效的调度器仍然是研究和工程领域的一个活跃方向。这需要对ML工作负载特性和底层硬件能力有深刻的理解，仔细平衡众多权衡，为复杂的AI应用提供最佳性能和效率。

## 参考资料

- [TVM: An End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3342371.3342416](https://doi.org/10.5555/3342371.3342416)
  本文介绍了TVM，一个深度学习编译器栈，它优化并部署模型到各种硬件后端，专注于异构执行和硬件特定优化。
- [A Survey on Deep Learning System Optimization: From Algorithm to Hardware](https://ieeexplore.ieee.org/document/9079943) — Yujun Chen, Shuchang Sheng, Xiaolong Ma, Peiyan Dong, Zhiye Tang, Junru Zhao, Yuanjie Xie, and Shaoli Liu (2020)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 108; Pages: 781-802; DOI: [10.1109/JPROC.2020.2991739](https://doi.org/10.1109/JPROC.2020.2991739)
  本调查全面概述了深度学习系统中的各种优化技术，包括对异构调度、通信优化和内存管理。
- [GSPMD: General and Efficient Parallelism for ML Workloads](https://arxiv.org/abs/2105.04663) — Yuanzhong Xu, Jiri Simsa, Jeremy Smith, D. J. Bernstein, Yuan Zhang, Sarah Sirajuddin, and Anna Goldie (2021)
  Journal: Proceedings of Machine Learning and Systems (MLSys '21); Publisher: MLSys; Volume: 3; Pages: 198-212; DOI: [10.48550/arXiv.2105.04694](https://doi.org/10.48550/arXiv.2105.04694)
  本文介绍了GSPMD，一个用于在异构设备上自动高效地并行化机器学习工作负载的系统，其包含了用于数据移动和设备放置的策略。
- [Ansor: Generating High-Performance Tensors with N-Dimensional Search Space Auto-Tuning](https://www.usenix.org/conference/osdi20/presentation/zheng) — Lianmin Zheng, Chengfan Jia, Minmin Sun, Zhao Wu, Cody Hao Yu, Ameer Haj-Ali, Yida Wang, Jun Yang, Danyang Zhuo, Koushik Sen, Joseph E. Gonzalez, Ion Stoica (2020)
  Journal: 14th USENIX Symposium on Operating Systems Design and Implementation (OSDI 20); Publisher: USENIX Association; Pages: 863-879; DOI: [10.5555/3474345.3474397](https://doi.org/10.5555/3474345.3474397)
  本文详述了一个深度学习编译器的自动调优框架，它依赖于准确的性能模型和硬件表征，以动态优化异构硬件上的张量程序。
