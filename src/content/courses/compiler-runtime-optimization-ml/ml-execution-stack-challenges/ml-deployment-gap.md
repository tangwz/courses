---
course: "compiler-runtime-optimization-ml"
chapter: "ml-execution-stack-challenges"
lesson: "ml-deployment-gap"
sourceId: 4431
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-1-ml-execution-stack-challenges/ml-deployment-gap"
title: "机器学习模型部署差距"
description: "分析机器学习模型训练与推理部署场景之间的性能差异。"
order: 1
plots: ["plots/4431-0.json"]
sourceHash: "8c4cb1c1549b896e3cfcf8e3acffdf2d3f44858f02495734d2d17e13f8e91b1c"
sourceCorrections: []
---

成功训练一个机器学习 (machine learning)模型是一项成就，但这仅是其中一部分。模型从研究或开发环境中经过验证的状态，到生产环境中高效、可靠服务的转变，通常会呈现出明显的性能差异。这种差异通常被称为**机器学习模型部署差距**。它表明了模型在其开发周期中（通常以强大的硬件上使用Python驱动框架为特点）的行为表现，与在实际应用中部署进行推理 (inference)时的实际运行特性之间的性能差距。理解这一差距的成因，是理解高级编译器和运行时优化必要性的基础。

### 在特定情境中定义“性能”

当我们讨论部署中的性能时，我们通常关注多项指标，与训练阶段相比，这些指标的优先级通常有所不同：

1. **延迟：** 处理单个推理 (inference)请求所需的时间（例如，分类一张图片，翻译一个句子）。这通常是面向用户应用中最重要的指标。
2. **吞吐量 (throughput)：** 单位时间内处理的推理请求数量（例如，每秒推理次数）。这对处理高请求量的后端系统来说很重要。
3. **功耗：** 执行推理所需的能量，这对在严格功耗预算下运行的移动和边缘设备来说尤为重要。
4. **内存占用：** 模型权重 (weight)和中间激活所需的RAM和存储量。这是资源受限硬件上的主要限制。

在训练期间，主要目标通常是最大化吞吐量（通常以每秒处理的样本数衡量），以最小化总训练时间，有时会牺牲单个样本的延迟或内存效率。而在部署中，特别是对于交互式服务，小批量推理的延迟最小化通常成为主要要求，同时还需考虑功耗和内存限制。这种优化目标的变化是导致部署差距的主要原因之一。

### 差距存在的原因：促成因素

多种因素共同导致了这种性能差异：

1. **硬件多样性：** 训练通常在大型集群上进行，这些集群配备高端GPU（如NVIDIA A100或H100），为大规模并行浮点计算做了优化。然而，部署目标却非常多样。它们包括强大的云端CPU和GPU，到专用AI加速器（TPU、NPU）、FPGA、嵌入 (embedding)式GPU，甚至资源受限的微控制器。每个硬件平台都拥有独特的架构特点：不同的指令集（标量、SIMD、矩阵单元）、内存层级（缓存大小、带宽）、支持的数据类型（FP32、FP16、INT8）和功耗范围。在一种硬件上开发出来的模型，如果没有针对目标平台的调整，很少能在另一种硬件上表现最佳。
2. **软件环境不匹配：** 模型通常使用TensorFlow或PyTorch等高级Python框架开发。这些环境优先考虑开发者生产力，提供灵活的API和动态执行（即时模式）。虽然这非常适合实验，但这种动态性引入了开销：Python解释器锁、对象创建/销毁以及每个操作的框架分派逻辑。部署的模型通常在精简的、通常基于C++的运行时环境中运行。这些运行时通过直接执行预编译的计算图或优化过的核函数来最小化开销，绕过了大部分Python基础设施。虽然这减少了开销，但实现潜在性能提升需要仔细的编译和优化。
3. **批量动态：** 训练算法通常依赖大批量大小（例如 $N=256, 512$ 或更多）以获得稳定的梯度并有效利用并行硬件。框架和底层库（如cuDNN或oneDNN）执行的优化通常针对这些大批量情况进行调整，以最大化算术强度并隐藏内存延迟。相反，推理 (inference)工作负载经常涉及小批量大小，对于实时应用通常是 $N=1$。在低批量情况下的性能通常受限于内存带宽、核函数启动开销或并行计算单元的低效使用，而非原始计算能力。对大批量有效的优化，对小批量可能无效甚至有害。

   
   
   ![延迟与批量大小](plots/4431-0.json)
   
   

   > 比较显示，在优化运行时中，每样本延迟通常会随批量大小的增加而更大幅度地减少，相比即时框架，尤其是在框架开销占据主导地位的小批量大小情况下。
4. **数值精度变化：** 为达到性能和效率目标，部署的模型通常会从训练期间使用的标准32位浮点数（FP32）转换为低精度格式，如16位浮点数（FP16或Bfloat16）、8位整数（INT8），甚至更激进的方案。虽然低精度能显著加速计算并减少内存使用（可能达到 $2\times$ 到 $4\times$ 甚至更多），但它需要仔细处理。这涉及量化 (quantization)（将FP32值映射到低精度域）、可能需要重新训练（量化感知训练），以及生成利用专门低精度硬件指令的代码。简单的转换可能导致准确度大幅下降，而最佳实现则需要复杂的编译器和运行时支持。
5. **图级别与操作级别执行：** 框架通常在即时模式下逐操作执行模型。然而，编译器可以查看整个计算图。这种全局视图使得在单个操作级别无法实现的优化成为可能，例如算子融合（将多个操作合并到一个核函数中以减少内存流量和开销）、跨操作边界的代数简化，以及优化的内存布局转换（例如，根据硬件偏好在NCHW和NHWC格式之间转换）。这些图级别优化是编译执行路径中重要的性能改进来源。

解决机器学习 (machine learning)模型部署差距正是专用机器学习编译器和运行时出现的原因。它们充当桥梁，将灵活框架中开发的高级模型描述，通过多层抽象和优化，转换为高效、针对特定硬件的机器代码。本课程后续章节将分析这些系统中采用的高级技术，以系统地分析模型、优化计算图和张量操作、为多种硬件生成代码，并高效管理执行，最终弥合开发与部署性能之间的差距。

## 参考资料

- [TVM: An End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3305417.3305467](https://doi.org/10.5555/3305417.3305467)
  这篇开创性论文介绍了TVM，一个深度学习编译器栈，旨在解决在多样化硬件上部署机器学习模型的挑战，通过图优化和硬件专用代码生成直接弥合部署差距。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://ieeexplore.ieee.org/document/8578665) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Andrey Ogarev, Mark Sandler, Hartwig Adam, and Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; DOI: [10.1109/CVPR.2018.00869](https://doi.org/10.1109/CVPR.2018.00869)
  这篇来自谷歌的广泛引用论文提出了一种将神经网络量化为8位整数的方法，以在各种硬件上实现高效推理，直接处理了部署差距中提到的数值精度因素。
- [Deep Learning Compilers: A New Horizon for Optimizing AI Workloads](https://dl.acm.org/doi/10.1145/3397943) — Xiaoke Yu, Jiong Luo, Cheng Li, Yunji Chen, and Tianshi Chen (2020)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery (ACM); Volume: 53; Pages: 1-36; DOI: [10.1145/3397943](https://doi.org/10.1145/3397943)
  这篇综述提供了深度学习编译器设计与实现的全面概览，详细说明了用于弥合开发与部署之间性能差距的各种优化技术，适用于多样化硬件。
