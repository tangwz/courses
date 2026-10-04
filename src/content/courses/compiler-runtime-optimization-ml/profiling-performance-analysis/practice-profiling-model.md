---
course: "compiler-runtime-optimization-ml"
chapter: "profiling-performance-analysis"
lesson: "practice-profiling-model"
sourceId: 4593
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/practice-profiling-model"
title: "动手实践：分析优化模型的性能"
description: "练习使用GPU/CPU性能分析工具，分析经ML编译器优化后的简单模型的性能表现。"
order: 8
plots: ["plots/4593-0.json", "plots/4593-1.json"]
sourceHash: "49b1979c697aa49752515841ecdf68341aee125b842c620b4390fe9358c2d857"
sourceCorrections: []
---

理论很重要，但性能优化本质上是一门经验科学。对一个简单的、经过优化的机器学习 (machine learning)模型组件进行性能分析，以展示优化实践。我们将模拟一个场景：一个在许多机器学习模型中很重要的矩阵乘法（GEMM）运算，被ML编译器编译并针对NVIDIA GPU进行了优化。我们的目的是使用NVIDIA Nsight Compute来分析其性能表现。

### 场景设置

假设我们有一个已编译的可执行文件`gemm_optimized`，它执行矩阵乘法$C = A \times B$，其中$A$、$B$和$C$是大型矩阵。ML编译器已应用了诸如分块、共享内存利用和指令调度等优化，以生成高效的CUDA内核。

我们的目标硬件是NVIDIA GPU（例如，Ampere或Hopper架构的GPU）。我们将主要使用NVIDIA Nsight Compute (`ncu`)，这是一个详细的内核性能分析工具。

### 步骤1：运行性能分析工具

Nsight Compute可以通过命令行或其图形用户界面（GUI）使用。对于自动化分析或脚本编写，命令行通常更受青睐。为了捕获详细的性能数据，我们可以在`ncu`下执行已编译的程序。

```bash
# 确保CUDA工具包的二进制文件在您的PATH中
# 示例：对可执行文件'gemm_optimized'进行性能分析
# --set full: 收集一套全面的指标（可能耗时）
# -o profile_report: 将报告保存到名为'profile_report.ncu-rep'的文件
# ./gemm_optimized: 要进行性能分析的可执行文件

ncu --set full -o profile_report ./gemm_optimized
```

此命令运行`gemm_optimized`并收集应用程序启动的每个CUDA内核的详细性能数据，将其保存到`profile_report.ncu-rep`。为了更快地分析特定方面，您可以使用预定义的指标集（例如，`--set roofline`，`--set memory`）或指定单个指标。

### 步骤2：分析性能报告

您可以使用Nsight Compute GUI (`nv-nsight-cu`) 打开`profile_report.ncu-rep`文件，或通过命令行直接分析其内容（`ncu --query-metrics ... profile_report.ncu-rep`）。让我们关注通常在GUI或详细CLI报告中查看的主要部分。

#### 识别内核

报告会列出所有启动的CUDA内核。识别负责矩阵乘法的主要内核。它的名称可能具有提示性，如`gemm_kernel`、`matmul_core`，或是由编译器内部表示派生出的复杂名称。将您的分析集中在此内核上，特别是如果它占用了大部分GPU执行时间。

#### 性能部分

1. **GPU 光速 (SOL) 吞吐量 (throughput)**:

   - 此部分提供了一个宏观视角，比较了内核实际达到的计算吞吐量（FLOPS、Tensor Core操作）和内存带宽（DRAM、L2、L1）与硬件的理论峰值能力。
   - *解释*：一个接近计算峰值的内核是计算受限的。一个接近内存带宽峰值的内核是内存受限的。通常，内核性能介于两者之间，这可能表明在指令延迟、控制流或缓存利用方面存在潜在的瓶颈。

   
   
   ![简化版光速图示例](plots/4593-0.json)
   
   

   > 这个简化版屋脊图展示了两个内核。内核A（红色）运行在内存和计算上限以下，表明存在效率低下。内核B（蓝色）更接近计算上限，表明它可能是计算受限的。
2. **占用率**:

   - 衡量在流式多处理器（SM）上同时有多少个warp（32个线程的组）处于活跃状态，相对于最大可能值。低占用率意味着SM未充分利用，可能无法充分隐藏延迟。
   - Nsight Compute会显示`已达到的占用率`，并识别限制因素：`每SM的块数`、`每线程的寄存器数`、`每块的共享内存`。
   - *解释*：由寄存器导致的低占用率表明存在寄存器压力（编译器可能将寄存器溢出到本地内存，这会很慢）。由共享内存导致的低占用率表示内核配置请求的每块共享内存超过可用量，从而限制了并发块。由块限制导致的低占用率可能意味着网格大小太小，或线程块大小对于该问题来说过大。
3. **指令统计**:

   - 提供关于warp执行效率的信息。查看`指令发射槽利用率`（使用了多少指令发射槽）和`每时钟周期执行指令数`（IPC）等指标。
   - 按指令类型（整数、浮点、内存、控制流）的分类可以显示不平衡情况。
   - *解释*：低指令发射利用率或低IPC可能指向指令依赖、长延迟操作（如`sqrt`或超越函数），或编译器调度未能充分发挥并行性。高控制流分歧（warp中不同线程选择不同路径）会显著降低性能。
4. **内存工作负载分析**:

   - 对于理解内存受限的内核很重要。检查L1/TEX缓存命中率、L2缓存命中率和DRAM带宽。
   - 诸如`内存吞吐量`等指标显示了与峰值相比的实际带宽。查看内存操作（全局、本地、共享）的分类情况。
   - *解释*：低缓存命中率加上高DRAM带宽使用量强烈表明内存受限。这可能源于不良的数据局部性（非合并内存访问）或编译器选择的低效分块策略。高共享内存流量理想情况下应对应高缓存命中率（如果用作L1缓存）或高效重用。高本地内存流量（溢出）通常是不利的。

   
   
   ![内核内存带宽利用率](plots/4593-1.json)
   
   

   > 比较了不同内存级别实际达到的带宽与理论峰值带宽。DRAM级别的高利用率表明内核可能是内存受限的。
5. **源代码/汇编关联（可选但有效）**:

   - 如果在编译和性能分析期间设置了源代码关联，Nsight Compute可以将性能计数器映射回CUDA C++或PTX/SASS汇编代码的特定行。
   - *解释*：这对于定位导致停顿的具体指令（例如，高延迟指令，缓存性能差的内存加载/存储）非常有价值。它有助于确认编译器优化（如循环展开或指令调度）是否有效，或者是否有特定生成的指令成为瓶颈。

### 步骤3：迭代与假设验证

性能分析很少是一次性完成的过程。基于分析结果：

- **如果计算受限且SOL较低**：检查指令统计信息是否存在延迟或调度问题。查看汇编代码是否存在低效指令序列。也许编译器需要提示或不同的调度转换（例如，通过可用的话语或编译器标志）。
- **如果内存受限**：分析内存工作负载详情。缓存命中率是否低？DRAM带宽是否饱和？这可能表明需要调整分块参数 (parameter)、数据布局（如果可能在预编译阶段），或在编译器中尝试不同的循环转换。
- **如果受限于占用率**：如果受寄存器限制，能否简化内核，或引导编译器使用更少的寄存器？如果受共享内存限制，能否调整算法，或减少每个块的共享内存用量？

本次实践展示了性能分析工具如何连接高层ML模型与低层硬件执行。通过系统地分析Nsight Compute等工具提供的性能指标，您可以诊断编译器和运行时引入或未解决的瓶颈，从而指导进一步的优化工作，以实现ML工作负载的最佳性能。请记住，查阅您选择的性能分析工具和硬件的特定文档，以获取详细的指标定义和高级功能。

## 参考资料

- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  NVIDIA Nsight Compute的官方文档，提供了使用分析器、解释其指标和理解GPU性能分析的详细信息。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  涵盖了CUDA编程模型、GPU架构和性能优化技术，这些对于解释NVIDIA Nsight Compute等工具的分析结果至关重要。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — David B. Kirk and Wen-mei W. Hwu (2016)
  Publisher: Morgan Kaufmann
  一本关于GPU架构和使用CUDA进行并行编程的综合教材，其中性能分析和优化部分为理解分析指标提供了背景。
- [The Roofline Model: An Analytical Model for Floating-Point Performance on Multicore Architectures](https://doi.org/10.1145/1654059.1654083) — Samuel Williams, Andrew Waterman, and David A. Patterson (2009)
  Journal: SC '09: Proceedings of the ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis; Publisher: ACM; Pages: 1-11; DOI: [10.1145/1654059.1654083](https://doi.org/10.1145/1654059.1654083)
  介绍了用于性能分析的基础Roofline模型，与‘GPU光速吞吐量’部分直接相关。
