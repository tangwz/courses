---
course: "compiler-runtime-optimization-ml"
chapter: "profiling-performance-analysis"
lesson: "gpu-kernel-profiling"
sourceId: 4589
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/gpu-kernel-profiling"
title: "GPU核心性能分析 (Nsight Compute, ROCprof)"
description: "GPU核心性能详细分析：占用率、内存带宽、指令停滞。"
order: 4
plots: ["plots/4589-0.json"]
sourceHash: "54a6ee33b17f9725197073cb9b43e5046a229305a158992580b6adac7df68650"
sourceCorrections: []
---

系统级性能分析工具可以对CPU、GPU和互连组件上的时间消耗分布提供有益的概览，但它们通常将GPU核心执行视为不透明的时间块。当机器学习 (machine learning)编译器融合多个操作或应用复杂的循环转换并生成自定义核心时，系统级工具可能只简单地显示一个长时间运行的核心。要弄清该核心运行缓慢或效率低下的原因，您需要使用专门的GPU核心性能分析工具检查其内部。这些工具提供与GPU硬件架构直接关联的细粒度性能数据。

对于NVIDIA GPU，主要工具是**NVIDIA Nsight Compute**。对于使用ROCm软件栈的AMD GPU，对应的工具是**ROCprof**，通常与\*\*Radeon GPU Profiler (RGP)\*\*结合使用以进行可视化。这些性能分析工具使您能够收集针对单个核心启动的详细硬件性能计数器和指标。

### 为什么核心级性能分析是必要的

优化过的机器学习 (machine learning)核心，特别是那些由高级编译器生成的，会充分使用GPU硬件的性能。性能瓶颈可能源于核心执行内部的多种因素：

1. **占用率限制**：GPU的计算能力是否得到充分使用？低占用率，即每个多处理器上活跃的warp（NVIDIA）或wavefront（AMD）数量低于理论可能值，可能限制了隐藏内存延迟和保持执行单元繁忙的能力。这可能是由于每个线程的寄存器使用量高、过多的共享内存（AMD上的LDS）分配或编译器选择的线程块大小所致。
2. **指令瓶颈**：核心是否受到特定类型指令的瓶颈影响？性能分析工具可以分解指令组合，说明核心是否大部分时间花在浮点运算（FP32、FP16、TF32）、整数运算、内存访问指令或控制流（分支）上。针对Tensor Cores（NVIDIA）或Matrix Cores（AMD）等专门单元很重要，性能分析工具可以验证这些单元是否得到有效使用。
3. **内存层级效率低下**：核心是否有效使用GPU缓存（L1、L2）并访问DRAM？高L1/L2缓存未命中率与高DRAM带宽使用率相结合，表明存在内存限制行为。核心性能分析工具提供关于缓存命中率、请求与实际达到的内存吞吐量 (throughput)以及内存操作引起的延迟的详细指标。
4. **执行停滞**：warp/wavefront是否经常停滞，等待资源？性能分析工具可以找出停滞的主要原因，例如等待内存数据（内存延迟）、等待前一条指令的结果（数据依赖）、同步屏障或次优的指令调度。

### NVIDIA Nsight Compute

Nsight Compute侧重于分析单个CUDA核心启动。它从GPU硬件收集广泛的性能计数器，并通过不同的“部分”展示它们，每个部分侧重于性能的特定方面。

**基本原理和指标：**

- **部分**：性能数据的组织视图，例如“GPU光速”（将实际性能与理论最大值进行比较）、“占用率”、“内存负载分析”、“调度器统计”和“Warp状态统计”。
- **源代码关联**：Nsight Compute尝试将性能指标与源代码（CUDA C++、PTX汇编或在有调试信息时甚至更高级别的中间表示）关联起来。这有助于确定代码中的哪些行引起了性能问题。
- **占用率**：报告*理论占用率*（基于硬件限制的最大可能值）和*实际占用率*（核心执行期间每个SM实际平均活跃的warp数量）。实际占用率相对于理论值较低通常表示资源限制（寄存器、共享内存）。
- **内存吞吐量 (throughput)**：提供在内存层级不同级别（L1、L2、DRAM）之间请求和传输数据的详细分解。比较请求与实际达到的带宽有助于找出内存瓶颈。



![内存性能比较：核心A 对比 核心B](plots/4589-0.json)



> 比较两个核心。核心A显示中等的L1命中率但DRAM带宽利用率很高，这表明它可能在DRAM层面受到内存限制。核心B具有高L1命中率和低DRAM使用率，表明更好的缓存局部性和潜在的计算限制。

- **指令统计**：分解执行指令的组合（FP32、FP64、INT、内存、控制流、Tensor Core），并计算诸如每时钟指令数（IPC）之类的指标。低IPC可能表示频繁的停滞。
- **停滞原因**：量化 (quantization)warp停滞的原因（例如，`Wait` 等待内存依赖、`Instruction Fetch` 指令获取、`Execution Dependency` 执行依赖、`Barrier` 屏障）。这对于了解延迟瓶颈非常有帮助。

**工作流程：**

1. **性能分析**：在`ncu`下启动您编译的机器学习 (machine learning)应用（例如，`ncu --set full -o profile_report ./my_ml_app`），或者将图形界面附加到正在运行的进程。
2. **数据收集**：Nsight Compute拦截核心启动，如果需要，会重新运行它们以收集不同组的计数器（由于硬件对同时收集计数器的限制），并收集数据。
3. **分析**：在图形界面中检查生成的报告文件（`.ncu-rep`）或分析命令行输出。从“GPU光速”和“占用率”等高层部分开始，然后使用“内存负载分析”或“调度器统计”等详细部分调查已识别的特定瓶颈。使用源代码关联将问题映射回代码区域。

### AMD ROCprof 和 Radeon GPU Profiler (RGP)

对于运行ROCm软件栈的AMD GPU，`rocprof`是用于收集核心性能计数器的命令行工具。结果通常使用Radeon GPU Profiler (RGP) 进行可视化和分析，尽管`rocprof`也可以输出原始数据（例如CSV）。

**基本原理和指标：**

- **计数器**：ROCprof依靠收集AMD GPU架构上可用的特定硬件性能计数器（例如，用于内存的`FETCH_SIZE`、`WRITE_SIZE`，用于计算单元的`VALU_UTILIZATION`、`SALU_UTILIZATION`，用于缓存性能的`L2CacheHit`）。
- **Wavefronts**：AMD GPU上的基本执行单元，类似于NVIDIA的warp。占用率指标与每个计算单元（CU）的活跃wavefront数量相关。
- **LDS（本地数据共享）**：类似于NVIDIA共享内存的片上内存。高LDS使用率会限制wavefront占用率。
- **VGPRs/SGPRs**：矢量和标量通用寄存器。高使用率会限制占用率。

**核心指标（通常从计数器得出）：**

- **占用率**：与NVIDIA类似，测量每个CU的活跃wavefront与最大可能数量的比率，受VGPRs、SGPRs和LDS等资源限制。
- **计算利用率**：`VALUUtilization` 等指标显示主矢量执行单元的繁忙程度。
- **内存带宽**：计数器跟踪内存层级不同级别（矢量内存、L1、L2、HBM）之间的数据移动。得出的指标包括实际达到的带宽。
- **缓存性能**：`L2CacheHit` 等计数器直接测量缓存效率。
- **指令组合**：计数器可以提供关于矢量ALU、标量ALU、内存和分支指令组合的见解。

**工作流程：**

1. **确定计数器**：确定需要哪些硬件计数器来调查潜在瓶颈（例如，如果怀疑内存限制，则使用内存计数器；对于计算，则使用VALU计数器）。您可以使用 `rocprof --list-basic` 或 `--list-derived` 列出可用计数器。
2. **性能分析**：使用`rocprof`运行您的应用，并指定要收集的计数器或指标。一个典型的调用可能如下所示：`rocprof --stats -o results.csv ./my_ml_app`。对于适合RGP的更详细的跟踪分析，您可以使用标志生成`.sqtt`文件。
3. **分析**：
   - **CSV/文本输出**：分析生成的`results.csv`或控制台输出，计算相关指标（例如缓存命中率、带宽）并比较核心性能。
   - **RGP可视化**：将生成的跟踪文件（例如`.sqtt`、`.rgp`）加载到Radeon GPU Profiler中。RGP提供时间线视图、核心持续时间分析、wavefront占用率图表以及与核心执行阶段相关的详细硬件计数器视图。这提供了一种强大的方式来可视化核心内部的执行流程和瓶颈。

### 有效地解释性能分析数据

收集数据只是第一步。正确解释它需要背景信息：

- **了解您的硬件**：了解您的目标GPU的理论峰值性能（FLOPS、内存带宽）和架构限制（缓存大小、寄存器文件大小、最大占用率）。将实际达到的指标与这些理论值进行比较。一个核心如果达到峰值DRAM带宽的80%，很可能受到内存限制。
- **找出限制因素**：使用性能分析数据确定主要瓶颈。占用率是否因寄存器而低？内存带宽是否饱和？执行单元是否未充分使用？是否由于特定依赖关系而出现明显的停滞？
- **回溯关联**：只要有可能，使用源代码关联功能（Nsight Compute）或在RGP中分析核心的行为，同时结合编译器的中间表示（例如MLIR、LLVM IR）或生成的汇编（PTX/GCN）。这有助于将低级问题（如高寄存器压力）与特定的高级操作或编译器决策（如激进融合导致大型核心）关联起来。
- **迭代优化**：性能分析是一个迭代过程。使用从性能分析中获得的见解来指导进一步的优化工作。这可能涉及调整编译器标志（例如，控制融合的激进程度、启用/禁用特定优化）、稍微修改机器学习 (machine learning)模型结构，甚至实现自定义核心。更改后，重新进行性能分析以衡量影响。

“通过使用Nsight Compute和ROCprof/RGP，您对GPU上编译的机器学习核心的执行获得了必要的了解。这种详细分析对于诊断顽固的性能问题以及确保编译器执行的复杂优化能在目标硬件上转化为速度提升非常有帮助。”

## 参考资料

- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA Nsight Compute的官方指南，详细介绍了其功能、指标和在NVIDIA GPU上的工作流程。
- [ROCm Profiling Tools (ROCprof, RGP)](https://docs.amd.com/en-us/latest/understand_rocm_profiling/index.html) — AMD (2024)
  AMD ROCm 性能分析工具的官方文档，涵盖用于计数器收集的ROCprof和用于可视化的Radeon GPU Profiler。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  提供了CUDA架构、编程模型和性能考虑的基础知识，这些知识与解释Nsight Compute指标直接相关。
- [AMD Instinct MI300 Series CDNA™ 3 Architecture: Programmer's Reference Guide](https://developer.amd.com/wp-content/resources/ROCm_CDNA_3_PRG_Public.pdf) — AMD (2024)
  详细介绍了AMD CDNA GPU的架构，解释了波前、寄存器和内存层次结构等概念，这对于理解ROCprof数据至关重要。
