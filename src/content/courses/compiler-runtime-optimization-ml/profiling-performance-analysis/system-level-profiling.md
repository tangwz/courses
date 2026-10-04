---
course: "compiler-runtime-optimization-ml"
chapter: "profiling-performance-analysis"
lesson: "system-level-profiling"
sourceId: 4587
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/system-level-profiling"
title: "系统级性能分析 (CPU、GPU、互连)"
description: "概述了Linux perf、NVIDIA Nsight Systems、AMD uProf等用于全系统分析的工具。"
order: 2
plots: []
sourceHash: "d09092f961f51cca8d3fe676fa7dda431b7978992c791708712bc49c94608baa"
sourceCorrections: []
---

在对单个CPU函数或GPU核进行细致的、周期精确的分析之前，了解系统整体运行表现十分重要。系统级性能分析提供了这种宏观视角，协助您识别主要资源限制：是受限于CPU处理、GPU计算、内存带宽、主机与设备之间的数据传输速度，还是这些因素的组合？回答这个问题能更有效地引导您后续的优化工作。

编译后的机器学习 (machine learning)模型通常涉及复杂的相互作用，在主机CPU（运行运行时、调度操作，并可能执行某些算子）、加速器（GPU、TPU等，执行大部分计算）、系统内存、设备内存以及连接它们的互连（如PCIe或NVLink）之间。优化单个核函数可能整体效果不大，如果真正的性能瓶颈在于CPU上的数据准备或互连上的传输时间。系统级性能分析工具旨在捕获这些组件的并发活动，呈现统一的执行时间线。

### 考量指标与观察事项

在系统层面，您应着重观察：

- **CPU利用率：** CPU是否持续繁忙？某些核心是否饱和，而其他核心空闲？当GPU本应活跃时出现高CPU使用率，可能表明存在受限于CPU的预处理、低效的运行时调度或驱动开销。工具通常能区分用户时间（应用程序代码）和内核时间（操作系统/驱动活动）。
- **GPU活动情况：** GPU是否在预期时间活跃？核函数启动之间是否存在显著空隙？整体GPU利用率百分比可能具有误导性；显示核函数执行时长和启动开销的时间线视图更具参考价值。持续的、背靠背的核函数执行通常是理想的。
- **内存使用：** 监测主机（RAM）和设备（GPU VRAM）的内存消耗。过多的内存使用可能导致分配失败或依赖较慢的内存层级。观察内存分配模式和潜在的压力点。
- **互连流量：** 测量连接CPU和GPU的总线（通常是PCIe）的带宽利用率。数据传输是否占满了可用带宽？相较于计算时间，传输时间过长通常表明有优化空间，例如重叠传输与计算，或通过算子融合或不同数据布局完全减少数据移动。
- **API调用与事件：** 追踪运行时API（例如CUDA API调用、ROCm HIP调用）和相关操作系统事件的执行。这有助于将高层操作（如启动核函数或传输数据）与观察到的硬件活动联系起来，并发现运行时或驱动层中潜在的开销。

### 常用系统级性能分析工具

有多种工具提供全系统性能可见性。选择通常取决于具体的硬件和软件环境。

- **Linux `perf`：** 一个集成在Linux内核中的多功能工具。`perf`可以采样CPU性能计数器，追踪内核事件，分析特定进程，并提供关于CPU缓存行为、分支预测和系统调用的信息。虽然主要侧重于CPU，但其追踪内核事件（如调度和I/O）的能力为系统整体行为提供了背景。对于诊断受限于CPU的情况或高内核/驱动开销特别有用。
- **NVIDIA Nsight Systems (`nsys`)：** 这是NVIDIA GPU平台上进行系统级性能分析的常用工具。它的主要优势在于生成详细的时间线，关联CPU线程活动、CUDA API调用、CUDA核函数执行、内存传输（memcpy、页面迁移）、NVLink流量、操作系统运行时事件，以及适用的DirectX或Vulkan调用。这种统一视图对于定位导致停顿或效率低下的交互非常重要，例如CPU等待GPU结果、GPU等待CPU数据或PCIe带宽限制。
- **AMD ROCm开发工具 (Radeon GPU Profiler, `rocprof`)：** 对于使用ROCm的AMD GPU平台，`rocprof`提供命令行分析功能，可收集核函数执行时间、API调用追踪（HIP）和性能计数器。Radeon GPU Profiler (RGP) 提供GUI来可视化这些数据，提供类似于Nsight Systems的时间线视图，显示核函数分派、内存传输（HSA信号）和主机API调用。这有助于识别AMD硬件上CPU-GPU交互流程中的瓶颈。
- **Intel VTune Profiler：** 虽然在CPU和Intel GPU微架构的细致分析方面表现出色（稍后会介绍），但VTune也提供平台级分析模式。它可以捕获CPU活动、集成/独立Intel GPU使用情况、内存带宽和系统事件，并将它们呈现在时间线上。对于在Intel CPU和GPU上运行的工作负载，它特别有用，能提供关于线程调度、跨平台内存访问模式以及潜在I/O瓶颈的信息。
- **通用系统监控工具：** `htop`、`dstat`、`vmstat`和`iostat`等工具提供CPU负载、内存使用、磁盘I/O和网络活动的实时或记录快照。虽然对于关联特定机器学习 (machine learning)操作的细节较少，但它们对于快速检查整体系统健康状况和识别大规模资源饱和（例如，RAM耗尽、持续磁盘交换）非常有用。

### 理解系统级时间线

Nsight Systems或Radeon GPU Profiler等工具生成的时间线视图通常是最具参考价值的输出。在分析编译后的机器学习 (machine learning)工作负载的这些时间线时，请留意以下模式：

1. **GPU空闲：** GPU长时间处于空闲或利用率不足的状态。检查这些空隙期间的CPU活动。CPU是否忙于准备下一批数据、运行不受支持的算子，或被I/O阻塞？这表明可能存在CPU端的瓶颈或低效的调度。
2. **CPU瓶颈：** CPU持续高利用率，可能仅在少数核心上，而GPU处于等待状态。这可能发生在数据加载/预处理、运行时图操作期间，或者某些操作在CPU而非加速器上执行时。
3. **数据传输开销：** 主机和设备之间代表内存拷贝（例如`cudaMemcpy`、`hipMemcpy`）的长条在时间线上占据主导。这表明互连（PCIe）带宽可能是一个限制因素，或者数据移动可以减少或与计算重叠。将实际达到的带宽与理论最大值进行比较。
4. **高启动延迟 / 小核函数：** 频繁的、短小的GPU核函数之间夹杂着明显的空隙。这可能表明相较于执行时间，核函数启动开销较高，这通常出现在编译器未能有效融合的操作中。
5. **同步点：** 显式同步调用（例如`cudaStreamSynchronize`）会使CPU线程停顿以等待GPU结果。虽然有时是必要的，但频繁或长时间的停顿表明可能存在流水线气泡。

> 简化的时间线，说明了潜在的瓶颈。相较于核函数1 (k1)，预处理 (p2)、主机到设备传输 (t1) 或等待 (p4) 的长时间持续表明可能存在CPU、互连或同步瓶颈。Nsight Systems 等工具提供了此类时间线的更详细版本。

系统级性能分析是重要的初步诊断步骤。它提供了所需的背景，以理解在执行编译模型时，硬件和软件组件复杂交互中最显著的性能限制所在。在这里获得的认知将引导您使用更专业的CPU和GPU核函数分析器，我们将在后续章节中讨论它们，以分析特定计算密集型或内存密集型操作中的根本原因。

## 参考资料

- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  NVIDIA 平台上进行系统级性能分析的综合指南，涵盖 CPU、GPU、内存和互连活动。
- [ROCm Profiling Tools Documentation](https://rocm.docs.amd.com/en/latest/programming/profiling.html) — Advanced Micro Devices (AMD) (2025)
  AMD ROCm 分析工具的官方文档，详细说明了 rocprof 和 Radeon GPU Profiler 在系统级洞察方面的用法。
- [Intel VTune Profiler User Guide](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/) — Intel Corporation (2024)
  Publisher: Intel Corporation
  Intel VTune Profiler 用户指南，提供有关 Intel CPU 和 GPU 平台级分析的信息，包括时间线视图。
- [Linux perf Wiki](https://perf.wiki.kernel.org/index.php/Main_Page) — Linux Kernel Community (2024)
  Linux perf 工具的官方维基百科，提供 CPU 性能监控和内核事件追踪的文档和使用示例。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — David B. Kirk, Wen-mei W. Hwu (2016)
  Publisher: Elsevier
  第三版。一本关于 GPU 并行计算的基础著作，包含异构系统性能分析和优化策略的章节。
