---
course: "compiler-runtime-optimization-ml"
chapter: "profiling-performance-analysis"
lesson: "memory-access-analysis"
sourceId: 4591
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/memory-access-analysis"
title: "内存访问模式分析"
description: "使用性能分析工具检测低效的内存访问模式和带宽限制。"
order: 6
plots: ["plots/4591-0.json"]
sourceHash: "c276b3133b183472c83f143186f4a72ff539bdaa6202b4811133fb7a5f4caf25"
sourceCorrections: []
---

机器学习 (machine learning)模型的高效运行，尤其是在经过多层编译器优化后，很大程度上取决于内存系统的高效利用。计算密集型操作在优化后可能变为内存密集型，而编译器施加的复杂变换（如算子融合、布局改变和分块）直接影响数据访问方式。因此，对内存访问模式进行性能分析，对于验证这些优化的效果以及找出剩余瓶颈非常重要。通用性能分析工具和专门的硬件工具提供了观察内存子系统行为的必要手段。

### 识别内存低效问题

需要注意的重要模式包括：

1. **低带宽利用率：** 硬件有理论峰值内存带宽（例如，到DRAM的GB/s）。如果核函数持续达到明显较低的带宽，这通常表明访问模式效率不高，例如读取许多小的、非连续的数据块，而非大的、连续的块。即使计算单元很忙，低内存带宽利用率也表明内存系统没有得到有效的数据供应，这可能限制整体吞吐量 (throughput)。
2. **高缓存未命中率：** 编译器采用分块和布局转换等技术，专门用于改善数据局部性，最大限度地利用更快的缓存级别（CPU上的L1、L2、LLC；GPU上的L1/纹理、L2、共享内存）。性能分析工具报告的高未命中率表明这些优化对于特定问题规模或硬件来说不足，或可能适得其反。这导致频繁、高延迟地访问较慢的内存级别（如DRAM）。
   - **空间局部性问题：** 访问内存中相距较远的数据元素，即使它们在时间上需要按顺序访问（例如，当矩阵按行主序存储时按列遍历），也会削弱缓存预取机制和空间局部性原则。
   - **时间局部性问题：** 如果相同数据在被重用之前就被从缓存中逐出，导致必须重复从较慢的内存中获取，这表明时间局部性差，通常可以通过更好的分块或融合来解决。
3. **非合并内存访问（GPU）：** GPU通过让warp内（通常32个线程）的线程在单个事务中同时访问连续的内存位置来实现高内存带宽。如果线程访问分散的位置（跨步访问），硬件必须发出多个内存事务来满足warp的请求，从而大幅降低有效带宽。性能分析工具会报告与每次请求的内存事务相关的指标，突显合并访问效率。
4. **共享内存组冲突（GPU）：** GPU上的共享内存分成多个组。如果warp内的多个线程尝试同时访问属于同一组的地址，则会引起组冲突，使访问串行化并引入延迟。性能分析工具通常可以检测并报告这些冲突的频率。
5. **过多的主机-设备数据传输：** 对于带有独立GPU或加速器的系统，通过PCIe总线（或等效互连）传输数据会产生显著的开销。系统级性能分析工具可以量化 (quantization)在这些传输上花费的时间。虽然有时无法避免，但尽量减少传输的频率和大小，并利用异步操作使传输与计算重叠，这很重要。融合等编译器优化旨在减少中间张量传输。
6. **NUMA效应（多插槽CPU）：** 在非统一内存访问（NUMA）系统上，访问连接到不同CPU插槽的内存会导致更高的延迟和更低的带宽。性能分析工具可以帮助识别跨插槽内存流量，这可能暗示次优的进程/线程亲和性设置或运行时的数据分配不佳。

### 使用性能分析工具进行内存分析

不同的工具提供了对内存性能各个方面的洞察：

- **GPU性能分析工具（例如NVIDIA Nsight Compute, AMD ROCprof）：** 这些工具对核函数的详细分析非常重要。

  - **内存吞吐量 (throughput)：** 报告不同内存级别（全局DRAM、L2、L1/共享内存）实际达到的带宽。将其与特定GPU的理论最大值进行比较。
  - **内存延迟分析：** 提供执行停顿的原因，通常会定位高延迟内存指令（例如，`L1MISS_STALL`、`L2MISS_STALL`）。
  - **合并访问指标：** 量化 (quantization)内存事务效率（例如，每次请求的事务数、扇区吞吐量）。少量请求对应大量事务表示合并效果不佳。
  - **缓存指标：** 详细说明L1/纹理缓存和L2缓存的命中/未命中率。
  - **共享内存指标：** 显示共享内存吞吐量和组冲突。
- **CPU性能分析工具（例如Intel VTune Profiler, Linux `perf`）：** 这些工具对于分析工作负载中CPU密集部分或主机端运行时很有用。

  - **缓存未命中：** `perf stat`或`perf record/report`等工具结合适当的硬件事件计数器（例如，`cache-misses`、`LLC-load-misses`）可以识别缓存性能问题。VTune提供了详细的微架构分析功能。
  - **内存带宽：** 使用性能监控单元（PMU）计数器（例如，通过`perf`或VTune的内存访问分析）监控带宽。
  - **TLB未命中：** 转换旁路缓冲（TLB）未命中会增加延迟；性能分析工具可以跟踪这些事件。
- **系统级性能分析工具（例如NVIDIA Nsight Systems, AMD uProf, Intel CoFluent）：** 提供整个系统的时间线视图。

  - **主机-设备传输：** 可视化`cudaMemcpy`或等效调用，显示持续时间、达到的带宽以及它们是否与核函数执行重叠。
  - **API调用：** 追踪与内存分配或管理相关的运行时API调用。
  - **CPU/GPU交互：** 显示CPU线程启动工作与GPU核函数执行之间的交互。

### 解释数据并与优化关联

性能分析工具的原始指标需要在ML模型和已应用的编译器优化的背景下进行解释。

- **低带宽 + 高延迟停顿：** 通常指向受延迟限制的核函数，可能是由于分散的内存访问、指针追逐或每线程非常小的工作集。检查GPU合并访问指标和CPU缓存未命中分析。循环融合是否可能创建了具有复杂访问模式的过于复杂的核函数？
- **高带宽 + 计算停顿：** 表明内存系统正在有效传输数据，但计算单元是瓶颈。这通常是内存优化后的理想状态。
- **高缓存未命中率：** 质疑编译器所做的分块或布局选择的有效性。分块大小是否适合缓存大小？`NCHW`到`NHWC`的转换是否确实改善了目标硬件上操作序列的局部性？性能分析数据可以指导调整编译器启发式方法或手动调优。



![内存带宽利用率示例（GPU DRAM）](plots/4591-0.json)



> GPU上不同核函数的带宽利用率示例，该GPU的理论峰值DRAM带宽为900 GB/s。核函数B（GEMM）显示出良好的利用率，而核函数A和C可能受到除原始带宽之外的因素限制，例如访问模式或延迟。

- **非合并访问/组冲突：** 直接表明GPU架构的代码生成不理想。检查核函数对应的源代码或中间表示（如PTX/GCN）。共享内存是否得到了有效利用？循环调度或线程映射能否改进？

分析内存访问模式直接反映了以内存为中心的编译器优化的成功与否。它有助于区分计算密集型和内存密集型场景，指导进一步的优化工作，无论是针对计算调度（如果内存高效）还是改善数据布局、分块、预取或合并（如果内存是瓶颈）。通过将性能分析工具数据与特定的编译器阶段和变换关联起来，可以对优化如何与硬件实际情况互动有更清楚的认知。

## 参考资料

- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy and David A. Patterson (2017)
  Publisher: Elsevier
  提供对内存层次结构、缓存和性能瓶颈的详细理解，这对于解释内存访问模式至关重要。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  CUDA编程的官方指南，详细介绍了与GPU和Nsight分析器使用相关的内存层次结构、访问模式和优化技术。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  讨论深度学习编译器（如TVM）如何应用直接影响内存访问模式和整体性能的转换（例如，操作符融合、平铺、数据布局）。
- [Intel VTune Profiler User Guide](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide) — Intel Corporation (2023)
  Publisher: Intel Corporation
  领先CPU分析器的官方文档，解释如何在CPU架构上识别缓存未命中、内存带宽问题和NUMA效应。
