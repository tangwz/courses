---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-ml-runtime-systems"
lesson: "runtime-async-scheduling"
sourceId: 4548
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-6-advanced-ml-runtime-systems/runtime-async-scheduling"
title: "异步执行与调度"
description: "实现异步任务执行模型和调度算法，以便重叠计算和数据传输。"
order: 4
plots: ["plots/4548-0.json"]
sourceHash: "610d6389a8e5b0469445aaccb30d617c887a158cb1a941d81c166e99aefe45a3"
sourceCorrections: []
---

现代机器学习 (machine learning)工作负载，特别是部署在异构硬件上的工作负载，通常涉及一系列复杂的操作，包括数据传输、CPU上的预处理/后处理，以及GPU或专用NPU等加速器上的密集计算。顺序执行这些操作可能导致硬件资源大量未充分利用，因为一个组件在另一个组件完成任务时常常处于闲置等待状态。异步执行和精巧的调度是运行时不可或缺的功能，它们能有效减少这些低效情况并提升吞吐量 (throughput)。

其主要思想是将机器学习模型推理 (inference)或训练步骤表示为任务的有向无环图（DAG），节点代表操作（例如，内核启动、内存复制、同步事件），边代表依赖关系。运行时的职责是尽可能高效地执行此图，在遵守依赖关系的同时，抓住并行和重叠的机会。

### 任务与依赖的表示

运行时管理的每个工作单元都封装为一个任务。一个任务可能对应于：

- 一个 GPU 内核启动（例如，`cudaLaunchKernel`、SYCL 内核提交）。
- 一个内存传输操作（例如，`cudaMemcpyAsync`、`sycl::queue::memcpy`）。
- 一个 CPU 密集型计算。
- 同步原语（例如，记录或等待事件）。

依赖关系决定执行顺序。例如，GPU 内核在其输入数据复制到 GPU 内存之前无法执行，而主机到设备的数据复制在其源数据在主机上准备就绪之前无法开始。这些依赖关系构成了任务 DAG 的结构。

> 一个简化的任务图，显示了依赖关系。内核 1 和 2 在各自数据传输完成后，可能在不同的流上并行运行。内核 3 依赖于内核 1 和内核 2 的完成，可能通过事件进行同步。

### 执行原语：流与事件

硬件加速 API 提供了异步执行的原语。在 CUDA 中，这些是*流*，而在 SYCL/OpenCL 中，它们是*队列*。入队到同一流/队列的操作通常由设备顺序执行，但不同流/队列上的操作可以并行执行，这取决于硬件资源可用性和明确的依赖关系。

- **流/队列：** 表示提交给设备的独立操作序列。将任务提交到不同流可以使硬件调度器潜在地重叠它们的执行。
- **事件：** 作为流内部或流之间的同步点。一个事件可以在一个流上的任务完成后被记录，而另一个流可以被指示等待该事件，然后才执行后续任务。这对于管理跨流依赖关系很重要，如上图所示，内核 3 等待信号内核 1 和 2 完成的事件。

### 任务图的调度算法

任务图定义后，运行时调度器会决定实际的执行顺序和时机。常见的策略包括：

1. **拓扑排序：** 执行 DAG 的基本方法。任务按照遵守所有依赖关系的顺序执行。一个简单的实现会维护每个任务的待处理依赖计数，并在计数为零时将任务添加到“就绪队列”。
2. **就绪队列管理：** 调度器通常维护一个依赖已满足的任务队列。调度器从该队列中选择任务，分派给可用的执行资源（例如，GPU 流、CPU 线程）。选择任务的策略可以有所不同（先进先出、后进先出、基于优先级）。
3. **基于优先级的调度：** 可以为任务分配优先级。例如，图关键路径上的任务可能获得更高的优先级，以最小化整体延迟。内存传输可能被优先处理，以确保计算单元持续有数据输入。
4. **数据局部性感知调度：** 在具有复杂内存层次结构的系统中（例如，CPU 上的 NUMA、GPU 上的 HBM 与 DRAM），调度器可能会优先处理那些操作已驻留在有利内存位置的数据的任务。

调度器的目标通常是通过使计算单元保持忙碌，并尽可能使数据传输与计算重叠来最大化资源利用率。

### 实现计算与通信重叠

通过流/队列的异步操作是实现重叠的主要方式。考虑一个典型的模式：复制输入数据（主机到设备，H2D）、在设备上计算、将结果复制回（设备到主机，D2H）。

- **同步执行：** 每个步骤都等待上一个步骤完成。CPU 在 GPU 计算期间可能闲置，GPU 在数据传输期间也可能闲置。
- **异步执行：** 为计算和内存复制使用单独的流，允许运行时在不等待的情况下发出这些操作。例如：
  1. 发出块 1 的 H2D 复制（流 M）。
  2. 发出块 0 的计算（流 C），*依赖于 H2D 块 0 完成*。
  3. 发出块 -1 结果的 D2H 复制（流 M），*依赖于计算块 -1 完成*。

通过使用多个流和适当的事件同步来流水线化数据块的处理，运行时可以大幅度地将数据传输的延迟隐藏在计算背后。



![同步与异步执行时间线](plots/4548-0.json)



> 执行时间线的比较。在同步执行中，操作顺序发生。在异步执行中，下一/前一块的数据传输（H2D，D2H）可以与当前块的 GPU 计算重叠，从而减少总执行时间。

### 同步考量

异步性虽具强大能力，但有时仍需明确的同步。例如，CPU 可能需要等待 GPU 结果才能继续后续逻辑，或者在最终归约步骤之前，来自多个并行流的结果必须就绪。

运行时 API 提供了同步原语：

- `cudaStreamSynchronize(stream)` / `queue.wait()`：阻塞调用 CPU 线程，直到指定流/队列上所有先前提交的操作完成。
- `cudaEventSynchronize(event)` / `event.wait()`：阻塞调用 CPU 线程，直到指定事件被记录（即，相关任务完成）。
- `cudaStreamWaitEvent(stream, event)` / `queue.ext_oneapi_submit_barrier(event_list)`：在流/队列上入队一个等待操作；该流/队列上的后续操作直到事件被记录后才会开始。这*不会*阻塞主机线程。

过多或放置不当的同步点会抵消异步执行的优势，重新引入序列化点并降低并行性。运行时设计者必须谨慎管理同步，以确保正确性而不牺牲性能。最小化主机-设备同步点通常是一个重要的优化目标。

### 异步调度中的挑战

设计有效的异步调度器涉及多项挑战：

- **开销：** 管理任务图、依赖关系、事件和流会产生运行时开销。这种开销必须相对于任务自身的执行时间而言是最小的。
- **资源争用：** 即使存在重叠，操作也可能争夺有限的硬件资源，如内存带宽或加速器上的特定执行单元。调度器可能需要启发式方法来缓解这种争用。
- **动态性：** 如果任务图动态变化（例如，由于动态形状或控制流），调度器必须快速高效地适应。
- **复杂性：** 实现和调试多流、事件驱动的异步执行逻辑本身就具有复杂性。

熟练掌握异步执行和调度对于构建能够充分发挥现代硬件能力的高性能机器学习 (machine learning)运行时不可或缺。它需要深刻理解目标硬件的并发机制，谨慎管理依赖关系，以及巧妙的调度策略来有效重叠操作。

## 参考资料

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA
  CUDA编程的综合指南，详细介绍了使用流和事件的异步执行、内存管理以及NVIDIA GPU的优化技术。
- [SYCL 2020 Specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) — Khronos Group (2020)
  Publisher: Khronos Group
  对SYCL异构计算标准的定义，描述了任务图、队列、事件以及主机-设备同步，以实现可移植的并行编程。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  介绍了TVM，一个深度学习编译器栈，通过转换和调度计算图来优化异构硬件上的执行，直接解决了ML工作负载的异步执行和资源管理问题。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-12-811986-0) — David B. Kirk, Wen-mei W. Hwu (2016)
  Publisher: Morgan Kaufmann
  一本关于GPU编程的基础教材，详细解释了并行架构、CUDA流、事件以及优化异构计算和数据传输重叠的高级技术。（第4版）
