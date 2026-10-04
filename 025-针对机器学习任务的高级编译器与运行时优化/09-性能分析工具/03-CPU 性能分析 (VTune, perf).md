# CPU 性能分析 (VTune, perf)

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/cpu-performance-analysis)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管 GPU 和专用加速器通常处理大部分机器学习 (machine learning)工作负载的计算，但 CPU 仍是重要的组成部分。它执行模型图的一部分，驱动加速器，管理运行时系统，处理数据加载和预处理，以及运行控制流逻辑。因此，优化 CPU 性能对应用程序整体速度非常重要。详细介绍了如何使用专门的 CPU 性能分析工具，特别是 Intel VTune Profiler 和 Linux `perf`，来分析在 CPU 上运行的已编译机器学习代码的性能。

如本章开头所述，对机器学习编译器转换后的代码进行性能分析带来特有的挑战。函数名可能被修改或对应于大型、融合的计算核心，这使得直接与原始模型关联变得困难。此外，大量使用库（如 MKL-DNN/oneDNN、OpenBLAS）意味着性能通常依赖于这些预优化的例程。有效的 CPU 性能分析需要能够超出源代码层面，并分析实际硬件执行细节的工具。

### Intel VTune Profiler

Intel VTune Profiler 是一款功能强大的性能分析工具，用于理解 CPU（以及 GPU/FPGA）的行为。它提供图形界面和多种分析类型，适合用于分析机器学习 (machine learning)工作负载。

#### 机器学习的分析类型

- **热点分析：** 通常是起点。它使用低开销的采样来发现 CPU 时间花费最多的函数或代码区域。对于已编译的机器学习代码，这通常指向特定的计算核心（生成的或来自库的）、运行时调度函数或内存管理例程。它有助于确定优化工作的优先级。
- **微架构审查：** 这种分析更深入，利用硬件性能监测单元（PMU）来收集关于 CPU 流水线的详细指标。您可以分析以下指标：
  - **每条完成指令的时钟周期数 (CPI)：** 高 CPI 表示效率低下；CPU 在每条有效指令上花费许多周期。
  - **缓存未命中：** 发现 L1、L2 或 L3 缓存未命中率高的情况，这会使 CPU 流水线停滞。这通常指示数据局部性差，可能可以通过编译器分块策略或布局转换来解决。
  - **分支预测错误率：** 高发生率表明控制流存在问题，有时可以通过编译器或运行时进行优化。
  - **向量 (vector)化强度：** 衡量 SIMD（单指令多数据）单元（如 AVX2、AVX-512）的利用效率。计算密集型核心中较低的强度表明编译器错失了向量化 (quantization)机会。
- **内存访问分析：** 对于张量密集型工作负载很重要。此分析有助于确定内存带宽瓶颈，区分缓存受限操作和 DRAM 受限操作，并分析内存延迟问题。它可以显示编译器循环转换或数据布局选择导致的低效内存访问模式。

#### 将 VTune 用于已编译的机器学习

您通常通过在其控制下启动机器学习应用程序或将其附加到已运行的进程来运行 VTune。VTune 收集数据并以各种视图（例如，摘要、自下而上、调用者/被调用者）呈现。您通常可以从函数级摘要向下查看源代码或汇编代码，尽管将高度优化、生成的代码关联回原始机器学习图操作需要理解编译器的转换。寻找直接显示汇编指令上 PMU 事件计数的注释，以进行精细分析。



[交互图表：VTune 热点分析](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-9-profiling-performance-analysis/cpu-performance-analysis#plot-7lldos)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "VTune 热点分析",
    "xaxis": {
      "title": "函数/模块"
    },
    "yaxis": {
      "title": "CPU 时间 (%)"
    },
    "barmode": "stack",
    "legend": {
      "traceorder": "reversed"
    }
  },
  "data": [
    {
      "type": "bar",
      "name": "L3 限制",
      "x": [
        "KernelA_Fused",
        "RuntimeScheduler",
        "libOneDNN.so",
        "DataPreprocessing",
        "Other"
      ],
      "y": [
        5,
        2,
        8,
        1,
        4
      ],
      "marker": {
        "color": "#ff922b"
      }
    },
    {
      "type": "bar",
      "name": "L2 限制",
      "x": [
        "KernelA_Fused",
        "RuntimeScheduler",
        "libOneDNN.so",
        "DataPreprocessing",
        "Other"
      ],
      "y": [
        8,
        1,
        5,
        2,
        5
      ],
      "marker": {
        "color": "#fcc419"
      }
    },
    {
      "type": "bar",
      "name": "L1 限制",
      "x": [
        "KernelA_Fused",
        "RuntimeScheduler",
        "libOneDNN.so",
        "DataPreprocessing",
        "Other"
      ],
      "y": [
        12,
        2,
        7,
        4,
        6
      ],
      "marker": {
        "color": "#94d82d"
      }
    },
    {
      "type": "bar",
      "name": "前端限制",
      "x": [
        "KernelA_Fused",
        "RuntimeScheduler",
        "libOneDNN.so",
        "DataPreprocessing",
        "Other"
      ],
      "y": [
        5,
        8,
        2,
        3,
        5
      ],
      "marker": {
        "color": "#74c0fc"
      }
    },
    {
      "type": "bar",
      "name": "指令完成",
      "x": [
        "KernelA_Fused",
        "RuntimeScheduler",
        "libOneDNN.so",
        "DataPreprocessing",
        "Other"
      ],
      "y": [
        10,
        2,
        18,
        10,
        10
      ],
      "marker": {
        "color": "#40c057"
      }
    }
  ]
}
```

</details>



> VTune 微架构审查的 CPU 时间细分示例，显示了潜在的瓶颈，例如内存停滞（L1/L2/L3 限制）或指令不足（前端限制）与有效工作（指令完成）的对比。`KernelA_Fused` 显示了可能因内存访问而停滞的大量时间。

### Linux `perf`

Linux `perf` 是一个功能强大、用途广泛的命令行性能分析工具，内置于 Linux 内核中。它使用 CPU 的性能监测单元（PMU）以低开销对硬件事件进行采样或计数。

#### `perf` 的常见用法

- **`perf stat`：** 为命令或进程 ID 提供常见硬件事件（周期、指令、缓存未命中、分支未命中）的总计计数。对快速了解性能特点很有用。

  ```bash
  # 统计整个机器学习推理脚本执行期间的事件
  perf stat python run_inference.py --model compiled_model.bin
  ```
- **`perf record`：** 对程序的执行进行采样。它定期记录指令指针和其他信息（例如使用 `-g` 选项的调用堆栈）。这会生成一个 `perf.data` 文件。

  ```bash
  # 记录带有调用图的性能数据
  perf record -g ./my_compiled_ml_app --input data.npy
  ```
- **`perf report`：** 分析由 `perf record` 生成的 `perf.data` 文件。它以分层视图显示在每个函数、库甚至单个指令中收集到的样本百分比。您可以在终端中交互式浏览此报告，以查看热点和调用链。
- **`perf annotate`：** 反汇编由 `perf report` 发现的热点函数，并用在每条指令处发生的样本百分比来标注汇编代码。这对于确定导致停滞或消耗周期的具体指令非常重要，尤其是在编译器生成的代码中。

#### `perf` 的优点和缺点

- **优点：** 轻量，在 Linux 系统上普遍存在，内核级可见性，高度可脚本化，支持大量的硬件和软件事件。
- **缺点：** 主要基于命令行（尽管存在外部 GUI），对于复杂的微架构问题，解释可能不如 VTune 直观；在高度优化的二进制文件中，将样本精确关联回高级源代码有时在没有正确调试符号的情况下可能具有挑战性。

#### 将 `perf` 用于已编译的机器学习 (machine learning)

与 VTune 类似，您可以使用 `perf record` 在已编译模型执行时捕获数据。`perf report` 通常会显示机器学习运行时执行引擎内部、机器学习编译器生成的特定核心或 oneDNN 或 OpenBLAS 等供应商库中的热点。在这些热点函数上使用 `perf annotate` 可以让您检查汇编代码，并查看硬件事件（例如通过 `perf record -e cache-misses ...` 记录的缓存未命中）频繁发生的位置。这可以直接为编译器开发或调整提供信息。例如，在生成的循环嵌套中看到加载指令上的大量缓存未命中可能表明编译器的分块或预取策略需要调整。

### 解释与行动

VTune 和 `perf` 都提供数据；技巧在于结合机器学习 (machine learning)编译的背景来解释这些数据。

- **高 CPI / 前端限制：** 可能表示指令获取瓶颈或复杂的依赖关系。编译器指令调度或代码布局优化可能有帮助。
- **高缓存未命中率 / 内存限制：** 表明数据局部性差。这指向改进编译器分块策略、数据布局转换（例如 NCHW 与 NHWC）或软件预取。
- **低向量 (vector)化强度：** 表示编译器未能有效利用 SIMD 单元。这可能需要更改循环转换、数据对齐 (alignment)或在代码生成期间显式使用向量内在函数。
- **运行时开销：** 如果大量时间花费在运行时函数（调度、内存分配）中，则表明运行时系统设计本身存在瓶颈，可能需要优化调度器或内存管理器。

通过系统地应用这些 CPU 性能分析工具，您可以摆脱猜测，获得关于已编译机器学习代码如何与硬件交互的具体数据。这些数据对于诊断性能限制以及指导本课程中讨论的复杂编译器和运行时优化非常重要。

## 参考资料

- [Intel® VTune™ Profiler User Guide](https://www.intel.com/content/www/us/en/develop/documentation/vtune-profiler-user-guide/) — Intel Corporation (2023)
  Publisher: Intel Corporation
  Intel VTune Profiler 的综合指南，涵盖分析类型、数据收集和结果解释，用于性能调优。
- [Intel® VTune™ Profiler for Performance Analysis of Deep Learning Inference Workloads](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-vtune-profiler-for-deep-learning-inference.html) — Intel Corporation (2022)
  Publisher: Intel Corporation
  一篇技术文章，详细说明了如何使用 Intel VTune Profiler 专门分析和优化深度学习推理工作负载，侧重于常见瓶颈和分析类型。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Elsevier
  一本计算机体系结构的经典教材，解释了 CPU 性能指标、缓存和流水线等原理，对于理解性能分析工具的输出至关重要。
- [Systems Performance: Enterprise and the Cloud](https://www.informit.com/store/systems-performance-enterprise-and-the-cloud-9780133390094) — Brendan Gregg (2013)
  Publisher: Prentice Hall
  一份 Linux 系统性能分析的综合指南，其中有大量章节专门介绍 `perf` 及其用于详细 CPU 性能分析和瓶颈识别的高级功能。

---

[上一节](02-%E7%B3%BB%E7%BB%9F%E7%BA%A7%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%20%28CPU%E3%80%81GPU%E3%80%81%E4%BA%92%E8%BF%9E%29.md) · [下一节](04-GPU%E6%A0%B8%E5%BF%83%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%20%28Nsight%20Compute%2C%20ROCprof%29.md)
