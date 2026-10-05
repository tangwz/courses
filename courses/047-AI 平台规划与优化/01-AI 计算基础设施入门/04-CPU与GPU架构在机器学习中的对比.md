# CPU与GPU架构在机器学习中的对比

来源：[原文](https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-1-foundations-ai-compute-infrastructure/comparing-cpu-gpu-architectures)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管CPU和GPU都是基于硅的处理器，但它们的内部架构截然不同，分别针对完全不同的任务进行优化。对于构建或管理AI基础设施的人来说，理解这种区别非常重要，因为为任务选择错误的工具会导致性能瓶颈和资源浪费。CPU擅长串行任务处理，而GPU则擅长并行计算。

### CPU：用于顺序逻辑的少数强大核心

中央处理器（CPU）专为低延迟、单线程性能而设计。它由少量高度复杂的核构成，现代服务器中通常有4到64个。每个核心都是一个强大的单元，能够非常快速地执行单条指令流。

CPU的架构特点包括：

- **大容量缓存：** CPU拥有大量的L1、L2和L3缓存，用于存储常用数据和指令，从而最大限度地减少从较慢的主内存（RAM）获取信息的时间。
- **复杂控制逻辑：** CPU芯片的很大一部分空间用于复杂的控制单元，包括分支预测器和推测执行引擎。这使得CPU能够智能猜测接下来需要哪些指令，优化带有复杂条件逻辑（`if-else`语句）和不可预测访问模式的工作流程。

在机器学习 (machine learning)流程中，这些特点使得CPU对于数据预处理、文件系统管理、协调整体训练循环以及运行操作系统等任务来说是不可或缺的。这些通常是顺序操作，无法轻易分解成数千个更小、相同的任务。

### GPU：用于高吞吐量 (throughput)的数千个简单核心

相比之下，图形处理器（GPU）是一种为高吞吐量、并行处理而构建的架构。现代GPU不是拥有少数强大的核心，而是包含数千个更简单、更专业的核。

GPU的架构特点包括：

- **海量并行性：** GPU上的所有核心可以同时对不同数据片段执行相同的指令。这种模型通常被称为SIMT（单指令多线程）。
- **高带宽内存：** GPU配备有自己专用的高速内存（如GDDR6或HBM），通过非常宽的内存总线连接。这对于为数千个核心提供数据，防止它们闲置必不可少。
- **简单控制逻辑：** GPU用于复杂控制逻辑的硅片空间要少得多，并且每个核心的缓存也比CPU小。它们被优化用于在大量数据流上重复运行相同的计算，而不是处理复杂的、决策繁重的程序流程。

> CPU（少数强大核心）与GPU（许多更简单的核心，分组为流式多处理器）之间的架构差异。

### 为何GPU在深度学习 (deep learning)中占主导地位

大多数深度学习模型的核心涉及矩阵乘法。例如，神经网络 (neural network)中的单层可以表示为：


$$
\text{输出} = \text{激活}(\text{权重} \cdot \text{输入} + \text{偏置})
$$


`权重` $\cdot$ `输入`的运算是一个大规模矩阵乘法。考虑矩阵乘法 $C = A \cdot B$。每个元素 $C_{ij}$ 是通过 $A$ 的一行与 $B$ 的一列的点积计算得出的。重要之处在于 $C_{ij}$ 的计算完全独立于任何其他元素（如 $C_{kl}$）的计算。

这是一个完全可并行化的问题。GPU可以将每个输出元素或小元素组的计算分配给其数千个核心，从而比CPU更快地完成整个矩阵乘法，CPU则必须顺序计算或以非常有限的并行度计算。这就是为什么CPU可能需要数小时才能完成的任务，在GPU上只需几分钟。

### 并排比较

下表总结了主要的架构差异及其对机器学习 (machine learning)工作负载的影响。

| 特点 | CPU（中央处理器） | GPU（图形处理器） |
| --- | --- | --- |
| **主要设计** | 低延迟，串行处理 | 高吞吐量 (throughput)，并行处理 |
| **核心数量** | 少（4-64个），但功能强大 | 多（数千个），但更简单 |
| **机器学习中最佳用途** | 数据准备，控制流，小模型推理 (inference) | 深度学习 (deep learning)模型训练，大规模推理 |
| **内存** | 访问主系统RAM | 拥有自己的高带宽显存（VRAM） |
| **优势** | 复杂逻辑，分支，任务切换 | 大数据块上的重复算术运算 |
| **劣势** | 不擅长大规模并行数学运算 | 串行任务和复杂逻辑效率低 |

最终，现代AI系统并非是选择CPU*或*GPU的问题，而是关于如何将它们协同使用。CPU充当指挥者，调度数据流并处理程序的所有顺序部分，而GPU是引入的专用协处理器，用于处理并行计算的繁重任务，这使得现代深度学习成为可能。

## 参考资料

- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy, David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  为理解CPU和内存架构（包括缓存层级和控制逻辑）提供了全面的基础，对理解比较中的CPU部分至关重要。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj (2022)
  Publisher: Morgan Kaufmann
  解释了GPU架构、SIMT模型和并行计算原理，与GPU的描述及其在并行处理方面的优势直接相关。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  探讨了深度学习的计算需求，特别是大型矩阵运算，阐明了为何GPU特别适合这些任务。
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  详细介绍了CUDA编程模型和底层GPU架构，包括流式多处理器和内存层次结构等核心概念。

---

[上一节](03-GPU%20%E5%9C%A8%E5%8A%A0%E9%80%9F%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md) · [下一节](05-TPU%E5%8F%8A%E5%85%B6%E4%BB%96ASIC%E7%AE%80%E4%BB%8B.md)
