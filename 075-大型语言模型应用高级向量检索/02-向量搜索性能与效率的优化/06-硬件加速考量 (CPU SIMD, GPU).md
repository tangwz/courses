# 硬件加速考量 (CPU SIMD, GPU)

来源：[原文](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/hardware-acceleration-considerations)

[返回章节目录](README.md) · [返回课程目录](../README.md)

了解底层硬件对向量 (vector)搜索性能的影响是必要的。尽管量化 (quantization)和高效索引等算法优化可以提高效率，但大多数向量搜索场景中的核心操作——计算查询向量与数据库向量之间的距离——计算量仍然很大，尤其是在大规模情况下。借助特定的硬件能力可以显著改善延迟和吞吐量 (throughput)。CPU 单指令多数据指令和图形处理器（GPU）可以用来加速这些计算。

### 借助 SIMD 发挥 CPU 并行性

现代 CPU 具有单指令多数据 (SIMD) 指令集，例如 SSE (流式单指令多数据扩展)、AVX (高级向量 (vector)扩展)、AVX2 和 AVX-512。这些指令使得单个 CPU 核心能够在特殊的宽寄存器（128、256 或 512 位）内同时对多个数据点执行相同的操作。

考虑计算两个向量 $x$ 和 $y$ 之间的平方 L2 距离：


$$
d^2(x, y) = \sum_{i=1}^{D} (x_i - y_i)^2
$$


传统的（标量）实现方式会遍历每个维度 $D$，计算差值，将其平方，然后累加到累加器中。借助 SIMD，多个维度（例如，4、8 或 16 个单精度浮点数，具体取决于指令集和寄存器宽度）可以在单个指令周期内并行处理。多个 $i$ 对应的差值 $(x_i - y_i)$ 可以一次性计算，并行平方，然后使用专门的 SIMD 指令进行求和。

**其在向量搜索中的应用：**

- **距离计算：** SIMD 为欧几里得 (L2) 距离、点积（内积，通常与余弦相似度有关）以及其他常见距离度量的计算提供显著加速。这直接加速了最近邻查找过程，无论是暴力扫描还是 HNSW 或 IVF 等 ANN 算法的搜索阶段。
- **库支持：** 许多高性能向量搜索库，例如 Faiss（CPU 版本）和 ScaNN，都经过专门优化，以便运用可用的 SIMD 指令。NumPy 等数值计算库，当链接到优化的 BLAS/LAPACK 实现（如 Intel MKL 或 OpenBLAS）时，也会运用 SIMD 进行向量化 (quantization)操作。
- **实际影响：**
  - 确保您的向量搜索库使用适当的编译标志进行编译，以便为您的目标 CPU 架构启用 SIMD 优化。通用构建版本与 AVX2/AVX-512 优化构建版本之间的性能差异会非常大。
  - 对于某些 SIMD 指令，数据对齐 (alignment)可能对于获得最佳性能是必需的，尽管现代编译器和库通常会处理这个问题。
  - 当向量维度是 SIMD 寄存器宽度容量的倍数时（例如，AVX2 对单精度浮点数是 8 的倍数），效果最为显著。

如果您的硬件支持 SIMD 并且您的软件正确运用它，SIMD 会带来“免费”的性能提升。它加速了单个核心上的计算，无需现代 CPU 上的专用硬件。

### 借助 GPU 实现大规模并行

图形处理器（GPU）提供另一种加速方法。GPU 最初设计用于渲染图形，其架构具有数千个相对简单的处理核心，针对并行计算进行了优化。这使得它们非常适合涉及大规模矩阵运算，对我们来说更重要的是，非常适合对海量数据集进行距离计算。

**运用 GPU 进行向量 (vector)搜索：**

- **暴力搜索加速：** 对于暴力搜索，GPU 可以同时计算查询向量与数百万个数据库向量之间的距离，显著减少搜索时间，即使与 SIMD 优化的 CPU 实现相比也是如此，尤其对于大型数据集。
- **ANN 加速：** GPU 可以加速 ANN 算法的组成部分：
  - **IVF：** 将查询分配到质心的距离计算，以及在选定 Voronoi 单元（倒排列表）中的距离计算。
  - **HNSW：** 图遍历过程中的距离计算，以查找候选邻居。
  - **量化 (quantization)：** GPU 可以加速乘积量化码本分配所需的距离计算或精细距离计算。
- **专用库：** 像 Faiss-GPU 这样的库专门设计用于运用 GPU 的能力。它们通常实现针对向量搜索原语进行了优化的 CUDA 内核。像 NVIDIA RAPIDS RAFT 这样的框架也提供 GPU 加速的原语，对构建最近邻搜索系统很有帮助。



[交互图表：距离计算的相对加速](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/hardware-acceleration-considerations#plot-1m17lkk)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "标量 CPU",
        "CPU SIMD (AVX2)",
        "中端 GPU"
      ],
      "y": [
        1,
        6,
        75
      ],
      "marker": {
        "color": [
          "#adb5bd",
          "#4263eb",
          "#12b886"
        ]
      }
    }
  ],
  "layout": {
    "title": "距离计算的相对加速",
    "yaxis": {
      "title": "相对速度 (标量 CPU = 1倍)"
    },
    "xaxis": {
      "title": "硬件方法"
    },
    "bargap": 0.4
  }
}
```

</details>



> 这说明了在大型数据集上使用不同硬件方法进行距离计算的性能提升。实际加速效果很大程度上取决于具体硬件、数据集大小、向量维度和软件实现。

**权衡与考量：**

- **成本：** 高性能 GPU 与 CPU 相比，代表着一项重要的硬件投入。
- **功耗：** GPU 通常比 CPU 消耗更多电力，尤其是在高负载下。
- **数据传输：** 通过 PCIe 总线在 CPU 主内存和 GPU 显存 (VRAM)之间移动数据（向量、索引结构）会引入延迟。高效的实现需要最大程度地减少这些传输，通常通过将索引驻留在 GPU 上来完成。
- **显存限制：** GPU 内存（显存）量限制了可以完全存储并在 GPU 上处理的索引大小。对于大于可用显存的索引，策略包括将索引分布到多个 GPU 上，或使用从 CPU 内存分页数据的技术（会影响性能）。
- **批处理：** GPU 性能通常会因将多个查询批量处理而显著受益，从而更好地运用并行核心。如果批次大小较小，单个查询的延迟可能不总是低于优化的 CPU 实现。

GPU 通常在大型数据集（数百万到数十亿向量）上需要非常低延迟或需要极高查询吞吐量 (throughput)的场景中受到青睐，在这些情况下，对专用硬件的投资是合理的。

### 选择合适的方法

依赖 CPU SIMD 还是投资 GPU 的决定取决于几个因素：

- **规模：** 对于较小的数据集（例如，小于 1-10 百万向量 (vector)，具体取决于维度），高度优化的 CPU SIMD 实现通常可以提供足够的性能和低延迟。随着数据集变大，GPU 的并行处理能力变得越来越有优势。
- **预算：** GPU 会增加显著成本。CPU SIMD 随现代处理器“内置”提供。
- **延迟 vs. 吞吐量 (throughput)：** GPU 在吞吐量方面表现出色，尤其是在批量查询时。对于小型数据集上的单查询、超低延迟要求，优化的 CPU 代码有时可能具有竞争力，尤其是在考虑 GPU 的 PCIe 传输开销时。
- **现有基础设施：** 部署限制、可用硬件、功耗/散热限制以及团队专业知识（CUDA 编程、GPU 管理）都发挥作用。

另外值得一提的是，存在一些混合方法，部分计算可能在 CPU 上进行（例如，像 IVF 分配这样的粗粒度量化 (quantization)），而更精细的距离计算则卸载到 GPU。

### 其他加速器

谷歌的张量处理单元 (TPU) 或专用 AI 芯片等专用硬件加速器正在出现。这些可以为机器学习 (machine learning)中常见的特定操作（包括距离计算）提供出色的性能。然而，它们的使用通常需要特定的软件生态系统（例如，TPU 的 TensorFlow、JAX），并且目前可能不如 CPU/GPU 解决方案那样普遍可用或灵活。

归根结底，理解不同硬件架构如何加速距离计算这一基本操作，对于优化向量 (vector)搜索非常重要。在不同硬件上以及使用适当编译的库对您的特定工作负载进行性能分析，是确定最佳方法以满足您的应用程序性能目标的最有效方式。

## 参考资料

- [Faiss: A Library for Efficient Similarity Search](https://proceedings.neurips.cc/conference/2017/advances-in-neural-information-processing-systems-30-2017) — Jeff Johnson, Matthijs Douze, Hervé Jégou (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Neural Information Processing Systems Foundation; Pages: 6893-6902
  描述了一个用于高效相似性搜索的领先库，详细介绍了其优化措施，包括CPU SIMD指令和GPU加速。
- [Programming Massively Parallel Processors: A Hands-on Approach](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj (2022)
  Publisher: Elsevier; DOI: [10.1016/C2020-0-02969-5](https://doi.org/10.1016/C2020-0-02969-5)
  一本关于CUDA编程和GPU架构的综合教材，对于理解GPU加速原理至关重要。

---

[上一节](05-%E5%B0%86%E5%85%83%E6%95%B0%E6%8D%AE%E4%B8%8E%E5%90%91%E9%87%8F%E4%B8%80%E5%90%8C%E9%AB%98%E6%95%88%E7%B4%A2%E5%BC%95.md) · [下一节](07-%E5%86%85%E5%AD%98%E7%AE%A1%E7%90%86%E4%B8%8E%E7%BC%93%E5%AD%98%E7%AD%96%E7%95%A5.md)
