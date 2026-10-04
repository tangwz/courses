---
course: "compiler-runtime-optimization-ml"
chapter: "jit-compilation-ml"
lesson: "jit-motivation-ml"
sourceId: 4560
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-7-jit-compilation-ml/jit-motivation-ml"
title: "ML中JIT编译的动因"
description: "了解在特定情况下，JIT编译相对于即时执行或提前（AOT）编译的优势。"
order: 1
plots: ["plots/4560-0.json"]
sourceHash: "1b84212dfc25724569dfe7714606daeba2cdfa9b32cf2eb6ac9c4c0433d8cb1b"
sourceCorrections: []
---

机器学习 (machine learning)工作流常常需要在解释型、即时执行所提供的即时反馈和灵活性，以及完全编译代码所带来的性能潜力之间进行权衡。默认在PyTorch等框架中常见的即时执行，会按照宿主语言（通常是Python）的定义，逐个处理操作。尽管这种方法对于调试和快速原型开发非常出色，但它为每个操作带来了解释器和框架分派的显著开销。此外，由于编译器在任何给定时间只能看到计算的一小部分，这限制了全局优化的范围。

提前（AOT）编译通过在执行开始前分析并优化整个计算图来解决这些限制。TensorFlow Lite、TVM的AOT运行时或ONNX Runtime等系统通常采用AOT编译。这使得积极的图重写、算子融合、静态内存规划以及生成针对特定硬件目标高度优化的内核成为可能。主要缺点是对静态信息的要求。AOT编译器通常需要整个图结构，包括张量形状和控制流，在编译前就已确定。这对于处理动态输入尺寸（例如，可变的批次大小、自然语言处理模型中的序列长度或不同的图像分辨率）或由运行时数据确定的条件执行路径的模型可能具有限制性。处理动态性通常需要采用变通方法，例如将输入填充到最大尺寸、编译多个图变体（分桶）或插入运行时检查，所有这些都可能引入低效或增加部署复杂度。

即时（JIT）编译成为一种应对这种权衡的策略。通过将计算图的部分或全部的最终编译推迟到运行时，JIT编译器可以获得AOT编译期间不可用的动态信息，同时仍能避免纯即时执行的每个操作开销。这种运行时上下文 (context)提供了若干重要的优化机会：

### 动态形状专业化

ML部署中一个常见难题是处理具有不同维度的张量。考虑一个处理不同批次大小请求的推理 (inference)服务器，或一个处理不同长度句子的自然语言模型。AOT编译必须生成适用于 *任何* 允许形状的代码，这通常导致通用、效率较低的内核，或依赖于浪费计算和内存带宽的填充策略。

JIT编译可以直接解决这个问题。当在运行时遇到特定的张量形状，例如用于图像批次的$32 \times 3 \times 224 \times 224$，JIT编译器可以生成专门针对 *这些精确维度* 优化的内核代码。这种专业化使得以下成为可能：

- **最佳内存布局**：根据特定的算子序列和涉及的形状选择最佳数据格式（例如，NCHW对比NHWC）。
- **精确循环分块**：设计循环结构和分块大小，使其完美匹配给定维度的缓存大小和向量 (vector)单元长度，最大限度地提高数据局部性和并行性。
- **高效资源分配**：精确地为观察到的工作负载大小定制GPU内核中的线程块配置和共享内存使用。

通过按需编译专用代码，JIT避免了为处理广泛可能形状而设计的通用AOT代码所固有的性能折衷。

### 运用运行时值

除形状外，JIT编译有时可以运用在运行时已知的 *实际张量值*。例如，如果在特定调用期间，计算图段的某些输入被识别为常量，JIT编译器可以执行常量折叠或专门化依赖于这些值的条件逻辑分支。尽管这不如形状专业化常见，但这种能力使得在没有运行时上下文 (context)的情况下，完全提前编译时根本不可能实现的优化成为可能。

### 减少框架开销和实现融合

即时执行为其灵活性付出了代价：每个操作通常涉及对Python解释器和ML框架分派机制的调用。对于包含许多小操作的模型，这种开销可能会占据执行时间的大部分。

JIT编译器分析图中的操作序列。通过识别可融合的序列（例如，元素级操作序列，或卷积后跟激活函数 (activation function)），JIT可以将它们编译成一个更大、单一的计算内核。执行这个融合的内核只涉及一次分派，与通过框架单独执行每个组成操作相比，显著减少了开销。这种分派成本的均摊，加上融合内核内改进指令调度和减少内存流量的潜力，是重要的性能驱动因素。考虑一个简单的序列，例如 `output = activation(conv(input) + bias)`。即时执行可能涉及3-4次独立的内核启动和框架分派。JIT编译器可以将其融合为一个单一内核，大幅减少开销并可能改善局部性。

### 分摊编译成本

JIT编译引入了初始开销：在运行时编译代码所需的时间。在优化代码执行前必须支付此成本。然而，对于重复执行的模型或函数（例如，在推理 (inference)循环内、迭代训练步骤中或在长期运行的服务器进程中），这种初始编译成本可以分摊到许多后续的快速执行中。



![执行时间：即时执行对比JIT](plots/4560-0.json)



> JIT编译会产生初始成本，但实现了更低的单次推理延迟，在一定次数的运行后比即时执行更高效。

正如可视化所示，虽然即时执行的启动时间可以忽略不计，但其单次推理成本较高。JIT编译有非零的编译启动成本，但由于所应用的优化，其后续的单次推理执行时间显著降低。JIT何时变得有优势取决于模型的复杂度、JIT优化的有效性以及编译代码被重用的次数。

### 自适应优化潜力

高级JIT系统可以纳入配置文件引导优化（PGO）或采用多层编译。它们可能最初执行快速、基本的编译，然后使用运行时分析信息（例如，频繁执行的代码路径、观察到的数据分布）在后台触发更积极的重新优化。这种适应性使得系统能够根据实际使用模式微调 (fine-tuning)性能，这相对于静态AOT方法是一个优势。

总之，JIT编译通过运用运行时信息，为优化ML工作负载提供了一个强大的机制。它允许基于动态张量形状和值的专业化，通过算子融合减少框架开销，并提供了一条通向自适应优化的途径。虽然它引入了编译延迟，但这种成本通常被重复执行期间的性能提升所抵消，使其成为许多ML部署场景的一个有吸引力的选择，特别是那些涉及可变输入或需要在灵活性和速度之间取得平衡的场景。我们将在以下部分研究用于实现这些好处的具体技术，例如跟踪和脚本化。

## 参考资料

- [TorchScript Reference](https://pytorch.org/docs/stable/jit.html) — PyTorch Contributors (2018)
  Publisher: PyTorch
  官方文档，解释了PyTorch的JIT编译机制TorchScript，直接解决了本节讨论的许多动机，如性能优化和部署。
- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3342333.3342371](https://doi.org/10.5555/3342333.3342371)
  这篇开创性论文介绍了TVM，一个深度学习编译器框架，强调了JIT等优化技术在处理动态ML工作负载和异构硬件方面的必要性。
- [XLA: Optimizing TensorFlow computations](https://www.tensorflow.org/xla) — The TensorFlow Authors (2024)
  Publisher: Google
  解释了XLA（TensorFlow的JIT编译器）如何通过执行图级优化（包括算子融合和动态形状处理）来加速机器学习模型。
