# LLM的编译器优化

来源：[原文](https://apxml.com/zh/courses/llm-compression-acceleration/chapter-6-hardware-acceleration-systems-optimization/compiler-optimizations-llms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然优化后的计算核在微观层面发挥硬件能力，并且并行等技术可以在不同设备上适用，但编译器在中间层级运行，将LLM的高级计算图转换为可在目标硬件上高效执行的低级代码。编译器优化对于实现硬件和模型优化技术（例如量化 (quantization)和剪枝）的全部潜力不可或缺。

专为深度学习 (deep learning)设计的编译器，例如 Apache TVM、谷歌的 XLA（加速线性代数），或集成在 TensorRT 和 ONNX Runtime 等运行时的编译器，在生成可执行代码之前，会对LLM的计算图执行一系列分析和转换。这些优化旨在减少冗余计算，最大限度地减少内存访问开销，并最大化硬件单元的利用率。

### LLM的编译器优化方法

有几种编译器技术在加速LLM推理 (inference)方面特别有效：

1. **算子合并：** LLM计算通常涉及一系列逐元素操作，或一个操作的输出立即作为下一个操作输入的序列（例如，矩阵乘法后跟偏置 (bias)相加和激活函数 (activation function)）。算子合并将多个此类操作组合成一个更大的计算核。

   - **优点：**
     - **减少内存流量：** 合并操作之间的中间结果保存在更快的局部内存（寄存器或缓存）中，而不是写回并从较慢的主内存（如GPU显存 (VRAM)）中读取。这对于通常受内存带宽限制的LLM来说极其重要。
     - **降低计算核启动开销：** 启动许多小型计算核会产生开销。单个合并的计算核减少了这种开销，提高了整体GPU利用率。
     - **提高指令级并行性：** 更大、合并后的计算核为编译器和硬件调度器提供了更多机会，以找到可以并行执行的独立指令。

   考虑Transformer模块中的典型序列：

   > 算子合并将矩阵乘法、偏置相加和ReLU激活等顺序操作组合成单个计算核，从而减少内存传输和启动开销。
2. **常量折叠：** 在编译过程中，编译器识别计算图中仅依赖于常量输入（如模型权重 (weight)或推理期间不更改的配置参数 (parameter)）的部分。它离线预计算这些部分，有效地将计算“折叠”成常量张量。

   - **优点：** 减少运行时所需的计算量。虽然许多LLM权重是常量，但激活的动态特性限制了其适用范围（相比其他模型），但它仍然可以简化图的一部分（例如，如果位置编码 (positional encoding)是静态的，则预计算其一部分；如果偏置为零，则简化偏置相加）。
3. **布局优化：** 张量（多维数组）在内存中的存储方式显著影响性能，特别是在GPU等硬件上，它们倾向于特定的数据布局以实现内存合并和向量 (vector)化操作。例如，用于矩阵乘法的权重可能以行主序或列主序存储。输入激活可能以NCHW（批次，通道，高度，宽度）或NHWC等格式处理。编译器可以自动转换张量布局，以匹配硬件或特定优化计算核（如cuDNN或Tensor Core计算核）所期望的最优格式。

   - **优点：**
     - **改进内存合并：** 访问连续内存块速度快得多。优化的布局确保硬件warp中的线程访问相邻数据点。
     - **高效向量化 (quantization)：** 对齐 (alignment)数据以便由SIMD（单指令多数据）单元或专用硬件（如NVIDIA的Tensor Cores）处理，这些硬件在特定矩阵块形状和布局上高效运行。
     - **减少转置开销：** 通过全局选择正确的布局，编译器可以最大限度地减少运行时显式、高成本数据转置操作的需求。
4. **代数简化：** 编译器可以应用数学恒等式来简化表达式。例如，乘以1或加上0可以被消除。虽然这看起来微不足道，但这些简化可能来自图构建或其他优化过程，移除它们可以清理图。
5. **静态内存规划：** 编译器分析整个计算图和每个张量（中间激活）的生命周期。然后，它可以预先规划内存分配，一旦张量不再需要，就重用内存缓冲区。

   - **优点：** 减少运行时动态内存分配开销，并最大限度地减少执行模型所需的峰值内存占用。这对于在内存受限设备上部署大型模型尤其重要。

### 与量化 (quantization)和稀疏性的相互作用

当处理量化或剪枝模型时，编译器优化变得更加重要。

- **量化：** 编译器需要理解低精度数据类型（例如INT8、INT4、FP8），并定位能高效处理它们的硬件指令。算子合并可能会将量化/反量化步骤与计算操作结合起来，以最大程度地减少精度转换。
- **稀疏性：** 对于剪枝模型，特别是那些具有结构化稀疏性的模型，编译器可以运用稀疏模式的知识，生成跳过涉及零值权重 (weight)或激活的计算的代码。这需要编译器支持稀疏矩阵格式和操作，这是一个活跃的开发方向。

### 编译器框架和工具链

现代深度学习 (deep learning)部署通常依赖于复杂的编译器框架：

- **MLIR（多级中间表示）：** 一个核心的编译器基础设施项目（LLVM的一部分），旨在表示和优化机器学习 (machine learning)模型。许多特定领域的编译器都建立在MLIR之上。
- **Apache TVM：** 一个开源编译器栈，它接受来自PyTorch、TensorFlow和ONNX等框架的模型，并为各种硬件后端（CPU、GPU、专用加速器）生成优化代码。它采用自动调优（AutoTVM、AutoScheduler）等高级技术，以找到最优的计算核实现和合并策略。
- **XLA（加速线性代数）：** 谷歌为TensorFlow和JAX设计的编译器，面向CPU、GPU和TPU。它执行广泛的图优化，包括合并和布局转换。
- **供应商专用编译器（例如TensorRT）：** 硬件供应商通常提供与其硬件功能紧密集成的自有编译器（如NVIDIA的TensorRT）。这些编译器可以自动应用许多优化，包括选择优化计算核、执行合并，以及处理量化 (quantization)的精度校准。

采用这些编译器技术是部署高性能LLM工作流程中的标准组成部分。通过将高级图转换为优化且硬件特定的代码，编译器弥合了模型定义和高效执行之间的差距，相比单独的模型压缩，显著降低了延迟并提高了吞吐量 (throughput)。

## 参考资料

- [TVM: An End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578--594; DOI: [10.5555/3291244.3291295](https://doi.org/10.5555/3291244.3291295)
  介绍了Apache TVM，一个开源深度学习编译器栈，它自动优化并为多种硬件后端生成代码，涵盖了许多文中讨论的优化技术。
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://ieeexplore.ieee.org/document/9369931) — Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Nick Deshler, John Field, Junjie Gui, River Huang, Roman Popov, River Riddle, Stella Laurenzo, Siddhartha Phadke, Nicolas Quesnel, George R. St. Amant, Sigal Sahar, Vincent To, Lengning Liu, and Edwin Choi (2021)
  Journal: 2021 IEEE/ACM International Symposium on Code Generation and Optimization (CGO); Publisher: IEEE; Pages: 247-260; DOI: [10.1109/CGO51591.2021.9369931](https://doi.org/10.1109/CGO51591.2021.9369931)
  描述了MLIR，一个灵活的编译器基础设施，作为构建机器学习领域专用编译器的基础，包括表示和转换。
- [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) — NVIDIA Corporation (2024)
  Publisher: NVIDIA Corporation
  官方文档，提供了使用TensorRT优化和部署深度学习模型的实践细节，包括图优化和硬件定向的具体信息。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — TensorFlow team (2024)
  Publisher: Google
  官方文档，介绍了XLA，谷歌为TensorFlow和JAX开发的优化编译器，它为CPU、GPU和TPU提供图优化。

---

[上一节](03-LLM%E5%B1%82%E7%9A%84%E4%BC%98%E5%8C%96%E7%AE%97%E5%AD%90.md) · [下一节](05-%E5%88%86%E5%B8%83%E5%BC%8F%E6%8E%A8%E7%90%86%E7%AD%96%E7%95%A5.md)
