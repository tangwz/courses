---
course: "compiler-runtime-optimization-ml"
chapter: "jit-compilation-ml"
lesson: "jit-runtime-specialization"
sourceId: 4566
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-7-jit-compilation-ml/jit-runtime-specialization"
title: "运行时专门化与多态性"
description: "Discuss how JIT compilers can specialize code based on runtime information like tensor shapes or values."
order: 4
plots: []
sourceHash: "9c0e5dd42d1494fc53b57503ad533a522d7d459bf4577de58b039159a7ecf682"
sourceCorrections: []
---

进行运行时专门化是即时 (JIT) 编译在机器学习 (machine learning)环境中的一个主要优点。与预先 (AOT) 编译不同，预先编译必须生成能够处理可能未知输入特性的代码，而JIT编译器则在执行期间使用可用的具体信息。这使得能够生成针对给定调用的特定情况进行高度优化的代码。

### 形状专门化

ML JIT中最常见的运行时专门化形式是形状专门化。许多机器学习 (machine learning)模型，尤其是在推理 (inference)期间，可能会处理具有不同维度的输入，例如在线服务中的动态批次大小或自然语言处理中的可变序列长度。

考虑一个常见的张量操作，比如2D卷积。AOT编译器可能会生成包含由符号形状变量（例如，$N, C, H, W$）确定的循环边界的通用代码。这通常需要运行时检查或效率较低的循环结构。

JIT编译器观察到具体的输入张量形状，例如，$32 \times 3 \times 224 \times 224$，可以直接使用这些信息：

1. **固定循环边界：** 遍历维度的循环可以将其边界硬编码为常量（例如，`for i = 0 to 223`）。这有助于底层编译器后端（如LLVM）进行更好的指令调度、循环展开和预取。
2. **优化内存访问：** 了解确切的维度有助于生成针对内存中张量数据的特定步长和连续性进行优化的内存访问模式，提高缓存局部性。
3. **优化内核选择：** JIT可以选择或生成专门针对该输入形状进行调整的内核实现，可能使用与通用情况不同的算法或平铺策略。
4. **向量 (vector)化：** 常量维度有助于更有效的自动向量化 (quantization)，因为编译器可以确定确切的迭代次数和对齐 (alignment)属性，从而有效使用SIMD指令（AVX、NEON等）。

### 类型和值专门化

除了形状，JIT还可以根据数据类型进行专门化。如果模型图是通用定义的，但在运行时总是接收`float32`张量，JIT可以专门为`float32`操作编译代码，从而避免与动态类型处理相关的开销。同样，如果识别到`bfloat16`或`int8`等低精度类型，JIT可以生成使用专用硬件指令的代码（如果存在，例如NVIDIA GPU上的Tensor Core指令，或其他加速器上的矩阵乘法单元）。

值专门化不太常见但可能发生。如果子图的某些输入张量总是持有特定的常量值（例如，作为张量传递的配置标志），JIT可能会传播这些常量并相应地简化计算。

### 处理多态性：守卫和代码版本控制

使得专门化成为可能的动态特性也带来了复杂性。当运行时信息在不同调用之间发生变化时会怎样？例如，如果批次大小从32变为64？为批次大小32专门化的代码不再有效或不再最优。这就是多态性管理发挥作用的地方。

JIT系统采用方法来处理运行时上下文 (context)中的变化：

1. **守卫：** 当生成专门化的代码时，JIT会在编译函数的入口点插入运行时检查，即“守卫”。这些守卫检查当前输入特性（例如形状、类型）是否与代码专门化时的假设一致。

   - **守卫示例 (伪代码)：**

     ```python
     def compiled_function(input_tensor):
       # 守卫：检查形状是否与专门化版本匹配
       if input_tensor.shape != (32, 3, 224, 224):
         # 不匹配：回退或触发重新编译
         return fallback_or_recompile(input_tensor)
       else:
         # 匹配：执行针对 (32, 3, 224, 224) 专门化的高度优化代码
         return execute_specialized_code_32_3_224_224(input_tensor)
     ```
2. **代码版本控制和缓存：** JIT不会在每次不匹配时都重新编译，而是通常会维护一个专门化代码版本的缓存。每个版本都与其编译时对应的运行时属性（如形状元组）相关联。当JIT遇到一组新的属性时，它会首先检查缓存。

   - 如果存在匹配的版本，会检查其守卫，然后执行缓存的代码。
   - 如果不存在匹配的版本，JIT会编译一个新的专门化版本，将其添加到缓存中并执行。

> JIT编译器通过代码版本控制来处理多态性的流程。传入的调用会与缓存的专门化版本进行检查；缓存未命中会导致为新形状重新编译。

### 权衡与考量

运行时专门化可以带来显著的性能提升，但伴随着一些权衡：

- **编译开销：** 每次新的专门化都需要编译时间。形状频繁变化可能导致明显的暂停或高初始延迟（“预热时间”），因为JIT需要编译必要的版本。自适应编译等方法（稍后讨论）旨在减轻这一点。
- **内存消耗：** 缓存多个专门化代码版本会占用内存。JIT运行时常使用缓存淘汰策略（例如，最近最少使用 - LRU）来管理缓存大小，可能丢弃不常用专门化，这些专门化可能之后需要重新编译。
- **守卫开销：** 执行守卫会增加少量的运行时成本。尽管与专门化带来的提升相比通常可以忽略不计，但设计不当或数量过多的守卫会影响性能，特别是对于非常小的计算。
- **复杂性：** 实现专门化和多态性处理会大幅增加JIT编译器和运行时系统的复杂性。

高效的JIT系统会平衡这些因素，通常借助启发式方法或分析信息（配置文件引导优化 - PGO）来决定何时专门化可能带来超过开销的益处，以及哪些专门化值得缓存。运行时专门化与智能多态性管理相结合，是JIT编译器在动态机器学习 (machine learning)执行环境中实现高性能的一个显著特点。

## 参考资料

- [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI '18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3305381.3305421](https://doi.org/10.5555/3305381.3305421)
  介绍了TVM，一个开源深度学习编译器，它自动化张量程序优化，包括依赖于形状的内核生成和针对硬件的向量化。
