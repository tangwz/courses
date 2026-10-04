---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-ml-intermediate-representations"
lesson: "mlir-dialects-operations"
sourceId: 4454
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations/mlir-dialects-operations"
title: "MLIR：方言和操作"
description: "了解 MLIR 的核心组成部分，包括其方言系统和操作定义。"
order: 3
plots: []
sourceHash: "0d3d0c7c8f8118be214daff155a2a4174b1af63f4fe1106961db707a1513245d"
sourceCorrections: []
---

传统编译器中间表示 (IR) 通常缺少足够的表达能力来表示机器学习 (machine learning)模型中的分层特性和特定领域的操作。MLIR（多级中间表示）通过提供一个通用、可扩展的基础设施来处理这个问题。MLIR 不提供固定的 IR，而是提供了一个定义和组合不同*方言*的框架，每种方言都针对特定的抽象级别或领域进行了定制。

### MLIR 的核心组成部分

MLIR 的强大之处源于其几个基本、定义明确的结构：

1. **操作（Ops）：** 语义和结构的主要单元。一个 Op 可以代表一个计算（如矩阵乘法）、一个结构元素（如函数定义）、一个常量值或一个元操作（如模块终止）。每个 Op 都属于一个特定的方言，并具有：

   - 一个唯一名称（例如，`arith.addi`、`func.func`、`linalg.matmul`）。
   - 零个或多个操作数（输入），它们是由其他 Op 或块参数 (parameter)生成的 SSA（静态单赋值）值。
   - 零个或多个结果（输出），它们是由 Op 定义的新 SSA 值。
   - 一个属性列表，提供静态信息（配置、常量值）。
   - 零个或多个区域，支持嵌套。
   - 一组定义其行为的属性（例如，可交换性、副作用）。
2. **属性：** 附加到 Ops 的编译时常量值。它们代表在执行期间不变的静态信息，例如字面常量、类型信息、维度置换或配置字符串。MLIR 具有内置属性（整数、浮点数、字符串、类型），方言也可以定义自定义属性。例如，一个卷积 Op 可能有步长 (`#stride<[1, 1]>`) 和填充 (`#padding<"SAME">`) 属性。
3. **类型：** 定义 SSA 值（Op 结果和块参数）的数据类型。MLIR 提供标准内置类型，如整数（`i32`、`i8`）、浮点数（`f32`、`f16`）、向量 (vector)（`vector<4xf32>`）和张量（`tensor<10x20xf32>`）。重要的是，类型系统可通过方言进行扩展，支持表示专用硬件类型或抽象的特定领域类型（如量化 (quantization)类型）。
4. **区域：** 包含在 Op 中的有序块列表。区域提供了嵌套和定义作用域的机制。例如，一个 `func.func` Op 包含一个代表函数体的区域，而像 `scf.for` 这样的循环 Op 则包含其循环体的区域。这种分层结构对 MLIR 的多级特性很重要。
5. **块：** 区域内 Ops 的序列，类似于传统编译器中的基本块。一个块接受一个参数列表（SSA 值），并包含一个 Ops 序列。块中的最后一个 Op 必须是一个*终结* Op（如 `func.return` 或 `cf.br`），它定义控制流，将执行转移到其他块或终止包含的区域。

> MLIR 函数 Op (`func.func`) 的高级结构，其中包含一个带有一个块的区域。该块接受一个参数，按顺序执行 Ops（使用 SSA 值），并以一个终结 Op 结束。

### 方言：特定领域命名空间

MLIR 的核心结构有意保持精简。其真正的强大之处和特有性源于*方言*。方言作为一组相关 Ops、属性和类型的命名空间。可以将其视为提供特定领域构造的库或模块。

MLIR 附带了几个标准方言：

- **`builtin`：** 定义了模块结构、基本类型（整数、浮点数）、属性（字符串、数组）和函数类型定义等基本内容。它构成了任何 MLIR 程序的基础。
- **`func`：** 定义了与函数相关的 Ops，如 `func.func`（函数定义）、`func.call` 和 `func.return`。
- **`arith`：** 提供标准的算术操作（加法、减法、比较等），主要作用于标量和向量 (vector)类型。
- **`vector`：** 定义了用于操作 `vector` 类型值的 Ops（例如，广播、混洗、收缩）。
- **`tensor`：** 定义了操作 `tensor` 类型值的 Ops（例如，`tensor.extract`、`tensor.insert`、`tensor.cast`）。
- **`linalg`：** 提供对张量和缓冲区的高级、结构化操作，通常以通用方式表示（例如，`linalg.generic`、`linalg.matmul`、`linalg.conv_2d_nhwc_hwcf`）。这种方言对于在生成详细循环之前抽象线性代数计算很重要。
- **`memref`：** 表示具有相关布局信息的内存缓冲区（`memref` 类型），并提供分配（`memref.alloc`）、释放和访问的 Ops。更接近于基于缓冲区的代码生成。
- **`scf`：** 结构化控制流方言，提供带有显式区域嵌套的循环（`scf.for`、`scf.parallel`）和条件（`scf.if`）Ops。
- **`cf`：** 控制流方言，提供传统的控制流基本块分支（`cf.br`、`cf.cond_br`），类似于 LLVM IR 的控制流。

这些编译器项目定义了自己的方言。例如，TensorFlow 可能会使用 `tf` 或 `mhlo` 方言来直接表示其图操作。面向特定加速器的编译器可能会定义一个自定义方言来表示硬件的独特指令。

### 示例：MLIR 中的简单函数

我们来看一个小的 MLIR 片段，它表示一个逐元素添加两个 32 位浮点张量的函数：

```mlir
// func.func 定义了一个名为 'add_tensors' 的函数。
// 它接受两个参数：%arg0 和 %arg1，它们的类型都是 tensor<10x20xf32>。
// 它返回一个相同类型的结果。
func.func @add_tensors(%arg0: tensor<10x20xf32>, %arg1: tensor<10x20xf32>) -> tensor<10x20xf32> {
  // 函数体是一个单独的块。

  // 'linalg.generic' Op 执行逐元素操作。
  // 'indexing_maps' 定义了输入/输出的每个维度如何与循环迭代器（仿射映射）关联。
  // 'iterator_types' 为维度 'd0' 和 'd1' 指定并行循环。
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, // 输入 0 映射
      affine_map<(d0, d1) -> (d0, d1)>, // 输入 1 映射
      affine_map<(d0, d1) -> (d0, d1)>  // 输出映射
    ],
    iterator_types = ["parallel", "parallel"] // 两个并行循环
  }
  // 'ins' 指定输入张量（%arg0, %arg1）。
  // 'outs' 指定输出张量缓冲区。此处根据操作数隐式分配。
  ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<10x20xf32>)
  outs(%arg0 : tensor<10x20xf32>) { // 注意：输出缓冲区形状推断通常使用输入形状。
  // 该区域定义了逐元素计算。
  ^bb0(%in0: f32, %in1: f32, %out_unused: f32): // 块参数对应于输入/输出中的元素
    // 'arith.addf' 对标量元素执行浮点加法。
    %sum = arith.addf %in0, %in1 : f32
    // 'linalg.yield' 返回输出张量计算出的元素。
    linalg.yield %sum : f32
  } -> tensor<10x20xf32> // Op 返回结果张量。

  // 'func.return' 终止函数，返回结果张量。
  func.return %result : tensor<10x20xf32>
}
```

在此示例中：

- `func.func`、`func.return` 来自 `func` 方言。
- `linalg.generic`、`linalg.yield` 来自 `linalg` 方言。
- `arith.addf` 来自 `arith` 方言。
- `tensor<10x20xf32>` 和 `f32` 是内置类型。
- `@add_tensors` 是一个符号名称（一个属性）。
- `%arg0`、`%arg1`、`%result`、`%in0`、`%in1`、`%sum` 是 SSA 值。
- `linalg.generic` 的 `{...}` 中的文本包含属性（`indexing_maps`、`iterator_types`）以及一个定义元素级计算的区域。

这种结构化表示结合了来自不同方言（`func`、`linalg`、`arith`）的 Ops，清晰地定义了函数的签名、其高级张量操作（`linalg.generic`）以及其嵌套区域中精确的逐元素计算。它作为各种优化的起点，最终会被降级为机器码。

精简的核心结构（Ops、区域、块、属性、类型）与可扩展方言系统的结合，使 MLIR 成为构建现代编译器的强大根基，尤其是在机器学习 (machine learning)等需要多级抽象并存且协作的复杂领域。我们将在后续章节中看到 MLIR 方言如何用于表示整个机器学习图，以及转换如何逐步将这些表示降级为可执行代码。

## 参考资料

- [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/abs/2002.11054) — Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Alan Mycroft, Andreas Herrmann, Andrew Gross, Arjun Singh, Benoit Steiner, Boris Babaian, Brian Retford, Carlo Contavalli, Chris Jones, Daniel J. M. Smith, Eric Chung, George A. Gravvanis, Giovanni Campagna, Hideki Saito, Jacques Pienaar, Jean-Philippe Martin, Jeffrey P. Snyder, Jeremy Cade, Johannes Doerfert, José Pablo Hernández-Cervantes, Junjie Bai, Kenneth O’Brien, Kiran Varaganti, Lang H. Tran, Laurent B. Le Cunff, Michael Zolotukhin, Nicolas Vasilache, Owen Anderson, Peng Li, Rafael R. Guimaraes, Remy Guillet, Stephen Neuendorffer, Steven J. Young, Tatiana Shpeisman, Thomas Van Durme, Uday Bondhugula, Vineet Singh, Wayne Wong, and Zachary DeVito (2021)
  Journal: Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS); Publisher: Association for Computing Machinery (ACM); Pages: 1-15; DOI: [10.1145/3446779.3446790](https://doi.org/10.1145/3446779.3446790)
  这篇开创性论文介绍了MLIR的设计原则，包括方言、层次化IR及其针对特定领域优化的可扩展性。
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) — The MLIR Project Developers (2024)
  关于MLIR核心组件（操作、类型、属性、区域、块）及方言结构的权威文档，对理解其语法和语义至关重要。
- [Understanding MLIR](https://mlir.llvm.org/docs/UnderstandingTheIRStructure/) — The MLIR Project Developers (2024)
  提供了MLIR设计的高级概述和概念性解释，包括方言在实现多层次抽象中的作用。
- [TensorFlow MLIR: A unified intermediate representation for machine learning infrastructure](https://ai.googleblog.com/2019/07/tensorflow-mlir-unified-intermediate.html) — Benoit Steiner, Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Andrew Gross, and Peng Li (2019)
  Journal: Google AI Blog; Publisher: Google
  讨论了MLIR的方言系统如何应用于统一TensorFlow的各种IR，展示了其在机器学习工作负载中的实用性。
