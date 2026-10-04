---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-graph-level-optimizations"
lesson: "graph-control-flow"
sourceId: 4484
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/graph-control-flow"
title: "图中的控制流处理"
description: "讨论优化以图形式表示的条件执行和循环的策略。"
order: 6
plots: []
sourceHash: "4a3bba944a07b2eeb66176ba4bb32299d5fde115d6986159046eb2387a58df28"
sourceCorrections: []
---

虽然许多机器学习 (machine learning)模型可以表示为静态有向无环图（DAG），但应用程序常需要通过条件执行（if/else）和循环（while/for）来表达动态行为。在图表示中妥善处理这种控制流对ML编译器来说是一项主要难题，它会影响执行其他优化的能力以及最终的运行时性能。

与静态图中直接的数据依赖不同，控制流引入了关于哪些操作会执行以及条件依赖的不确定性。优化这些结构需要专门方法，以理解控制流操作的语义及其与数据流的协同作用。

### 在ML图中表示控制流

在优化控制流之前，我们需要一种在图中表示它的方式。常见的方法包括：

1. **函数式控制流操作符：** 高级框架常提供`If`、`Cond`、`WhileLoop`或`Scan`等特定操作符。这些操作符封装了表示条件分支或循环体的整个子图。TensorFlow的`tf.cond`和`tf.while_loop`，或JAX的`lax.cond`和`lax.scan`就是例子。编译器将它们视为特殊节点，根据输入条件或循环状态管理其所包含子图的执行。
2. **显式控制边：** 较低层级的图表示，特别是那些更接近传统编译器IR的表示，可能在数据边旁使用显式控制边。像`Merge`、`Switch`和`Enter`/`Exit`（在TensorFlow的GraphDef中常见）这样的节点根据布尔条件指导执行和数据的流动。这种表示使控制流结构更明确，但管理起来可能复杂。
3. **基于区域的控制流（例如MLIR）：** 像MLIR这样的系统使用附带“区域”（操作块）的操作。控制流操作如`scf.if`（结构化控制流方言）或`cf.cond_br`（控制流方言）包含表示“then”和“else”分支或循环体的区域。这种结构化方法结合了函数式操作符的特点和显式块结构，有助于在明确的范围内进行分析和转换。

表示方式的选择常取决于抽象级别。高级图通常使用函数式操作符，这些操作符在编译过程中会逐步降级为更显式的形式。

### 图控制流的优化技术

优化带有控制流的图涉及调整标准编译器技术，并发展针对ML计算结构的新方法。

#### 谓词提升与下沉

与传统编译器中的代码移动类似，操作有时可以跨越控制流边界进行移动。

- **提升：** 如果条件分支或循环内部的操作不依赖于*仅*在该分支/循环中计算的任何值，并且无论选择哪条路径都需要执行它（或如果在控制流结构合并*之后*需要其结果），则可能可以将其“提升”到控制流结构之前。如果该操作出现在多个分支中，这可以减少重复计算。
- **下沉：** 相反，如果分支前的操作*仅*在特定分支内使用，则可以将其“下沉”到该分支中。如果未执行该分支，这可以避免不必要的运算。

依赖性分析在这里很重要。我们必须确保移动操作不会违反任何数据依赖关系并保持原始程序语义。

#### 分支简化与合并

当控制`If`操作的条件可以静态评估时（例如，它仅依赖于常量或静态形状信息），编译器可以执行分支消除。

- **常量条件：** 如果条件始终为真或始终为假，则整个`If`结构可以被对应于被选用分支的子图替换，而另一个分支可以完全剪除。
- **合并相同操作：** 如果“then”和“else”分支都在其开始或结束处立即执行相同的操作，则该操作可以移到条件结构外部（在条件检查之前或分支合并之后），从而简化这两个分支。

> 两个'then'和'else'分支中都存在的相同操作`OpA(x)`被提升到条件分割之前，从而简化了图结构。

#### 图层面的循环优化

标准循环优化在图转换中也有其对应：

- **循环展开：** 对于已知迭代次数较少的循环，编译器可以多次复制循环体子图，消除循环控制开销。这对于循环分支开销较高的硬件目标尤其有效。然而，它会增加代码大小。
- **循环不变代码外提：** 循环体子图内部的操作，如果其输入在迭代过程中不发生变化（即，它们仅依赖于循环外部的输入或常量），则可以移到循环外部，通常在循环开始之前。这可以避免每次迭代中的重复计算。识别循环不变式需要仔细分析流入和流出循环体子图的数据依赖关系。
- **循环剥离：** 有时循环的前几次或后几次迭代是特殊情况。剥离涉及将这些迭代从主循环体中提取出来，从而使主循环更简单，或者使得能够对剥离的迭代或剩余的循环进行进一步优化。

#### 与其他优化的关系

控制流使其他图遍历（如操作符融合）变得复杂。跨控制流边界融合操作通常很复杂，除非满足特定条件，否则通常不允许。例如，如果有利，可以将条件操作前的一个操作与*两个*分支中的操作融合，但仅在一个分支内融合操作需要仔细处理。

- **分支/循环内部融合：** 优化器常侧重于在条件分支或循环体定义的子图*内部*应用融合。
- **推测执行：** 在某些情况下，编译器可能会在条件完全确定之前，推测性地执行来自可能分支的代码，但这需要硬件支持和仔细管理，以便在推测错误时丢弃结果。与CPU指令调度相比，这在图级ML优化中不太常见。

妥善处理控制流对于优化具有动态行为的模型十分重要，例如循环神经网络 (neural network)（RNN）、处理变长序列的模型，或具有条件动作的强化学习 (reinforcement learning)智能体。此处讨论的技术使得编译器能够简化图结构、减少重复工作，并即使在存在条件逻辑和循环的情况下也能实现高效执行，为后续的张量级优化提供条件。

## 参考资料

- [Compilers: Principles, Techniques, and Tools](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFyzuTL11I_NDOJ8tPDgt2HH-3PXkTlW27CpYqYvp5fYAkgabbqIMTBQxN9-pNG9UHbAaD9f7qmTGgvFYnNbPKE_BX86YZ3Tek4O5xqnNwozvalwQjYulOyoAIyivKSa0SskrqQq33NsU-IwYw9rTkg9tLvUUIopfoI33RzLU=) — Alfred V. Aho, Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman (2006)
  Publisher: Pearson Education
  一本经典而全面的教科书，涵盖了基础的编译器优化，包括控制流分析、循环优化和代码移动技术等详细内容。
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://dl.acm.org/doi/10.1145/3473551) — Chris Lattner, Mehdi Amini, River Riddle, Albert Cohen, Alan Mycroft, Oleksandr Zinenko, Andy Davis, and Jacques Pienaar (2021)
  Journal: ACM Transactions on Architecture and Code Optimization (TACO); Publisher: Association for Computing Machinery (ACM); Volume: 18; Pages: 1-29; DOI: [10.1145/3473551](https://doi.org/10.1145/3473551)
  介绍了MLIR框架，解释了其通过区域和方言（如`scf`和`cf`）来表示和优化结构化控制流的设计。
- [Better performance with tf.function](https://www.tensorflow.org/guide/function) — TensorFlow Authors (2024)
  Publisher: TensorFlow Documentation
  官方指南，解释了如何在TensorFlow模型中使用`tf.function`，包括它在追踪过程中如何处理和优化`tf.cond`和`tf.while_loop`等动态控制流操作。
