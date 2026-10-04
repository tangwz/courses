---
course: "compiler-runtime-optimization-ml"
chapter: "advanced-graph-level-optimizations"
lesson: "graph-rewriting-systems"
sourceId: 4470
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-3-advanced-graph-level-optimizations/graph-rewriting-systems"
title: "图重写系统"
description: "理解用于模式匹配和重写计算图的机制。"
order: 1
plots: []
sourceHash: "f8494752339de6c3ca5391602697fa9180de4dc79809a8be4dd3e30bd82a489e"
sourceCorrections: []
---

如前所述，优化计算图涉及改变其结构以提升效率。这些变换，比如算子融合或代数简化，并非随意应用。相反，它们依赖于被称为**图重写系统**的系统化机制。这些系统提供了一种基本框架，用于识别图中的特定模式，并应用预定义的变换，以更优化的结构替换这些模式。

图重写系统作用于计算图，计算图通常表示为有向无环图（DAG）。在此DAG中，节点代表运算（如卷积、矩阵乘法或激活），边代表数据依赖关系，通常是运算之间流动的数据。

### 图重写系统的组成部分

图重写系统主要由两部分组成：

1. **一套重写规则：** 每条规则定义一个特定的变换。它通常包含两部分：

   - **模式（或左侧 - LHS）：** 描述要搜索的子图结构。此模式指定运算类型、它们的连接关系（依赖关系），以及节点属性（如数据类型或张量形状）可能存在的限制。
   - **替换（或右侧 - RHS）：** 定义应替换匹配模式的新子图结构。这可能涉及创建新节点、删除现有节点，以及调整边（数据依赖关系）以将替换子图正确连接到主图中。
2. **一个引擎：** 该组件组织重写过程。其职责包括：

   - **模式匹配：** 高效地在整个计算图中搜索重写规则中定义的所有模式出现。
   - **规则应用：** 选择要应用的匹配规则，并按照规则的替换模式执行图修改。
   - **策略：** 确定应用规则的顺序和迭代方法。应用一条规则可能会启用或禁用其他规则，因此策略对最终优化后的图影响很大。

### 模式匹配

模式匹配是在较大计算图中寻找与重写规则中定义的模式同构（结构等效）的子图的过程。此过程需要考虑：

- **节点类型：** 匹配特定的运算（例如，`Conv2D`、`ReLU`、`Add`）。
- **连接性：** 确保子图中节点之间的数据依赖关系与模式指定的边相符。
- **属性和限制：** 验证节点属性（如步幅、填充、数据类型，甚至常量值）是否满足模式中指定的任何限制。例如，融合模式可能只在数据类型匹配或操作数是常量时才适用。

匹配可以是简单的相邻节点检查，也可以是复杂的子图同构问题。相关技术常借鉴项重写和编译器优化，有时会使用专门的模式描述语言（如MLIR的模式描述语言 - PDL）来正式定义复杂的模式及其相关限制。

### 替换与变换

一旦模式匹配成功，重写引擎就会应用规则替换部分定义的变换。这涉及对图数据结构的细致操作：

1. **节点创建：** 实例化替换模式中指定的新节点。
2. **节点删除：** 移除属于匹配模式但不属于替换部分的节点。
3. **边重连：** 边（数据依赖关系）得到更新。原始匹配子图的输入可能会重新路由到替换子图的输入。替换子图的输出连接到原始匹配子图输出的消费者。

此变换必须保持计算的语义等效性（或进行改进，例如通过移除冗余运算），以及图结构的整体有效性。

### 应用策略

由于多条规则可能同时匹配图的不同部分，或者应用一条规则可能为其他规则创造机会，因此应用规则的策略很重要。常见的策略包括：

- **迭代应用：** 重复扫描图并应用规则，直到没有更多规则可以应用（达到不动点）。
- **基于优先级的应用：** 为规则分配优先级，确保可能更具影响或作为前提的变换优先应用。
- **成本建模：** 使用成本模型来评估应用规则的性能影响（例如，缩短执行时间、减少内存使用），从而指导选择过程，倾向于能带来最大效益的变换。

贪心应用（应用第一个匹配的规则）常用于简化，但不能保证找到最优的图配置，这通常是一个NP难问题。

### 示例：代数简化

考虑一个简单的代数恒等式：`Transpose(Transpose(A)) = A`。一条重写规则可以体现这一点：

- **模式：** 一个 `Transpose` 运算，其输入是另一个 `Transpose` 运算的输出。
- **替换：** 将外部的 `Transpose` 运算替换为从内部 `Transpose` 的输入直接连接到外部 `Transpose` 的结构。实际上，两个 `Transpose` 节点都被绕过，如果没有其他使用者，它们可能会被移除。

> 该重写规则识别出两个连续的转置操作，并将其替换为从原始输入到最终输出的直接连接。

### 示例：简单融合

另一种常见模式是将一个运算与其后续的激活函数 (activation function)融合，例如 `Convolution -> ReLU`。

- **模式：** 一个 `ReLU` 运算，其输入是 `Conv` 运算的输出。通常包含限制条件，例如 `Conv` 的输出仅被此 `ReLU` 使用。
- **替换：** 一个单一的 `FusedConvReLU` 运算，它接收原始 `Conv` 的输入，并产生等同于原始 `ReLU` 的输出。原始的 `Conv` 和 `ReLU` 节点将被移除。

> 将卷积和ReLU运算融合为单一的FusedConvReLU节点可减少内核启动开销并改善数据局部性。

图重写系统为后续章节讨论的高级图级优化提供了核心机制。通过定义模式及其对应的替换，编译器可以系统地重构ML计算图以提升性能，然后再进行低级代码生成。理解这些系统对掌握高级优化（如算子融合、代数简化和布局变换）如何在实践中实现非常重要。

## 参考资料

- [Graph Transformation: Basic Concepts and Applications](https://link.springer.com/book/10.1007/3-540-27202-0) — Hartmut Ehrig, Gregor Engels, Hans-Jörg Kreowski, Grzegorz Rozenberg (Editors) (2007)
  Publisher: Springer Berlin Heidelberg; DOI: [10.1007/3-540-27202-0](https://doi.org/10.1007/3-540-27202-0)
  关于图转换和重写系统的基础教材，提供了全面的理论概念。
- [TVM: An Automatic End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594
  介绍了TVM，一个深度学习编译器，通过灵活的重写系统实现图级优化，包括融合和其他转换。
- [MLIR's Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) — The MLIR Team (Current)
  MLIR框架中模式匹配和重写机制的官方文档，包括模式描述语言（PDL）的使用。
