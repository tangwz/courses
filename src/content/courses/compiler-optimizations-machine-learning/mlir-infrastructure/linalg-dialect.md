---
course: "compiler-optimizations-machine-learning"
chapter: "mlir-infrastructure"
lesson: "linalg-dialect"
sourceId: 7941
sourceUrl: "https://apxml.com/zh/courses/compiler-optimizations-machine-learning/chapter-4-mlir-infrastructure/linalg-dialect"
title: "Linalg 方言"
description: "使用 Linalg 方言表示结构化的张量操作，适用于缓冲化和融合。"
order: 2
plots: ["plots/7941-0.json"]
sourceHash: "5d47661ca671ff835506a4deb75f74b47069bd7ceb91d0f331a04221db465365"
sourceCorrections: []
---

线性代数是深度学习 (deep learning)的数学核心，但在编译器中表示这些操作却出奇地难。传统的中间表示（IRs）经常过早地将矩阵操作降级为循环，从而掩盖了平铺和向量 (vector)化等高级优化所需的几何结构。MLIR `linalg` 方言通过将线性代数操作视为一等公民来解决此问题。它提供了一种结构化的抽象，将循环的控制流与实际的数学计算分离开来。

这种分离使得编译器能够在生成具体代码之前，从数学角度分析迭代空间和数据布局。本节中，我们将研究 `linalg.generic` 操作、定义数据访问的仿射映射系统，以及此架构如何实现高性能代码生成。

### 结构化操作的设计理念

`linalg` 方言的核心设计原则是“结构化操作”。与标准函数调用或循环嵌套不同，结构化操作明确声明其输入和输出要求、迭代空间以及循环维度的代数属性。

在标准的 LLVM IR 中，矩阵乘法只是一系列加载、存储和算术指令，嵌入 (embedding)在三个嵌套循环内。在 `linalg` 中，它是一个声明了以下内容的单一操作：

1. **输入和输出：** 哪些张量或内存缓冲区被读取和写入。
2. **索引映射：** 循环索引如何映射到数据的维度。
3. **迭代器类型：** 哪些循环可以并行化，哪些执行归约操作。

这种更高层次的抽象使得编译器能够执行激进的转换，例如将 ReLU 激活融合到矩阵乘法中，而无需进行复杂的依赖分析。编译器清楚地知道访问了哪些内存区域以及如何访问，从而通过构造确保转换的有效性。

> `linalg.generic` 操作的结构，着重显示了数据定义与计算载荷之间的分离。

### linalg.generic 操作

`linalg.generic` 操作是所有其他命名操作（如 `linalg.matmul` 或 `linalg.conv_2d`）派生而来的通用形式。理解 `generic` 十分重要，因为编译器在优化期间会将命名操作规范化为此形式。

`linalg.generic` 操作定义了循环嵌套上的计算。它需要特定的属性来指导编译器。

#### 索引映射

索引映射是仿射映射，定义了循环迭代索引如何转换为张量坐标。它们写成 $(d_0, d_1, \dots) \to (e_0, e_1, \dots)$ 的形式。

考虑一个矩阵乘法 $C_{ij} = \sum_k A_{ik} B_{kj}$。这里有三个迭代变量：$i, j, k$。

- $A$ 以 $A[i, k]$ 的形式访问。
- $B$ 以 $B[k, j]$ 的形式访问。
- $C$ 以 $C[i, j]$ 的形式访问。

在 MLIR 中，我们定义了三个仿射映射来表示相对于迭代域 $(i, j, k)$ 的这些访问：

1. A 的映射：$(i, j, k) \to (i, k)$
2. B 的映射：$(i, j, k) \to (k, j)$
3. C 的映射：$(i, j, k) \to (i, j)$

#### 迭代器类型

该方言区分了可以并行执行的维度和强制执行顺序（归约）的维度。对于矩阵乘法，$i$ 和 $j$ 是**并行**迭代器，因为 $C$ 的每个元素都可以独立计算。$k$ 维度是**归约**迭代器，因为它将值累加到相同的内存位置。

#### 载荷区域

操作主体（该区域）包含标量实现。它描述了单个元素会发生什么。对于矩阵乘法，这是一个乘法累加操作。

矩阵乘法在 `linalg.generic` 形式下如下所示：

```mlir
#map_a = affine_map<(i, j, k) -> (i, k)>
#map_b = affine_map<(i, j, k) -> (k, j)>
#map_c = affine_map<(i, j, k) -> (i, j)>

func.func @matmul_generic(%A: tensor<128x128xf32>, 
                          %B: tensor<128x128xf32>, 
                          %C_init: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %result = linalg.generic {
    indexing_maps = [#map_a, #map_b, #map_c],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<128x128xf32>, tensor<128x128xf32>)
    outs(%C_init : tensor<128x128xf32>) {
  ^bb0(%a_elem: f32, %b_elem: f32, %c_elem: f32):
    // 标量计算载荷
    %prod = arith.mulf %a_elem, %b_elem : f32
    %sum = arith.addf %c_elem, %prod : f32
    linalg.yield %sum : f32
  }
  return %result : tensor<128x128xf32>
}
```

在此片段中，`ins` 和 `outs` 定义了数据操作数。区域 `^bb0` 接收由仿射映射映射的标量元素。`linalg.yield` 获取计算值并更新输出张量。

### 平铺与融合策略

Linalg 的一个主要优点是，平铺作为对 IR 本身的转换来实现，而不仅仅是循环生成逻辑。对 `linalg` 操作进行平铺会生成一个循环嵌套（通常使用 `scf` 或结构化控制流方言），其中内部主体是一个表示该平铺的更小的 `linalg` 操作。

这种递归定义保留了循环嵌套各个级别的操作语义。如果平铺一个矩阵乘法，内部核心仍然是矩阵乘法，只是作用于数据的小视图。

#### 平铺交互的可视化

当我们平铺一个操作时，我们实际上引入了遍历数据块的新循环。`linalg` 架构会根据定义中提供的仿射映射，自动计算输入张量所需的子视图（切片）。



![平铺矩阵乘法的内存访问模式 (块 32x32)](plots/7941-0.json)



> 平铺创建了不同的访问块。热图显示了四个不同的平铺正在处理，说明了全局操作如何分解为局部结构化操作。

#### 操作融合

Linalg 在操作融合方面表现出色。因为 `generic` 操作暴露了逐元素访问模式，编译器可以将生产者操作（如 `linalg.add`）直接融合到消费者操作（如 `linalg.matmul`）中。

Linalg 中的融合通常通过“平铺消费者并融合生产者”来实现。编译器首先平铺消费者操作。然后，对于消费者的每个平铺，它只计算该平铺所需的生产者切片。这通过在值被使用前立即计算它们来改善缓存局部性，将数据保留在寄存器或 L1 缓存中。

### 缓冲化：从张量到内存引用

MLIR 允许操作在 `tensor` 类型（不可变值，用于数学推导）或 `memref` 类型（可变内存缓冲区，表示物理硬件内存）上工作。高级优化通常在张量上进行。然而，硬件在内存上执行指令。

将基于张量的 Linalg 操作转换为基于缓冲区的操作的过程称为**缓冲化**。

1. **一次性缓冲化：** 现代 MLIR 使用“一次性”分析来处理整个模块。它确定何处进行原地更新是安全的（为输出重用输入内存），以及何处需要复制以保留值语义。
2. **目标传递风格：** 张量上的 Linalg 操作通常接受一个“out”操作数（如上例中的 `%C_init`）。在张量空间中，这充当初始化值。在缓冲区空间中，这成为写入结果的内存缓冲区。

一旦操作被缓冲化，它可以被降级为循环（`scf.for`），并最终降级到 LLVM 方言以生成机器代码。Linalg 方言有效地充当了张量程序的抽象定义与指针算术的具体现实之间的桥梁。
