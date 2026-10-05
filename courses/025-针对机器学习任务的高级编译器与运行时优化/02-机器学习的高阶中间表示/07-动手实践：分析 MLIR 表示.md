# 动手实践：分析 MLIR 表示

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-2-advanced-ml-intermediate-representations/practice-analyzing-mlir)

[返回章节目录](README.md) · [返回课程目录](../README.md)

阅读和理解 MLIR（多层中间表示）是任何在编译器层面优化机器学习 (machine learning)工作负载的人所需的基本能力。MLIR 的结构及其方言和运算，为表示不同抽象层级的计算提供了一个框架。对常见 MLIR 模式的分析提供了对其结构和功能的实践性理解。

我们假设您拥有可以显示或生成 MLIR 的工具。许多机器学习编译器框架（例如启用了 XLA 的 TensorFlow、IREE，或直接使用 LLVM/MLIR 的项目）可以在不同阶段导出其 MLIR 表示形式。

### 示例 1：逐元素相加

我们从一个基本的张量运算开始：将两个浮点数的二维张量相加。在像 `linalg` 这样的方言中（常用于张量上的线性代数运算），这可能看起来像这样：

```mlir
#map0 = affine_map<(d0, d1) -> (d0, d1)>

module {
  func.func @add_tensors(%arg0: tensor<128x256xf32>, %arg1: tensor<128x256xf32>) -> tensor<128x256xf32> {
    // 为结果分配内存 (简化表示)
    %init_result = linalg.init_tensor [128, 256] : tensor<128x256xf32>

    // 使用 linalg.generic 执行逐元素相加
    %result = linalg.generic {
      indexing_maps = [#map0, #map0, #map0], // 输入 A, 输入 B, 输出
      iterator_types = ["parallel", "parallel"] // 两个维度都是并行的
    }
    ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<128x256xf32>)
    outs(%init_result : tensor<128x256xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out_init: f32): // 内部计算的基本块
      %sum = arith.addf %in0, %in1 : f32
      linalg.yield %sum : f32 // 为此元素返回结果
    } -> tensor<128x256xf32>

    return %result : tensor<128x256xf32>
  }
}
```

**分析要点：**

- **方言：** 这里的主要运算是 `linalg.generic`。`linalg` 方言旨在处理张量上的结构化运算，通常作为降级为循环或特定硬件内部函数之前的中间层抽象。内部计算使用 `arith` 方言中的 `arith.addf` 运算。
- **类型：** 请注意显式张量类型：`tensor<128x256xf32>`。这立刻告知我们形状（128x256）和元素类型（32 位浮点数）。此静态形状信息对优化很重要。
- **运算：** `func.func` 定义一个函数。`linalg.init_tensor` 分配空间（实际上，内存管理更为复杂）。`linalg.generic` 是一个强大的运算，定义对张量元素进行的计算。其主体（`^bb0`）指定对每个元素执行的标量计算。`arith.addf` 执行浮点加法。`linalg.yield` 返回为此元素计算的值。
- **结构：** 运算以 SSA（静态单赋值）值作为操作数（例如 `%arg0`、`%arg1`），并产生 SSA 值作为结果（例如 `%result`）。仿射映射（`#map0`）定义循环迭代器如何映射到张量索引，这对于理解数据访问模式很重要。`iterator_types` 指定循环的性质（此处，“并行”表示这些维度上没有跨迭代依赖）。

### 示例 2：矩阵乘法

矩阵乘法是许多机器学习 (machine learning)模型的运算基础。一个高级表示，可能在领域特定方言中或使用 `linalg.matmul`，可能看起来像：

```mlir
module {
  func.func @matmul(%A: tensor<32x64xf32>, %B: tensor<64x128xf32>) -> tensor<32x128xf32> {
    %C = linalg.matmul ins(%A, %B : tensor<32x64xf32>, tensor<64x128xf32>)
                       outs(/* init tensor omitted */) -> tensor<32x128xf32>
    // 假设 %C 在矩阵乘法之前已正确初始化
    return %C : tensor<32x128xf32>
  }
}
```

如果我们查看此代码*在*降级为 `affine` 和 `scf`（结构化控制流）等方言之后，为 CPU 执行做准备，我们可能会看到这样的代码（高度简化）：

```mlir
#map_A = affine_map<(d0, d1, d2) -> (d0, d2)> // 访问 A[i, k]
#map_B = affine_map<(d0, d1, d2) -> (d2, d1)> // 访问 B[k, j]
#map_C = affine_map<(d0, d1, d2) -> (d0, d1)> // 访问 C[i, j]

module {
  func.func @matmul_lowered(%argA: memref<32x64xf32>, %argB: memref<64x128xf32>, %argC: memref<32x128xf32>) {
    // 结果维度 (i, j) 的外层循环
    scf.for %i = 0 to 32 step 1 {
      scf.for %j = 0 to 128 step 1 {
        // 内层归约循环 (k)
        %init_acc = arith.constant 0.0 : f32
        %acc = scf.for %k = 0 to 64 step 1 iter_args(%iter_acc = %init_acc) -> (f32) {
          // 从输入张量加载元素
          %a_val = affine.load %argA[%i, %k] : memref<32x64xf32> // 简化访问
          %b_val = affine.load %argB[%k, %j] : memref<64x128xf32> // 简化访问

          // 计算乘积并累加
          %prod = arith.mulf %a_val, %b_val : f32
          %new_acc = arith.addf %iter_acc, %prod : f32
          scf.yield %new_acc : f32
        }
        // 存储最终累加值
        affine.store %acc, %argC[%i, %j] : memref<32x128xf32> // 简化访问
      }
    }
    return
  }
}
```

**分析要点（降级示例）：**

- **方言转换：** `linalg.matmul` 运算消失了。我们现在看到用于循环的 `scf.for`、用于内存访问的 `affine.load`/`affine.store`（使用 `memref` 类型而非 `tensor`），以及用于核心计算的 `arith` 运算。这代表一个 *降级* 步骤。
- **抽象层级：** 这更接近传统命令式代码。循环结构、内存访问和算术运算都是显式的。此层级适用于平铺、向量 (vector)化以及最终生成机器指令等转换。
- **类型：** 请注意使用 `memref`（内存引用）而非 `tensor`。`memref` 通常意味着已分配的、具有特定内存布局的内存，而 `tensor` 则更抽象。仿射映射（尽管在此处已简化）对于分析内存访问模式以进行缓存优化变得非常重要。

### 分析指南

在检查不同工具或不同编译阶段生成的 MLIR 代码时，请考虑以下问题：

1. **存在哪些方言？** 这会告知您当前的抽象层级（例如，`mhlo` 用于高级图运算，`linalg` 用于结构化张量运算，`vector` 用于 SIMD，`affine`/`scf` 用于循环/内存，`gpu` 用于 GPU 特有内容，`llvm` 用于最终降级）。
2. **主要运算是什么？** 寻找表示计算、控制流或类型转换的运算。理解它们的操作数和结果。
3. **张量形状和类型如何表示？** 形状是静态的还是动态的？元素类型是什么？这严重影响优化选择。
4. **数据如何流动？** 追踪从生产者到消费者的 SSA 值。
5. **能否发现潜在的高级优化？** 寻找可以融合的运算序列（`linalg.generic` 通常是融合的结果）。
6. **如果已降级，结构与原始运算有何关联？** 能否看出 `matmul` 如何变为嵌套循环？这可能如何映射到硬件（例如，并行循环到 GPU 线程/块）？
7. **是否存在定义运算行为的属性？** 寻找运算上的属性，这些属性指定了步长、填充、数据布局或精度细节。

这种动手分析非常重要。通过阅读和理解 MLIR，您将获得对机器学习 (machine learning)编译器如何组织计算并应用优化的认识。它使您能够理解不同编译策略的影响，并找出可以提高性能的方面，摆脱机器学习框架的黑盒 (black box)视图。

## 参考资料

- [MLIR - Multi-Level Intermediate Representation](https://mlir.llvm.org/docs/) — The LLVM Project (2024)
  提供了MLIR核心架构、方言、操作、类型（张量、内存引用）等概念以及整体设计的全面信息。
- [Deep Learning Compilers](https://mitpress.mit.edu/books/deep-learning-compilers) — Tianqi Chen, Jonathan Chen, and Mu Li (2022)
  Publisher: MIT Press
  全面概述了深度学习的编译器技术，为MLIR如何融入优化机器学习工作负载的生态系统提供了背景。

---

[上一节](06-%E5%8F%AF%E6%89%A9%E5%B1%95%E6%80%A7%E4%B8%8E%E8%87%AA%E5%AE%9A%E4%B9%89%E6%96%B9%E8%A8%80.md) · [下一节](../03-%E9%AB%98%E7%BA%A7%E5%9B%BE%E7%BA%A7%E5%88%AB%E4%BC%98%E5%8C%96/01-%E5%9B%BE%E9%87%8D%E5%86%99%E7%B3%BB%E7%BB%9F.md)
