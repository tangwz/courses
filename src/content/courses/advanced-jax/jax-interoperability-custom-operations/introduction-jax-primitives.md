---
course: "advanced-jax"
chapter: "jax-interoperability-custom-operations"
lesson: "introduction-jax-primitives"
sourceId: 2909
sourceUrl: "https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations/introduction-jax-primitives"
title: "JAX 原语简介"
description: "理解JAX所知的基本运算——原语的理念。"
order: 5
plots: []
sourceHash: "f73c1d7082fe25c4978c6460c1f1692b016979ac687c3a82e89156650eddddc1"
sourceCorrections: []
---

为了有效扩展 JAX 或了解其内部运作，尤其是在与外部代码交互或优化性能时，我们需要查看熟悉的 NumPy 类似 API 的表面之下。JAX 功能的核心是**原语**的理念。

可以将原语视为 JAX 真正识别的基本、原子运算。当你使用 `jax.numpy.add` 或 `jax.numpy.dot` 等函数编写代码时，JAX 最终会（或*追踪*）将它们分解成一系列这些基本原语。例如，包括 `add`、`mul`、`dot_general`、`reduce_sum`、`transpose`、`conv_general_dilated` 等运算，以及 `scan_p` 或 `cond_p` 等控制流操作符。

### 原语的意义

原语因以下几个原因而有意义：

1. **转换目标：** `jit`、`grad` 和 `vmap` 等 JAX 转换不直接作用于 Python 代码。它们在原语层面上工作。

   - `jit`：将 Python 函数追踪为一系列原语，这些原语以称为 `jaxpr` 的中间格式表示。然后此 `jaxpr` 由 XLA 编译。
   - `grad`：依赖于为追踪过程中遇到的每个原语定义了特定的微分规则（向量 (vector)-雅可比积或 VJP 规则，以及雅可比-向量积或 JVP 规则）。
   - `vmap`：使用为每个原语定义的批处理规则，以确定如何将运算映射到额外的批处理维度上。
2. **编译接口：** 原语连接了高层 JAX API 和后端编译器，主要是 XLA（加速线性代数）。JAX 将 `jaxpr` 表示（一个原语图）转换为 XLA 的高级操作 (HLO) 指令。然后 XLA 进行复杂的优化，并将这些 HLO 指令编译成针对目标硬件（GPU、TPU 或 CPU）的高效机器码。
3. **可扩展性：** 尽管 JAX 提供了一整套内置原语，但该系统被设计为可扩展的。熟悉原语是向 JAX 添加新的自定义运算的根本。如果你需要集成用 C++ 或 CUDA 编写的专用算法，或者定义一个具有独特微分规则的运算，通常可以通过定义一个新的原语并为其提供必要的规则来实现。

### 转换过程

从 Python 函数到执行的过程通常遵循以下步骤，其中原语扮演着重要角色：

> 这是使用 JAX 运算的 Python 函数如何转换为可执行代码的视图。原语构成了 `jaxpr` 表示的主要部分，它是 XLA 编译器的输入。

### 原语与函数

区分 `jax.numpy.add` 这样的 JAX 函数和 `add_p` 这样的原语会有助于理解。函数 `jax.numpy.add` 是一个 Python 包装器，它提供了一个熟悉的 NumPy 风格接口。当 JAX 使用此函数追踪代码时，它通常会将相应的原语 `add_p` 注册到 `jaxpr` 中。原语是内部表示，承载着转换和编译所需的必要信息。

实质上，原语是不可简化的构成单元，JAX 强大的转换和编译系统就是建立在其之上。尽管在日常使用中你可能不会直接与它们交互，但了解它们的作用对于深入优化、调试和扩展 JAX 的能力是不可或缺的，尤其是在我们将在接下来的章节中研究自定义运算时。

## 参考资料

- [How JAX Primitives Work](https://jax.readthedocs.io/en/latest/jax-101/primitives.html) — The JAX Team (2024)
  Publisher: Google
  详细说明了JAX原语、其在追踪中的作用、`jaxpr`表示以及执行过程，直接支持本节的核心内容。
- [JAX: Composable transformations of Python+NumPy programs](https://github.com/google/jax) — The JAX Team (2018-present)
  JAX项目的官方GitHub仓库，作为该框架架构、设计和持续开发的主要资源。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla/) — Google Developers (2024)
  谷歌XLA编译器概述，它是JAX的后端，详细介绍了其在各种硬件加速器上优化数值计算的能力。
- [Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation](https://epubs.siam.org/doi/book/10.1137/1.9780898717765) — Andreas Griewank and George F. Corliss (2008)
  Publisher: Society for Industrial and Applied Mathematics; DOI: [10.1137/1.9780898717765](https://doi.org/10.1137/1.9780898717765)
  一本关于自动微分的基础教材，涵盖了前向和反向模式自动微分等概念的理论基础，这些概念对JAX的`grad`转换至关重要。
