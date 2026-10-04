---
course: "getting-started-with-jax"
chapter: "accelerating-functions-jit"
lesson: "why-compile-jax"
sourceId: 1693
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/why-compile-jax"
title: "速度提升：为何需要编译？"
description: "了解为加速器编译 Python 代码的性能优势。"
order: 1
plots: ["plots/1693-0.json"]
sourceHash: "ef4084cc5b1b1b6e224c137e6794a65c6ca2980e804e36261df12946f0310bc3"
sourceCorrections: []
---

JAX 提供了一个熟悉的 `jax.numpy` 接口，让你能够编写看起来与标准 NumPy 代码非常相似的数值计算代码。然而，如果你直接运行这些代码，你可能会发现结果不尽如人意：它的速度通常不比 NumPy 快多少，有时甚至可能更慢。既然 JAX 以在 GPU 和 TPU 等加速器上的高性能著称，为什么会这样呢？

答案在于标准 Python 代码的执行方式。Python 是一种解释型语言。当你运行一个 Python 函数时，解释器会逐行读取、解释并执行你的代码。这种动态特性提供了很大的灵活性，但也带来了显著的开销，尤其是在涉及循环或对大型数组重复操作的数值计算中。

考虑一个典型的数值计算任务，比如对一个大型数组逐元素应用函数。在纯 Python 中，这很可能涉及一个 `for` 循环。循环的每次迭代都会产生解释器开销：检查类型、查找方法、处理潜在错误以及执行 Python 字节码。虽然 NumPy 通过使用预编译的 C 或 Fortran 例程进行数组操作，从而显著改进了这一点，但 Python 解释器在协调这些操作时仍然发挥着作用。

现代硬件加速器，例如 GPU 和 TPU，擅长执行大型并行计算。它们拥有数千个核心，旨在同时执行数学运算。然而，Python 解释器的逐行执行模式却成了瓶颈。从 CPU 向 GPU/TPU 发送单个指令或小型操作会产生延迟。加速器等待指令的时间多于实际计算的时间。这就像试图通过扩音器一次只对一名工人喊话来指挥一个庞大的施工队伍；如果预先提供完整的蓝图，队伍能够完成更多工作。

这就是编译变得非常必要的地方。我们不再逐步解释 Python 代码，而是可以将整个函数或其重要部分转换为针对目标硬件（CPU、GPU 或 TPU）进行了优化的低级表示（例如 XLA HLO）。这个编译过程是在函数首次使用特定输入类型和形状调用时即时 (JIT) 发生的。

编译后的函数具有以下特点：

1. **减少解释器开销：** 在编译代码块执行期间，Python 解释器基本不再参与。
2. **优化操作：** 编译器可以将多个操作融合在一起，消除了在内存中存储中间结果的需要。例如，像 $y = a * x + b$ 这样的逐元素操作通常可以融合为 GPU/TPU 上的单个内核启动。
3. **发挥硬件并行性：** 编译器了解目标架构（CPU、GPU、TPU），并生成充分发挥其并行性的代码。它能够安排大量计算块在加速器上高效运行。

以下图表展示了针对某个数值计算任务，不同执行方法之间潜在的性能差异。



![性能对比](plots/1693-0.json)



> 在从解释型 Python 代码转向编译后的 JAX 代码时，一个典型的数值计算任务的相对速度提升。请注意 y 轴上的对数刻度。具体的加速效果因任务和硬件而有很大差异。

如果不进行编译，直接执行的 JAX 操作通常表现与 NumPy 操作相似，依赖于为单个操作调度预编译的内核，但它们之间仍然会产生 Python 开销。JAX 的速度 *潜力* 源于其编译能力，但这种潜力只有在明确请求时才能实现。

因此，为了获得 JAX 提供的性能优势，特别是在加速器上，我们需要指示 JAX 编译我们的函数。实现这一目的的主要工具是 `jax.jit`，我们将在接下来详细讨论它。它充当“蓝图生成器”，将你的 Python 函数转换为高度优化的可执行代码。

## 参考资料

- [Just-in-Time Compilation (JIT) in JAX](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) — The JAX Authors (2024)
  Publisher: The JAX Authors
  JAX官方文档，说明了即时编译加速函数的功能和优势。
- [JAX: Composable transformations of Python+NumPy programs](https://github.com/google/jax) — James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Skye Wanderman-Milne, and Adam Paszke (2018)
  Publisher: Google LLC
  JAX的基础项目库，概述了其设计原则，包括可组合的转换和JIT编译。
- [XLA: Optimizing Compilers for Machine Learning](https://www.tensorflow.org/xla/) — Google (2024)
  Publisher: Google
  XLA的官方文档，JAX使用XLA作为后端线性代数编译器，为各种硬件生成高性能代码。
