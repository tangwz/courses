# JAX 是什么？

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/what-is-jax)

[返回章节目录](README.md) · [返回课程目录](../README.md)

JAX 代表 Just After Execution（或者开玩笑地说，是“在加速器上运行的 NumPy，带梯度”）。它由 Google Research 开发，是一个专门为高性能数值计算和机器学习 (machine learning)研究设计的 Python 库。如果您熟悉 NumPy，会发现 JAX 的主要接口非常相似，但其内部运作方式大不相同，以实现显著的性能提升，特别是在 GPU（图形处理器）和 TPU（张量处理器）等现代硬件上。

可以将 JAX 看作一个平台，而不是简单的另一个数组库，它构建在两个主要支柱之上：

1. **熟悉的 NumPy 接口：** JAX 提供了 `jax.numpy`，这是一个旨在紧密模仿知名 NumPy API 的接口。这让您可以使用熟悉的函数（如 `jnp.array`、`jnp.dot`、`jnp.sum` 等，`jnp` 是约定俗成的别名）编写数值代码。如果您已经在 Python 科学计算环境中工作，这会大幅降低使用门槛。我们将在下一节考察其细微但重要的差异，例如数组的不可变性。
2. **可组合的函数变换：** 这是 JAX 真正与众不同的地方。JAX 不是像标准 Python 或 NumPy 那样逐行直接执行您的 Python 代码，而是围绕着变换您的 Python 函数而构建。最主要的变换是：

   - `jax.jit`：使用强大的 XLA（加速线性代数）编译器编译您的 Python 函数，以在加速器上实现显著的加速。
   - `jax.grad`：自动计算您 Python 函数的梯度（导数）。这对于机器学习中常见的优化任务非常重要。
   - `jax.vmap`：自动向量 (vector)化您的函数，使它们能够高效地在批处理数据上运行，而无需您手动重写循环或数组操作。
   - `jax.pmap`：实现函数在多个设备（例如，多个 GPU 或 TPU 核心）上的并行执行，主要用于数据并行。

这些变换可以任意组合，这意味着您可以对已经求过梯度的函数进行 `jit` 编译，或者对已 `jit` 编译且包含用于 `pmap` 的集体操作的函数进行 `vmap` 向量化 (quantization)。这种可组合性使得代码表达力强且性能高。

其核心在于，JAX 接收您用 `jax.numpy` 编写的 Python 函数，并使用这些变换将它们转换为中间表示。然后，此表示被送入 XLA 编译器，该编译器专门为目标硬件（CPU、GPU 或 TPU）优化计算。

> JAX 通过变换处理 Python 函数，通过 XLA 编译它们，并在目标硬件上执行优化后的代码。

这种基于变换的方法鼓励函数式编程风格。由于 JAX 通常会追踪您的函数以对其进行编译（正如我们将在 `jit` 中看到的那样），“纯粹”的函数（无副作用，对于相同的输入总是产生相同的输出）与 JAX 的变换配合得最顺畅。虽然 JAX 不严格强制纯粹性，但理解此原则有助于编写有效的 JAX 代码，特别是在管理状态时（第六章有提及）。

本质上，JAX 提供了一种方式，让您可以使用熟悉的 NumPy 式语法在 Python 中编写高级数值程序，同时获得在专用加速器上编译和执行的性能优势，以及自动微分和向量化等强大功能，所有这些都通过函数变换的应用实现。

## 参考资料

- [JAX Documentation](https://jax.readthedocs.io/en/latest/) — JAX developers (2024)
  JAX API、函数转换和核心概念的综合指南。
- [XLA: Accelerated Linear Algebra](https://www.tensorflow.org/xla) — Google (2024)
  Publisher: Google
  关于XLA编译器的信息，JAX使用它来优化加速器上的数值计算。

---

[下一节](02-JAX%20%E5%AF%B9%E6%AF%94%20NumPy.md)
