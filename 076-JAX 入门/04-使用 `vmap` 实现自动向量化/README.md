# 第 4 章：使用 `vmap` 实现自动向量化

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap)

[返回课程目录](../README.md)

在数值计算和机器学习中，您经常需要同时对多个数据点应用同一个函数，这个过程通常被称为批处理。尽管您可以编写显式循环或依赖于为批处理操作设计的函数，但这些方法有时会很冗长，或者需要手动仔细管理数组维度。

JAX 提供了一个转换，`jax.vmap`，专门用于自动向量化。它允许您将一个为处理单个数据点而编写的函数，有效地应用到整个批次（或多个批次）的数据上，通常无需重写原函数的逻辑。`vmap` 相当于自动为您的计算添加了一个“批次维度”。

在这一章，您将学习：

*   向量化的含义以及它提升性能的原因。
*   如何使用 `jax.vmap` 对处理单个或多个参数的函数进行向量化。
*   如何使用 `in_axes` 和 `out_axes` 参数控制哪些轴被映射。
*   如何通过嵌套调用 `vmap` 来应对更复杂的情形。
*   `vmap` 如何与 `jit` 和 `grad` 等 JAX 的其他转换配合。
*   使用 `vmap` 的一些重要事项。

学完本章后，您将能够使用 `vmap` 来让 JAX 中的批处理代码更简洁，并通常运行得更快。

## 小节

- 1. [向量化的原理](01-%E5%90%91%E9%87%8F%E5%8C%96%E7%9A%84%E5%8E%9F%E7%90%86.md)
- 2. [介绍 \`jax.vmap\`](02-%E4%BB%8B%E7%BB%8D%20%60jax.vmap%60.md)
- 3. [对特定参数进行映射（\`in_axes\`，\`out_axes\`）](03-%E5%AF%B9%E7%89%B9%E5%AE%9A%E5%8F%82%E6%95%B0%E8%BF%9B%E8%A1%8C%E6%98%A0%E5%B0%84%EF%BC%88%60in_axes%60%EF%BC%8C%60out_axes%60%EF%BC%89.md)
- 4. [处理多个批处理参数](04-%E5%A4%84%E7%90%86%E5%A4%9A%E4%B8%AA%E6%89%B9%E5%A4%84%E7%90%86%E5%8F%82%E6%95%B0.md)
- 5. [嵌套 \`vmap\`](05-%E5%B5%8C%E5%A5%97%20%60vmap%60.md)
- 6. [结合 \`vmap\` 与 \`jit\` 和 \`grad\`](06-%E7%BB%93%E5%90%88%20%60vmap%60%20%E4%B8%8E%20%60jit%60%20%E5%92%8C%20%60grad%60.md)
- 7. [\`vmap\`的性能考量](07-%60vmap%60%E7%9A%84%E6%80%A7%E8%83%BD%E8%80%83%E9%87%8F.md)
- 8. [动手实践：函数向量化](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%90%91%E9%87%8F%E5%8C%96.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-4-automatic-vectorization-vmap/quiz)
