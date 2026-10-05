# 第 5 章：使用 `pmap` 在多设备上并行计算

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap)

[返回课程目录](../README.md)

现代硬件通常包含多个加速器，例如 GPU 或 TPU。尽管 `jax.jit` 优化单设备代码，且 `jax.vmap` 高效处理批处理，但它们本身并不会将计算分布到`*`多个`*`设备上。本章介绍 `jax.pmap`（并行映射），它是 JAX 用于将计算分布到不同设备上并行运行的函数变换。

您将学到：
*   单程序多数据 (SPMD) 并行原理，以及 `pmap` 所采用的执行模型。
*   如何使用 `jax.pmap` 在多个设备上同时执行相同的函数，每个设备处理不同的数据切片。
*   指定如何使用 `in_axes` 和 `out_axes` 等参数将数据分发到设备并从设备收集的方法。
*   在经过 `pmap` 转换的函数中，集体操作（例如使用 `jax.lax` 原语在设备间求和或求平均）的用法。
*   `pmap` 如何与其他 JAX 变换（例如 `jit` 和 `grad`）结合使用。

学完本章后，您将能够在多设备系统上应用 `pmap` 为您的 JAX 程序实现数据并行。

## 小节

- 1. [数据并行 (SPMD) 介绍](01-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%20%28SPMD%29%20%E4%BB%8B%E7%BB%8D.md)
- 2. [介绍 \`jax.pmap\`](02-%E4%BB%8B%E7%BB%8D%20%60jax.pmap%60.md)
- 3. [将数据映射到设备 (\`in_axes\`, \`out_axes\`)](03-%E5%B0%86%E6%95%B0%E6%8D%AE%E6%98%A0%E5%B0%84%E5%88%B0%E8%AE%BE%E5%A4%87%20%28%60in_axes%60%2C%20%60out_axes%60%29.md)
- 4. [设备网格与轴名称](04-%E8%AE%BE%E5%A4%87%E7%BD%91%E6%A0%BC%E4%B8%8E%E8%BD%B4%E5%90%8D%E7%A7%B0.md)
- 5. [集体操作（\`lax.psum\`、\`lax.pmean\`等）](05-%E9%9B%86%E4%BD%93%E6%93%8D%E4%BD%9C%EF%BC%88%60lax.psum%60%E3%80%81%60lax.pmean%60%E7%AD%89%EF%BC%89.md)
- 6. [将 \`pmap\` 与其他变换结合使用](06-%E5%B0%86%20%60pmap%60%20%E4%B8%8E%E5%85%B6%E4%BB%96%E5%8F%98%E6%8D%A2%E7%BB%93%E5%90%88%E4%BD%BF%E7%94%A8.md)
- 7. [调试 \`pmap\` 化的函数](07-%E8%B0%83%E8%AF%95%20%60pmap%60%20%E5%8C%96%E7%9A%84%E5%87%BD%E6%95%B0.md)
- 8. [动手实践：并行计算](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B9%B6%E8%A1%8C%E8%AE%A1%E7%AE%97.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/quiz)
