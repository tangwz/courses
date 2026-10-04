# 第 2 章：通过 JIT 编译加速函数

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit)

[返回课程目录](../README.md)

尽管 JAX 提供了一个熟悉的 NumPy 风格的 API，但对于科学计算和机器学习中常见的密集数值计算，标准的 Python 执行速度可能较慢，尤其是在针对 GPU 或 TPU 等加速器时。为了获得高效率，JAX 依赖于编译。

本章主要讲解 `jax.jit`，这是一个即时 (JIT) 编译转换。您将了解 `jit` 如何将 Python 函数转换成针对您的硬件进行适配的优化后的可执行代码。我们将介绍：

*   `jax.jit` 作为装饰器或函数的基本使用。
*   即时 (JIT) 编译如何通过一个称为“追踪”的过程来工作。
*   追踪对 Python 控制流（循环、条件语句）的影响。
*   在编译函数中如何处理静态值与动态值。
*   使用 `jit` 时的常见错误和注意事项。

在本章结束时，您将能够使用 `jax.jit` 大幅提升您的 JAX 计算速度。

## 小节

- 1. [速度提升：为何需要编译？](01-%E9%80%9F%E5%BA%A6%E6%8F%90%E5%8D%87%EF%BC%9A%E4%B8%BA%E4%BD%95%E9%9C%80%E8%A6%81%E7%BC%96%E8%AF%91%EF%BC%9F.md)
- 2. [介绍 \`jax.jit\`](02-%E4%BB%8B%E7%BB%8D%20%60jax.jit%60.md)
- 3. [JIT 工作原理：追踪与编译](03-JIT%20%E5%B7%A5%E4%BD%9C%E5%8E%9F%E7%90%86%EF%BC%9A%E8%BF%BD%E8%B8%AA%E4%B8%8E%E7%BC%96%E8%AF%91.md)
- 4. [Python 控制流与 \`jit\`](04-Python%20%E6%8E%A7%E5%88%B6%E6%B5%81%E4%B8%8E%20%60jit%60.md)
- 5. [静态值与跟踪值](05-%E9%9D%99%E6%80%81%E5%80%BC%E4%B8%8E%E8%B7%9F%E8%B8%AA%E5%80%BC.md)
- 6. [\`jit\` 的常见问题](06-%60jit%60%20%E7%9A%84%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98.md)
- 7. [动手实践：应用 \`jit\`](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BA%94%E7%94%A8%20%60jit%60.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-2-accelerating-functions-jit/quiz)
