# 第 5 章：互操作性与自定义操作

来源：[原章节](https://apxml.com/zh/courses/advanced-jax/chapter-5-jax-interoperability-custom-operations)

[返回课程目录](../README.md)

JAX 计算常需要与科学 Python 生态的其他部分协同工作，或需要专门的底层功能。本章讨论如何将 JAX 与外部系统连接，并扩展其核心操作。

您将学到实用的方法，包括：
*   将数据在 JAX 和 NumPy 数组之间进行转换。
*   应用 DLPack 标准实现高效、零拷贝的数据共享，与 PyTorch 或 TensorFlow 等其他库。
*   在 JAX 程序中执行外部 Python 函数，使用 `host_callback` 和 `pure_callback`，了解它们的用途和限制。
*   理解 JAX 原语作为系统已知的基本操作。
*   构建自定义原语，通过定义它们的抽象求值行为（形状、数据类型），实现后端特定的低层化规则（XLA HLO 生成），并指定自定义微分规则（JVP/VJP），以确保它们能完全融入 JAX 的转换系统。

## 小节

- 1. [JAX 与 NumPy 的集成](01-JAX%20%E4%B8%8E%20NumPy%20%E7%9A%84%E9%9B%86%E6%88%90.md)
- 2. [使用 DLPack 实现零拷贝数据共享](02-%E4%BD%BF%E7%94%A8%20DLPack%20%E5%AE%9E%E7%8E%B0%E9%9B%B6%E6%8B%B7%E8%B4%9D%E6%95%B0%E6%8D%AE%E5%85%B1%E4%BA%AB.md)
- 3. [使用 jax.experimental.host_callback 调用外部 CPU/GPU 代码](03-%E4%BD%BF%E7%94%A8%20jax.experimental.host_callback%20%E8%B0%83%E7%94%A8%E5%A4%96%E9%83%A8%20CPU-GPU%20%E4%BB%A3%E7%A0%81.md)
- 4. [使用 jax.pure_callback 进行无副作用调用](04-%E4%BD%BF%E7%94%A8%20jax.pure_callback%20%E8%BF%9B%E8%A1%8C%E6%97%A0%E5%89%AF%E4%BD%9C%E7%94%A8%E8%B0%83%E7%94%A8.md)
- 5. [JAX 原语简介](05-JAX%20%E5%8E%9F%E8%AF%AD%E7%AE%80%E4%BB%8B.md)
- 6. [定义自定义原语](06-%E5%AE%9A%E4%B9%89%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8E%9F%E8%AF%AD.md)
- 7. [实现抽象求值规则](07-%E5%AE%9E%E7%8E%B0%E6%8A%BD%E8%B1%A1%E6%B1%82%E5%80%BC%E8%A7%84%E5%88%99.md)
- 8. [为后端（CPU/GPU/TPU）实现转换规则](08-%E4%B8%BA%E5%90%8E%E7%AB%AF%EF%BC%88CPU-GPU-TPU%EF%BC%89%E5%AE%9E%E7%8E%B0%E8%BD%AC%E6%8D%A2%E8%A7%84%E5%88%99.md)
- 9. [制定自定义原语的求导规则](09-%E5%88%B6%E5%AE%9A%E8%87%AA%E5%AE%9A%E4%B9%89%E5%8E%9F%E8%AF%AD%E7%9A%84%E6%B1%82%E5%AF%BC%E8%A7%84%E5%88%99.md)
- 10. [实践：整合 C++ 函数](10-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%95%B4%E5%90%88%20C%2B%2B%20%E5%87%BD%E6%95%B0.md)
