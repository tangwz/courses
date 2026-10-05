# 第 1 章：JAX 简介

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax)

[返回课程目录](../README.md)

本章介绍 JAX，它是一个为高性能数值计算设计的 Python 库，尤其适用于机器学习研究。我们会首先说明 JAX 是什么，以及它在 Python 科学计算体系中的位置。

你将了解 JAX 的核心思想，重点说明它与 NumPy 的关系以及对函数转换的依赖。我们将比较 `jax.numpy` 与标准 NumPy 库，指出你需要注意的主要异同点，例如不可变性。我们还将介绍安装 JAX 的具体步骤，以及如何为 CPU、GPU 和 TPU 等不同硬件进行配置。

最后，你将亲手实践创建和操作 JAX 数组（这一核心数据结构），并学习 JAX 如何在可用硬件设备上管理计算。到本章结束时，你将对 JAX 的作用、其基本 API 有一个初步认识，并知道如何设置环境来开始使用它。

## 小节

- 1. [JAX 是什么？](01-JAX%20%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F.md)
- 2. [JAX 对比 NumPy](02-JAX%20%E5%AF%B9%E6%AF%94%20NumPy.md)
- 3. [核心设计理念：函数变换](03-%E6%A0%B8%E5%BF%83%E8%AE%BE%E8%AE%A1%E7%90%86%E5%BF%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%8F%98%E6%8D%A2.md)
- 4. [安装与设置](04-%E5%AE%89%E8%A3%85%E4%B8%8E%E8%AE%BE%E7%BD%AE.md)
- 5. [使用 JAX 数组](05-%E4%BD%BF%E7%94%A8%20JAX%20%E6%95%B0%E7%BB%84.md)
- 6. [设备管理：CPU、GPU、TPU](06-%E8%AE%BE%E5%A4%87%E7%AE%A1%E7%90%86%EF%BC%9ACPU%E3%80%81GPU%E3%80%81TPU.md)
- 7. [动手练习：基本数组操作](07-%E5%8A%A8%E6%89%8B%E7%BB%83%E4%B9%A0%EF%BC%9A%E5%9F%BA%E6%9C%AC%E6%95%B0%E7%BB%84%E6%93%8D%E4%BD%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/getting-started-with-jax/chapter-1-introduction-to-jax/quiz)
