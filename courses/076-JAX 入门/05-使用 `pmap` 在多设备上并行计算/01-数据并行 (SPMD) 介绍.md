# 数据并行 (SPMD) 介绍

来源：[原文](https://apxml.com/zh/courses/getting-started-with-jax/chapter-5-parallelization-across-devices-pmap/intro-data-parallelism-spmd)

[返回章节目录](README.md) · [返回课程目录](../README.md)

`jax.jit`和`jax.vmap`等工具可以显著提升性能，但主要局限于单个计算设备（例如一个GPU或一个TPU核心）的范围。当您拥有多个加速器时，如何能同时运用它们来更快地处理更大的数据集或更复杂的模型？数据并行正是在此发挥作用。

数据并行是高性能计算和机器学习 (machine learning)中的一种常用策略。其主要思想很简单：如果您有一个大型数据集和多个处理单元，您可以将数据分发给这些单元，让每个单元对其分配到的数据块执行*相同*的计算。

### SPMD 模型

`jax.pmap`和许多数据并行实现所基于的执行模型被称为**SPMD**，它代表**单程序多数据**。

想象一下：

1. **单程序：** 您编写*一个*代码片段，*一个*函数，来定义您想要执行的计算（例如，神经网络 (neural network)的前向传播，物理模拟的一个步骤）。
2. **多数据：** 您的输入数据（如一大批图像或模拟参数 (parameter)）被分成更小的部分或分片。
3. **并行执行：** 每个可用设备（GPU、TPU核心）执行*完全相同的程序*（您的函数），但只对其分配到的数据分片进行操作。

这与MIMD（多指令多数据）等其他并行模型不同，在MIMD中，不同的处理器可能运行完全不同的程序。SPMD简化了编程模型，因为您只需考虑单一的程序结构。并行性源于将此单一程序同时应用于不同的数据。

假设您有一个函数 `process_data(x)` 和一个大型数据集 `X`。如果您有4个设备，SPMD方法会是这样：

> 数据被分割（分片）到多个设备上。每个设备并行地在其自己的数据分片上执行相同的程序（`process_data`）。结果通常在之后合并。

### JAX 为何选用 SPMD？

SPMD 模型与 JAX 的函数式编程方法及其对函数变换的侧重非常契合。`jax.pmap` 本质上是一种函数变换，它将一个为单个数据实例（或批次）编写的标准 Python 函数转换为一个可在多个设备上运行的 SPMD 程序。

这种方法的优点包括：

- **代码复用性：** 您编写的代码通常与为单个设备编写的代码非常相似，而 `pmap` 会处理并行执行的细节。
- **可伸缩性：** 它提供了一条将计算扩展到多个加速器的直接途径，这对于训练大型机器学习 (machine learning)模型或处理海量数据集非常重要。
- **效率：** 通过并行处理数据分片，与在单个设备上顺序处理整个数据集相比，可以显著减少总计算时间。

当然，有效的数据并行不仅仅涉及数据分割。通常，设备需要在计算过程中进行通信，例如，为了聚合结果（如机器学习训练中的梯度）或在模拟中交换边界信息。JAX 提供了一种称为“集合操作”（如用于对所有设备上的值求和的`jax.lax.psum`）的机制，这些操作在经过`pmap`转换的函数中运行，以处理这种设备间通信。我们将在本章后面讨论这些。

理解 SPMD 思想对有效使用 `pmap` 十分重要。它影响您如何组织数据输入以及如何思考计算在您可用硬件资源上的流程。随后的章节将展示如何使用 `jax.pmap` 在实践中应用此模型。

## 参考资料

- [JAX documentation for jax.pmap](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) — JAX core contributors (2024)
  官方文档，解释了 JAX 中 `jax.pmap` 用于并行执行的用法和功能，体现了 SPMD 模型。
- [An Introduction to Parallel Programming](https://www.elsevier.com/books/an-introduction-to-parallel-programming/pacheco/978-0-12-374260-5) — Peter S. Pacheco (2011)
  Publisher: Morgan Kaufmann
  一本经典的教科书，涵盖了并行计算的基本概念，包括对 SPMD 和 MIMD 模型的详细说明。
- [Distributed Deep Learning: A Review](https://link.springer.com/article/10.26599/BDMA.2019.2040001) — Yanzhao Hao, Haotian Zhang, Qun Li, and Haofei Li (2020)
  Journal: Big Data Mining and Analytics; Publisher: Springer; Volume: 3; Pages: 12-25; DOI: [10.26599/BDMA.2019.2040001](https://doi.org/10.26599/BDMA.2019.2040001)
  全面回顾了分布式深度学习技术，侧重于数据并行及其在机器学习中的应用。
- [JAX documentation for Distributed Arrays and Sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) — JAX core contributors (2024)
  解释了 JAX 如何管理跨多个设备的数据分布和分片，这对于有效的数据并行和 SPMD 执行至关重要。

---

[上一节](../04-%E4%BD%BF%E7%94%A8%20%60vmap%60%20%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%8A%A8%E5%90%91%E9%87%8F%E5%8C%96/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%87%BD%E6%95%B0%E5%90%91%E9%87%8F%E5%8C%96.md) · [下一节](02-%E4%BB%8B%E7%BB%8D%20%60jax.pmap%60.md)
