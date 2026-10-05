# 第 2 章：适用于机器学习的 Python 性能优化

来源：[原章节](https://apxml.com/zh/courses/advanced-python-programming-ml/chapter-2-python-performance-optimization-ml)

[返回课程目录](../README.md)

尽管 Python 为机器学习开发提供了极大的灵活性，但其解释型特性可能导致性能问题，尤其是在处理大型数据集或计算密集型算法时。在小样本上运行良好的代码，在生产环境中可能会变得极其缓慢。高效执行通常是实用机器学习应用的一项需求。

本章主要介绍识别并解决 Python 机器学习代码中性能瓶颈的方法。你将学习如何：

*   使用性能分析工具定位程序中的缓慢部分。
*   编写更高效的代码，运用优化的 NumPy 和 Pandas 使用技巧。
*   通过使用 Cython 和 Numba 等即时编译工具，加速重要的代码片段。
*   理解 Python 全局解释器锁 (GIL) 对 CPU 密集型任务并发性的影响。
*   应用内存分析和优化策略，以减少应用程序的内存占用。

应用这些方法后，你可以大幅提升数据处理、特征工程和模型训练的速度，使你的机器学习工作流程更快、更具扩展性。我们将从识别性能问题入手，逐步实施使用成熟 Python 库和技术的具体方案。

## 小节

- 1. [Python 代码性能分析：找出瓶颈](01-Python%20%E4%BB%A3%E7%A0%81%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%EF%BC%9A%E6%89%BE%E5%87%BA%E7%93%B6%E9%A2%88.md)
- 2. [优化 NumPy 操作](02-%E4%BC%98%E5%8C%96%20NumPy%20%E6%93%8D%E4%BD%9C.md)
- 3. [大型数据集的Pandas高效用法](03-%E5%A4%A7%E5%9E%8B%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84Pandas%E9%AB%98%E6%95%88%E7%94%A8%E6%B3%95.md)
- 4. [Cython简介：加速Python代码](04-Cython%E7%AE%80%E4%BB%8B%EF%BC%9A%E5%8A%A0%E9%80%9FPython%E4%BB%A3%E7%A0%81.md)
- 5. [使用 Numba 进行即时编译](05-%E4%BD%BF%E7%94%A8%20Numba%20%E8%BF%9B%E8%A1%8C%E5%8D%B3%E6%97%B6%E7%BC%96%E8%AF%91.md)
- 6. [理解Python的全局解释器锁（GIL）](06-%E7%90%86%E8%A7%A3Python%E7%9A%84%E5%85%A8%E5%B1%80%E8%A7%A3%E9%87%8A%E5%99%A8%E9%94%81%EF%BC%88GIL%EF%BC%89.md)
- 7. [内存分析与优化方法](07-%E5%86%85%E5%AD%98%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95.md)
- 8. [动手实践：优化特征工程函数](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BC%98%E5%8C%96%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E5%87%BD%E6%95%B0.md)
