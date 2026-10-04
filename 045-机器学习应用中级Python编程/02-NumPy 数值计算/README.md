# 第 2 章：NumPy 数值计算

来源：[原章节](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy)

[返回课程目录](../README.md)

机器学习的核心是高效处理数值数据。尽管 Python 内置列表灵活多用，但它们并未针对数据分析和模型构建中常见的大规模数值运算进行优化。NumPy (Numerical Python) 正是在此时发挥作用。它为 Python 中的科学计算提供了核心支持，引入了强大的 N 维数组对象，即 `ndarray`。

本章旨在让你熟练掌握 NumPy。你将学习如何：

*   使用多种方法创建 NumPy 数组。
*   进行基本的数组操作，包括索引、切片和形状变换。
*   使用通用函数 (ufuncs) 在整个数组上高效应用数学和逻辑运算。
*   理解广播机制，即 NumPy 用于对兼容形状的数组执行操作的方式。
*   使用 NumPy 进行基本的线性代数任务和统计计算。
*   从文件读取数据并将数组数据保存到文件。

掌握 NumPy 是有效使用 Pandas 和 Scikit-learn 等其他基于它构建的数据科学库的必要一步。到本章结束时，你将能够处理机器学习工作流程中核心的数值数据结构和操作。

## 小节

- 1. [NumPy 数组简介](01-NumPy%20%E6%95%B0%E7%BB%84%E7%AE%80%E4%BB%8B.md)
- 2. [数组创建方法](02-%E6%95%B0%E7%BB%84%E5%88%9B%E5%BB%BA%E6%96%B9%E6%B3%95.md)
- 3. [NumPy 数组的索引与切片](03-NumPy%20%E6%95%B0%E7%BB%84%E7%9A%84%E7%B4%A2%E5%BC%95%E4%B8%8E%E5%88%87%E7%89%87.md)
- 4. [数组计算与通用函数](04-%E6%95%B0%E7%BB%84%E8%AE%A1%E7%AE%97%E4%B8%8E%E9%80%9A%E7%94%A8%E5%87%BD%E6%95%B0.md)
- 5. [广播规则及应用](05-%E5%B9%BF%E6%92%AD%E8%A7%84%E5%88%99%E5%8F%8A%E5%BA%94%E7%94%A8.md)
- 6. [NumPy 中的线性代数运算](06-NumPy%20%E4%B8%AD%E7%9A%84%E7%BA%BF%E6%80%A7%E4%BB%A3%E6%95%B0%E8%BF%90%E7%AE%97.md)
- 7. [NumPy中的统计函数](07-NumPy%E4%B8%AD%E7%9A%84%E7%BB%9F%E8%AE%A1%E5%87%BD%E6%95%B0.md)
- 8. [读取和写入数组数据到文件](08-%E8%AF%BB%E5%8F%96%E5%92%8C%E5%86%99%E5%85%A5%E6%95%B0%E7%BB%84%E6%95%B0%E6%8D%AE%E5%88%B0%E6%96%87%E4%BB%B6.md)
- 9. [动手实践：NumPy 数组操作](09-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9ANumPy%20%E6%95%B0%E7%BB%84%E6%93%8D%E4%BD%9C.md)

章节测验：[在线测验](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-2-numerical-computing-numpy/quiz)
