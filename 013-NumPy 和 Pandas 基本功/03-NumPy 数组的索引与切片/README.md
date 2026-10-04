# 第 3 章：NumPy 数组的索引与切片

来源：[原章节](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing)

[返回课程目录](../README.md)

既然你已经会创建 NumPy 数组了，接下来我们着重讲解如何处理数组所含的数据。本章将介绍访问和修改这些数组中元素的方法。

你将学习如何：

*   使用索引获取单个元素 (例如，`array[i]` 或 `array[row, col]`)。
*   使用切片表示法提取子数组，适用于一维和二维结构 (例如，`array[start:stop:step]`)。
*   通过布尔索引根据逻辑条件选择数据。
*   使用整数数组访问特定、可能不连续的元素，这通常称为“花式索引”。
*   使用这些选择方法修改数组中的值。

掌握这些索引和切片方法是有效进行 NumPy 数据处理的基本要求。

## 小节

- 1. [访问单个元素](01-%E8%AE%BF%E9%97%AE%E5%8D%95%E4%B8%AA%E5%85%83%E7%B4%A0.md)
- 2. [一维数组切片](02-%E4%B8%80%E7%BB%B4%E6%95%B0%E7%BB%84%E5%88%87%E7%89%87.md)
- 3. [二维数组切片](03-%E4%BA%8C%E7%BB%B4%E6%95%B0%E7%BB%84%E5%88%87%E7%89%87.md)
- 4. [布尔索引](04-%E5%B8%83%E5%B0%94%E7%B4%A2%E5%BC%95.md)
- 5. [花式索引](05-%E8%8A%B1%E5%BC%8F%E7%B4%A2%E5%BC%95.md)
- 6. [修改数组子集](06-%E4%BF%AE%E6%94%B9%E6%95%B0%E7%BB%84%E5%AD%90%E9%9B%86.md)
- 7. [动手实践：从数组中选择数据](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BB%8E%E6%95%B0%E7%BB%84%E4%B8%AD%E9%80%89%E6%8B%A9%E6%95%B0%E6%8D%AE.md)

章节测验：[在线测验](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing/quiz)
