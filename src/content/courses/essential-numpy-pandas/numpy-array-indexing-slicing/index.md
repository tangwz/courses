---
course: "essential-numpy-pandas"
sourceUrl: "https://apxml.com/zh/courses/essential-numpy-pandas/chapter-3-numpy-array-indexing-slicing"
sourceId: 498
chapter: "numpy-array-indexing-slicing"
title: "NumPy 数组的索引与切片"
order: 3
description: "学习如何使用索引、切片、布尔索引和花式索引来访问和修改 NumPy 数组的元素和子集。"
hasQuiz: true
---

既然你已经会创建 NumPy 数组了，接下来我们着重讲解如何处理数组所含的数据。本章将介绍访问和修改这些数组中元素的方法。

你将学习如何：

*   使用索引获取单个元素 (例如，`array[i]` 或 `array[row, col]`)。
*   使用切片表示法提取子数组，适用于一维和二维结构 (例如，`array[start:stop:step]`)。
*   通过布尔索引根据逻辑条件选择数据。
*   使用整数数组访问特定、可能不连续的元素，这通常称为“花式索引”。
*   使用这些选择方法修改数组中的值。

掌握这些索引和切片方法是有效进行 NumPy 数据处理的基本要求。
