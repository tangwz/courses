---
course: "getting-started-julia-programming"
sourceUrl: "https://apxml.com/zh/courses/getting-started-julia-programming/chapter-4-working-with-collections"
sourceId: 1196
chapter: "working-with-collections"
title: "使用集合：数组、元组、字典和集合"
order: 4
description: "学习Julia多样的集合类型，包括数组、元组、字典和集合，以实现高效数据管理。"
hasQuiz: true
---

在掌握了Julia的基本数据类型和控制流之后，我们现在开始学习如何组织和管理数据集。存储单个信息固然有用，但许多编程任务需要以结构化的方式处理多个值。

本章介绍Julia的主要集合类型。您将学到：

*   **数组**: 有序、可变的序列，适用于内容可能变化的列表。我们将介绍如何创建数组，通过索引访问元素，并执行常见的修改操作。
*   **元组**: 有序、不可变的序列，非常适合用于创建后结构和内容不应改变的固定关联数据组。
*   **字典**: 无序集合，以键值对的形式存储数据。这种结构在已知键时能高效地取回数据。
*   **集合**: 包含唯一元素的无序集合，适用于检查成员关系或执行并集、交集等集合运算。

我们将讲解如何创建、访问和操作这些集合。您将了解它们各自的特点、常用方法和典型用途，从而能根据不同的数据管理情况选择合适的集合类型。此外，我们还将介绍推导式，这是一种简洁且富有表现力的数据集合构建语法。
