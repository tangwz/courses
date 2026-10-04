---
course: "data-structures-algorithms-ml"
chapter: "heaps-priority-queues-ml"
lesson: "heap-properties"
sourceId: 5319
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-5-heaps-priority-queues-ml/heap-properties"
title: "堆数据结构属性"
description: "了解二叉堆的定义和属性，包括最小堆和最大堆。"
order: 1
plots: []
sourceHash: "c60b1b6c41ca9d792546c4450a801ec00a16e76b3ca1e57250ee14189bca8040"
sourceCorrections: []
---

正如章节引言中提到，根据优先级管理元素对许多优化任务非常重要。**堆**是一种特殊化的基于树的数据结构，专门用于高效处理这些基于优先级的操作。虽然存在多种类型的堆，但我们将主要讲解**二叉堆**，它是最常见的实现方式。

二叉堆遵循两个主要属性，确保其效率：**堆属性**和**结构属性**。

## 堆属性：元素排序

堆的核心理念是它在父节点和子节点之间保持的特定排序关系。这种关系决定了堆是在顶部追踪最小元素还是最大元素。主要有两种类型：

### 最小堆

在最小堆中，每个父节点的值都小于或等于其子节点的值。此属性在树中的所有节点上递归成立。因此，整个集合中最小的元素始终位于根节点。

> 一个简单的最小堆。请注意，根节点 (10) 是最小值，并且每个父节点都小于或等于其子节点（例如，15 <= 40，15 <= 50）。

### 最大堆

相反，在最大堆中，每个父节点的值都大于或等于其子节点的值。这确保了集合中最大的元素始终位于根节点。

> 一个简单的最大堆。根节点 (100) 是最大值，并且每个父节点都大于或等于其子节点（例如，50 >= 15，50 >= 40）。

这种堆属性对于快速访问最小（在最小堆中）或最大（在最大堆中）元素很重要。

## 结构属性：完全二叉树

第二个属性与树的形状有关。二叉堆始终被维护为**完全二叉树**。这意味着：

1. 树的所有层级都已完全填充，最低层可能除外。
2. 如果最低层未满，则该层中的节点从左到右填充，中间没有空缺。

> 一个完全二叉树。除了最后一层外，所有层都已填满，最后一层从左到右填充（节点 H 和 I）。

为什么这个结构属性很重要？它允许二叉堆使用简单的数组或列表高效存储。父节点和子节点之间的关系可以直接从它们在数组中的索引计算出来，无需显式指针。

对于零索引数组中索引为 $i$ 的节点：

- 其左子节点位于索引 $2i + 1$。
- 其右子节点位于索引 $2i + 2$。
- 其父节点位于索引 $\lfloor (i - 1) / 2 \rfloor$。

这种数组表示形式紧凑且对缓存友好。下面的图表显示了之前最大堆到数组表示的映射：

> 最大堆结构映射到其对应的数组表示。元素逐层从左到右填充数组。

这两个属性，即堆属性（最小或最大排序）和结构属性（允许数组表示的完全二叉树），是构成堆的根本。它们使堆能够成为优先队列的高效支撑，并方便了插入和提取等操作在对数时间复杂度内完成，具体为 $O(\log n)$，其中 $n$ 是堆中的元素数量。我们将在下一节讨论这些操作。

## 参考资料

- [Introduction to Algorithms, Fourth Edition](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  本教材解释了二叉堆及其结构和堆属性，以及基于数组的实现。
- [Algorithms, 4th Edition](https://algs4.cs.princeton.edu/) — Robert Sedgewick and Kevin Wayne (2011)
  Publisher: Addison-Wesley Professional
  这是一本关于基本算法的参考书，涵盖了堆及其基于数组的表示。
- [Lecture 10: Heaps and Heapsort](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2020/resources/lecture-10-heaps-and-heapsort/) — Erik Demaine, Jason Ku, and Justin Solomon (2020)
  Publisher: MIT OpenCourseWare
  提供了来自一所知名大学关于二叉堆属性和结构的学术讲座。
