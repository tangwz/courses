---
course: "data-structures-algorithms-ml"
chapter: "heaps-priority-queues-ml"
lesson: "priority-queues-heaps"
sourceId: 5325
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-5-heaps-priority-queues-ml/priority-queues-heaps"
title: "使用堆实现优先级队列"
description: "了解堆如何为优先级队列的实现提供高效的底层结构。"
order: 3
plots: []
sourceHash: "f30e4cdfe6dba7d289ccab58019f47e8d78ecfe297c3fff33e003821b05f1294"
sourceCorrections: []
---

堆数据结构自身提供了基于堆属性（最小堆或最大堆）管理集合的高效方法，但它最直接的用途通常是作为**优先级队列**的支撑。优先级队列是一种抽象数据类型，其运行方式类似于普通队列或栈，但每个元素都有一个关联的“优先级”。优先级较高的元素会在优先级较低的元素之前得到处理。

优先级队列定义的主要操作通常是：

1. **插入:** 将元素及其关联的优先级添加到队列中。
2. **提取最小值 / 提取最大值:** 移除并返回优先级最小（对于最小优先级队列）或最大（对于最大优先级队列）的元素。
3. **查看:** 返回优先级最小/最大的元素，但不将其移除。

### 为什么要用堆实现优先级队列？

堆非常适合实现优先级队列，因为它们的性能特点。让我们看看其他选择：

- **未排序列表/数组:** 插入操作很快 ($O(1)$)，但查找和提取最小或最大元素需要遍历整个列表 ($O(n)$)。查看操作也需要 $O(n)$。
- **已排序列表/数组:** 查看和提取最小/最大元素（在两端）很快 ($O(1)$)，但插入操作需要找到正确位置并移动元素，耗时 $O(n)$。
- **平衡二叉搜索树:** 插入、提取和查看操作的时间复杂度都是 $O(\log n)$。然而，二叉搜索树维护了完全排序，这比优先级队列所需的功能更多，而且它们的实现可能比堆更复杂。

**堆提供了一个理想选择：**

- **插入:** 添加一个元素涉及将其放在末尾并“向上冒泡”以维护堆属性，耗时 $O(\log n)$。
- **提取（最小/最大）:** 移除根元素（优先级最高/最低的元素）涉及将其与最后一个元素交换，然后移除，并“向下冒泡”新的根元素，同样耗时 $O(\log n)$。
- **查看:** 访问最小（在最小堆中）或最大（在最大堆中）元素就是访问根节点，耗时 $O(1)$。

这种 $O(\log n)$ 的性能对于核心动态操作而言，使堆成为实现通用优先级队列的首选。

### 堆操作到优先级队列操作的映射

使用堆实现优先级队列涉及它们的操作之间直接的映射：

- **优先级队列 `insert(item, priority)`:** 对应于堆的 `insert` 操作。项目及其优先级被添加到堆中，并在 $O(\log n)$ 时间内调整堆结构以根据优先级值维护堆属性。
- **优先级队列 `extract_min()` (用于最小优先级队列):** 对应于堆的 `extract_min` 操作。根元素（具有最小优先级）被移除并返回，堆结构在 $O(\log n)$ 时间内恢复。
- **优先级队列 `extract_max()` (用于最大优先级队列):** 对应于堆的 `extract_max` 操作。根元素（具有最大优先级）被移除并返回，堆结构在 $O(\log n)$ 时间内恢复。
- **优先级队列 `peek()`:** 对应于访问堆的根元素（例如，如果使用基于数组的实现，则为 `heap[0]`）。这需要 $O(1)$ 时间。

下面是一个图表，说明了一个最小堆结构，其中节点包含 `(优先级, 项目)` 元组。优先级最小的元素 (3) 位于根部，可用于高效的查看或提取。

> 一个最小堆，将元素存储为 `(优先级, 项目)` 元组。根节点 `(3, 'C')` 包含优先级最低的项目。

### 在 Python 中使用 `heapq` 实现优先级队列

Python 的标准库包含 `heapq` 模块，它提供了一个高效的堆队列算法（也称为优先级队列算法）实现。重要的一点是，`heapq` 直接在 Python 列表上实现了**最小堆**。

以下是你如何使用 `heapq` 管理最小优先级队列：

```python
import heapq

# 初始化一个空列表作为堆
min_priority_queue = []

# 插入带有优先级（优先级，项目）的元素
heapq.heappush(min_priority_queue, (5, 'Process A'))
heapq.heappush(min_priority_queue, (2, 'Process B'))
heapq.heappush(min_priority_queue, (7, 'Process C'))
heapq.heappush(min_priority_queue, (3, 'Process D'))

print(f"最小堆结构: {min_priority_queue}")
# Output might look like: Min-heap structure: [(2, 'Process B'), (3, 'Process D'), (7, 'Process C'), (5, 'Process A')]
# 注意: 列表表示形式在视觉上不像树，但它维护了堆属性。

# 查看最小元素（最低优先级）
# 优先级最小的元素总是在索引 0 处
smallest_priority, smallest_item = min_priority_queue[0]
print(f"查看最小: 优先级 {smallest_priority}, 项目 {smallest_item}")
# Output: Peek smallest: Priority 2, Item Process B

# 提取最小元素
smallest_priority, smallest_item = heapq.heappop(min_priority_queue)
print(f"已提取最小: 优先级 {smallest_priority}, 项目 {smallest_item}")
# Output: Extracted smallest: Priority 2, Item Process B

print(f"提取后的堆: {min_priority_queue}")
# Output might look like: Heap after extraction: [(3, 'Process D'), (5, 'Process A'), (7, 'Process C')]

# 提取下一个最小元素
next_smallest_priority, next_smallest_item = heapq.heappop(min_priority_queue)
print(f"已提取下一个最小: 优先级 {next_smallest_priority}, 项目 {next_smallest_item}")
# Output: Extracted next smallest: Priority 3, Item Process D
```

**模拟最大优先级队列：**

由于 `heapq` 是一个最小堆实现，我们如何将其用于最大优先级队列（我们希望首先提取*最高*优先级的项目）？一种常见方法是插入优先级取反的元素。最小堆将把数值最小（即负数中最大的）优先级元素保留在根部，这对应于原始优先级最大的元素。

```python
import heapq

max_priority_queue = []

# 插入优先级取反的元素
heapq.heappush(max_priority_queue, (-5, 'High Priority Task'))
heapq.heappush(max_priority_queue, (-2, 'Low Priority Task'))
heapq.heappush(max_priority_queue, (-8, 'Urgent Task'))
heapq.heappush(max_priority_queue, (-4, 'Medium Priority Task'))

print(f"最大堆（模拟）结构: {max_priority_queue}")
# Output might look like: Max-heap (simulated) structure: [(-8, 'Urgent Task'), (-5, 'High Priority Task'), (-2, 'Low Priority Task'), (-4, 'Medium Priority Task')]

# 查看最大元素（最高优先级）
neg_priority, item = max_priority_queue[0]
print(f"查看最大: 优先级 {-neg_priority}, 项目 {item}")
# Output: Peek largest: Priority 8, Item Urgent Task

# 提取最大元素
neg_priority, item = heapq.heappop(max_priority_queue)
print(f"已提取最大: 优先级 {-neg_priority}, 项目 {item}")
# Output: Extracted largest: Priority 8, Item Urgent Task

print(f"提取后的堆: {max_priority_queue}")
# Output might look like: Heap after extraction: [(-5, 'High Priority Task'), (-4, 'Medium Priority Task'), (-2, 'Low Priority Task')]
```

记住在提取时再次对优先级取反，以获得原始值。

### 性能总结

使用堆，标准优先级队列操作实现了以下平均和最坏情况时间复杂度：

- **插入:** $O(\log n)$
- **提取最小值 / 提取最大值:** $O(\log n)$
- **查看:** $O(1)$
- **从 $n$ 个元素构建队列:** $O(n)$ (使用 `heapify` 操作)

这种效率使基于堆的优先级队列成为各种算法中的有用途径，包括 Dijkstra 和 A\* 等图搜索算法、事件驱动模拟、任务调度，以及解决选择问题，例如查找前 k 个元素，我们接下来将进行查看。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 6 (Heapsort)
  涵盖堆的基本概念、堆操作及其作为优先队列的应用，并进行了详细的算法分析。
- [heapq - Heap queue algorithm](https://docs.python.org/3/library/heapq.html) — Python Software Foundation (2024)
  Python 内置 `heapq` 模块的官方文档，详细介绍了其用于最小堆操作的函数及实际使用示例。
- [Lec 7 | Heaps and Priority Queues](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lec-7-heaps-and-priority-queues/) — Erik Demaine, Srini Devadas (2011)
  Publisher: MIT OpenCourseWare
  麻省理工学院“算法导论”课程材料，提供了堆及其在实现优先队列中的应用的学术解释。
