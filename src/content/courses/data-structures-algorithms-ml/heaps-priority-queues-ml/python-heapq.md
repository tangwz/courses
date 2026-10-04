---
course: "data-structures-algorithms-ml"
chapter: "heaps-priority-queues-ml"
lesson: "python-heapq"
sourceId: 5334
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-5-heaps-priority-queues-ml/python-heapq"
title: "Python 的 heapq 模块"
description: "使用 Python 内置的 heapq 库进行高效的堆和优先队列操作。"
order: 6
plots: []
sourceHash: "492961f312fe7636bee9598fb1d409a9adb42bedbd88dcd02944887190406af5"
sourceCorrections: []
---

堆是必不可少的数据结构，但从头高效实现它们可能会很复杂。幸运的是，Python 的标准库提供了 `heapq` 模块，它提供了堆队列算法（也称为优先队列算法）的高度优化实现。

`heapq` 模块直接使用标准的 Python 列表，将其视作最小堆。这意味着最小的元素总是位于根部（索引 0）。我们来看看它的主要功能。

### 堆的基本操作

要使用 `heapq` 函数，您首先需要导入该模块：

```python
import heapq
```

然后您就可以对一个标准的 Python 列表进行操作了。

#### 添加元素：`heapq.heappush`

要在添加元素到堆的同时保持堆的属性，请使用 `heapq.heappush(heap, item)`。该函数会就地修改列表。

```python
# 从一个空列表开始（我们的堆）
min_heap = []

# 向堆中添加元素
heapq.heappush(min_heap, 50)
heapq.heappush(min_heap, 20)
heapq.heappush(min_heap, 80)
heapq.heappush(min_heap, 10)
heapq.heappush(min_heap, 30)

print(f"Heap after pushes: {min_heap}")
# 预期输出（顺序可能因实现细节而略有不同，
# 但 10 将是第一个元素）：
# Heap after pushes: [10, 20, 80, 50, 30]
```

在内部，`heappush` 将元素添加到列表末尾，然后将其“上浮”到正确位置以保持最小堆属性。此操作通常需要 $O(\log n)$ 时间，其中 $n$ 是堆中元素的数量。

> 一个表示最小堆 `[10, 20, 80, 50, 30]` 的图示。根元素（10）是最小的元素。

#### 移除最小元素：`heapq.heappop`

要从堆中移除并返回最小元素（根元素），请使用 `heapq.heappop(heap)`。这也会就地修改列表并保持堆的属性。

```python
# 沿用上一个示例
smallest_item = heapq.heappop(min_heap)

print(f"Popped item: {smallest_item}")
print(f"Heap after pop: {min_heap}")
# 预期输出：
# Popped item: 10
# Heap after pop: [20, 30, 80, 50]
```

`heappop` 将根元素替换为列表中的最后一个元素，然后将其“下沉”以恢复堆属性。与 `heappush` 类似，此操作通常需要 $O(\log n)$ 时间。

> 弹出最小元素（10）后的堆结构。20 现在是根元素。

#### 访问最小元素

由于 `heapq` 使用的列表中 `heap[0]` 总是最小元素，您只需访问索引 0 的元素，即可查看最小值而不将其移除。请记住，首先要检查堆是否为空。

```python
if min_heap:
    print(f"Smallest item currently in heap: {min_heap[0]}")
else:
    print("Heap is empty.")
# 预期输出：
# Smallest item currently in heap: 20
```

### 高效构建堆：`heapq.heapify`

如果您已经有一个项目列表，并希望将其转换为一个有效的堆，您可以重复调用 `heappush`。然而，一个更高效的方法是 `heapq.heapify(list)`。该函数会*就地*重新排列列表元素以满足堆属性。

```python
data = [50, 20, 80, 10, 30, 5, 90, 45]
print(f"Original list: {data}")

heapq.heapify(data) # 将列表就地转换为堆

print(f"Heapified list: {data}")
print(f"Smallest item: {data[0]}")
# 预期输出（堆化列表的顺序可能不同，最小的元素在最前面）：
# Original list: [50, 20, 80, 10, 30, 5, 90, 45]
# Heapified list: [5, 10, 80, 20, 30, 50, 90, 45] # 示例输出
# Smallest item: 5
```

`heapify` 的优点是其时间复杂度。它在线性时间 $O(n)$ 内运行，通常比使用 `heappush` 一个接一个地插入 $n$ 个元素更快，后者总共需要 $O(n \log n)$ 时间。

### 组合操作

`heapq` 还提供了高效地组合推入和弹出操作的函数。

- `heapq.heappushpop(heap, item)`: 将 `item` 推入 `heap`，然后弹出并返回最小的元素。这比分别调用 `heappush` 后再调用 `heappop` 更高效。这对于维护固定大小的堆很有用（例如，始终保留 'k' 个最大元素）。
- `heapq.heapreplace(heap, item)`: 从 `heap` 中弹出并返回最小的元素，然后推入新的 `item`。它与 `heappushpop` 不同之处在于它是*先*弹出。这可能会稍微更高效一些，但如果堆最初为空，则会引发 `IndexError`。

```python
# 使用 heappushpop 的示例（维护前 3 个元素）
top_k_heap = [10, 20, 30] # 假设这已经是一个堆
heapq.heapify(top_k_heap) # 确保它是一个堆

# 处理新项目，只保留最大的 3 个（使用最小堆来保存最小元素）
# 如果我们将其建模为使用大小为 'k' 的最小堆来保留 'k' 个最大元素，
# 当堆大小超过 'k' 时，我们推入一个新项目并弹出最小的元素。
# 如果堆已经达到大小 'k'，heappushpop 很有用。

new_item = 5
if len(top_k_heap) < 3:
    heapq.heappush(top_k_heap, new_item)
elif new_item > top_k_heap[0]: # 仅当大于堆中最小元素时才考虑
    smallest = heapq.heappushpop(top_k_heap, new_item)
    # smallest 变量现在包含被推出（即被替换掉）的元素（本例中为 10）

print(f"Heap after considering {new_item}: {top_k_heap}") # 输出: [20, 30, 5] -> 需要堆化逻辑

# 使用 heapreplace 的示例
data_heap = [10, 20, 30]
heapq.heapify(data_heap)

smallest = heapq.heapreplace(data_heap, 5) # 将 10 替换为 5，返回 10
print(f"Replaced item: {smallest}")
print(f"Heap after replace: {data_heap}")
# 预期输出：
# Replaced item: 10
# Heap after replace: [5, 20, 30] # 5 被推入，10 被弹出。堆属性得到维护。
```

### 实现最大堆

由于 `heapq` 只直接实现了最小堆，如果您需要一个最大堆（其中最大元素始终位于根部）怎么办？常用方法包括修改堆中存储的值：

1. **取负值：** 存储每个数值的负值。最小的负值对应于最大的原始值。弹出时，将结果再取回负值。
2. **使用元组：** 存储元组，其中第一个元素是优先级取负值（或乘以 -1），第二个元素是实际项目。Python 的元组比较按字典顺序比较元素，因此它将根据第一个元素（取负的优先级）进行排序。

```python
# 方法 1：对值取负以实现最大堆
max_heap_negated = []
data = [10, 50, 20, 80, 30]

for item in data:
    heapq.heappush(max_heap_negated, -item) # 推入负值

print(f"Internal min-heap (negated): {max_heap_negated}")

largest_item = -heapq.heappop(max_heap_negated) # 弹出最小的负值，再取回负值
print(f"Largest item popped: {largest_item}") # 应该是 80
print(f"Internal heap after pop: {max_heap_negated}")

# 方法 2：使用元组（优先级，项目）实现最大堆
max_heap_tuples = []
data_with_ids = [('task1', 10), ('task2', 50), ('task3', 20)]

for task_id, priority in data_with_ids:
    heapq.heappush(max_heap_tuples, (-priority, task_id)) # 推入 (-优先级, 项目)

print(f"Internal min-heap (tuples): {max_heap_tuples}")

neg_priority, highest_priority_task = heapq.heappop(max_heap_tuples)
print(f"Highest priority task popped: {highest_priority_task} (Priority: {-neg_priority})") # 任务2, 50
print(f"Internal heap after pop: {max_heap_tuples}")
```

### 选择问题的实用工具

`heapq` 还提供了便捷函数 `nlargest(k, iterable)` 和 `nsmallest(k, iterable)`，它们能高效地从任何可迭代对象中找到 $k$ 个最大或最小的项目。这些函数内部通常使用堆，尤其当 $k$ 相对于项目总数 $n$ 较小时。

```python
scores = [34, 78, 12, 45, 99, 67, 23, 88, 50]

top_3 = heapq.nlargest(3, scores)
bottom_3 = heapq.nsmallest(3, scores)

print(f"Scores: {scores}")
print(f"Top 3 scores: {top_3}")     # 输出: [99, 88, 78]
print(f"Bottom 3 scores: {bottom_3}") # 输出: [12, 23, 34]
```

使用 `heapq` 可以让您在机器学习 (machine learning)工作流中使用高效的内置实现来处理优先队列和堆相关任务，例如在集束搜索中管理候选者、查找最近邻（与其他结构结合时），或实现选择算法。其性能（推入/弹出为 $O(\log n)$，堆化为 $O(n)$）使其适合处理对效率有要求的海量数据。

## 参考资料

- [heapq - Heap queue algorithm](https://docs.python.org/3/library/heapq.html) — Python core developers (2024)
  提供 Python heapq 模块的官方文档和示例。
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press; Pages: Chapter 6
  一本标准算法教材，介绍了堆和优先队列的理论基础。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons; Pages: Chapter 9
  提供数据结构和算法的 Python 示例，包含详细的堆讨论。
