---
course: "data-structures-algorithms-ml"
chapter: "heaps-priority-queues-ml"
lesson: "practice-heap-selection"
sourceId: 5338
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-5-heaps-priority-queues-ml/practice-heap-selection"
title: "实践：使用堆进行选择"
description: "使用 Python 的 heapq 模块动手实现选择算法。"
order: 7
plots: []
sourceHash: "acf309e11fd94cea6c469a33df99bb0dd8bb3c75fa4b9421decd18bddb994853"
sourceCorrections: []
---

堆和优先队列用于处理一个常见问题：从数据集中选出前 *k* 个元素，而无需对整个数据集进行排序。这是一项常遇到的任务，以多种形式出现，比如查找最常用的 *k* 个词、得分最高的 *k* 个预测，甚至作为更复杂算法的子程序。

假设你有一大串传感器读数，你只需要跟踪迄今为止最高的 5 个温度。每次新读数到达时都对所有读数进行排序会非常低效。堆提供了一种更好的方法。

### 查找 K 个最大元素

主要思路是维护一个迄今为止遇到的 *k* 个最大元素的集合。最小堆出乎意料地适合这个任务。为什么是*最小*堆？因为它能让我们高效地找到并移除当前前 *k* 个元素中*最小*的那个。

算法如下：

1. **初始化最小堆：** 创建一个空的最小堆。我们将使用此堆来存储已见的 *k* 个最大元素。
2. **处理元素：** 遍历输入数据中的每个元素。
3. **填充堆：** 如果堆当前包含少于 *k* 个元素，只需使用 `heappush` 将当前元素添加到堆中。
4. **比较和替换：** 如果堆已经包含 *k* 个元素，将当前元素与堆中*最小*的元素（在最小堆中总是位于根部，即索引 0）进行比较。
   - 如果当前元素*大于*堆的根部元素，这意味着这个新元素属于前 *k* 个最大元素的集合。使用 `heappop` 移除最小元素（根部），并使用 `heappush` 添加当前元素。`heapq` 模块提供了一个方便的 `heapreplace` 函数，可以高效地合并这两个步骤。
   - 如果当前元素小于或等于堆的根部元素，它不属于前 *k* 个，因此你可以忽略它并处理下一个元素。
5. **结果：** 处理完输入数据中的所有元素后，最小堆将包含 *k* 个最大元素。堆的根部（堆中最小的值）将是总体上第 *k* 大的元素。

**复杂度：** 对于输入中的每个 *n* 元素，我们最多执行一次堆操作（推入或替换），这需要 $O(\log k)$ 时间，因为堆的大小受 *k* 限制。因此，总时间复杂度为 $O(n \log k)$。这比对整个数据集进行排序（$O(n \log n)$）要好得多，尤其当 *k* 远小于 *n* 时。空间复杂度为 $O(k)$，用于存储堆。

**Python 实现 (`heapq`)**

我们来使用 Python 的 `heapq` 模块查找列表中 3 个最大的元素。请记住，`heapq` 实现的是最小堆。

```python
import heapq

def find_k_largest(data, k):
    """使用最小堆查找数据中 k 个最大的元素。"""
    min_heap = [] # 我们的最小堆，用于存储 k 个最大元素
    for element in data:
        if len(min_heap) < k:
            # 堆还没满，直接添加元素
            heapq.heappush(min_heap, element)
        else:
            # 堆已满，将当前元素与堆中最小的元素（根部）进行比较
            # 如果当前元素更大，则替换掉最小的元素
            if element > min_heap[0]:
                # 高效地弹出根部并推入新元素，同时保持堆的性质
                heapq.heapreplace(min_heap, element) 

    # 堆现在包含 k 个最大元素
    # min_heap[0] 是第 k 大的元素
    # 返回排序后的列表，仅为清晰展示前 k 个元素
    return sorted(min_heap, reverse=True) 

# 示例用法
scores = [34, 56, 12, 89, 67, 99, 45, 78, 23, 91]
k = 3
top_k = find_k_largest(scores, k)
print(f"最大的 {k} 个分数是: {top_k}") 
# 预期输出：最大的 3 个分数是: [99, 91, 89]

# 如果你只需要循环后的第 k 大元素：
# kth_largest = min_heap[0] 
# print(f"第 {k} 大的分数是: {kth_largest}") # 应该是 89
```

### 查找 K 个最小元素

查找 *k* 个最小元素遵循类似的思路，但这次我们使用*最大*堆。最大堆将存储迄今为止遇到的 *k* 个最小元素。当考虑一个新元素时，我们将其与堆中当前*最大*的元素（最大堆的根部）进行比较。如果新元素更小，它将替换堆中当前最大的元素。

由于 Python 的 `heapq` 只提供了最小堆的实现，我们使用一种常见技巧：在最小堆中存储元素的*负值*。这实际上将最小堆变成了关于原始值的最大堆。

1. **初始化最小堆（用于负值）：** 创建一个空的最小堆。
2. **处理元素：** 遍历每个元素。
3. **填充堆：** 如果堆大小小于 *k*，将*负值*元素（`-element`）推入堆中。
4. **比较和替换：** 如果堆有 *k* 个元素，将*负值*的当前元素（`-element`）与堆的根部（这是最小的负值，对应于 k 个最小元素中最大的原始值）进行比较。
   - 如果 `-element` *大于*堆的根部（`min_heap[0]`），这意味着原始的 `element` *小于*当前存储在我们 k 个最小元素集合中的最大值。使用 `heapreplace` 将根部替换为 `-element`。
   - 否则，忽略当前元素。
5. **结果：** 处理完所有数据后，堆包含 *k* 个最小元素的负值形式。再次取反即可得到实际的最小元素。

**Python 实现（带负值处理的 `heapq`）**

我们来使用这种技巧查找 4 个最小元素。

```python
import heapq

def find_k_smallest(data, k):
    """使用模拟的最大堆查找数据中 k 个最小的元素。"""
    # 使用存储负值的最小堆来模拟最大堆
    max_heap = [] 
    for element in data:
        neg_element = -element # 对元素取负
        if len(max_heap) < k:
            heapq.heappush(max_heap, neg_element)
        else:
            # 将负值元素与根部（最小的负值 = 最大的原始值）进行比较
            # 如果 neg_element > 根部，则 element < 堆中最大值
            if neg_element > max_heap[0]: 
                heapq.heapreplace(max_heap, neg_element)

    # 堆中包含 k 个最小元素的负值。取反回来。
    # 取反回来并排序，以便清晰展示底部 k 个元素
    smallest_k = sorted([-val for val in max_heap]) 
    return smallest_k

# 示例用法
temps = [2.5, -1.0, 0.5, 3.0, -2.2, 1.8, 0.0, -0.5]
k = 4
bottom_k = find_k_smallest(temps, k)
print(f"最小的 {k} 个温度是: {bottom_k}")
# 预期输出：最小的 4 个温度是: [-2.2, -1.0, -0.5, 0.0]

# 循环后，第 k 小的元素是 -max_heap[0]
# 在此示例中，kth_smallest = -max_heap[0] 将是 0.0（因为 max_heap[0] 是 -0.0）
```

### 便捷函数：`heapq.nlargest` 和 `heapq.nsmallest`

Python 的 `heapq` 模块为这些常见任务提供了直接函数：`heapq.nlargest(k, iterable)` 和 `heapq.nsmallest(k, iterable)`。这些函数内部实现了基于堆的选择算法，并提供了更简洁的接口。对于大多数实际应用，建议使用这些内置函数。

```python
import heapq

data = [34, 56, 12, 89, 67, 99, 45, 78, 23, 91]
k = 3

largest_k = heapq.nlargest(k, data)
smallest_k = heapq.nsmallest(k, data)

print(f"Using nlargest: {largest_k}")   # 输出：使用 nlargest: [99, 91, 89]
print(f"Using nsmallest: {smallest_k}") # 输出：使用 nsmallest: [12, 23, 34] 
```

虽然自己实现逻辑有助于理解堆如何实现高效性，但在实际中，为简洁起见并可能获得更好的性能（因为有优化的 C 语言实现），应使用 `nlargest` 和 `nsmallest`。

### 机器学习 (machine learning)中的应用场景

在机器学习中，你可能会在哪些地方使用 k 选择技术？

- **特征选择：** 找出重要性评分最高的前 *k* 个特征（例如，通过训练好的树集成模型，如随机森林或梯度提升计算得出）。这有助于理解模型行为或进行特征降维。
- **近邻（简化）：** 在基本的 k-近邻 (k-NN) 实现中，计算查询点到所有 *n* 个训练点的距离后，你需要找出距离最小的 *k* 个点。与对所有距离进行排序（$O(n \log n)$）相比，堆提供了一种高效的方法（$O(n \log k)$）来找到这 *k* 个邻居。请注意，专门的数据结构（如 KD 树或球树，稍后会讨论）通常用于在高维度中更快地进行*近似*近邻搜索，但堆适用于中等数据集上的精确 k-NN 或作为其他基于距离算法的一部分。
- **推荐系统：** 根据模型生成的预测相关性分数，为用户选择排名前 *k* 的推荐项目。
- **异常检测：** 找出被检测算法标记 (token)为异常分数最高的排名前 *k* 个数据点，从而将注意力集中在最不寻常的观测值上。
- **集束搜索：** 在自然语言处理中常见的序列生成任务（如机器翻译或文本摘要）中，集束搜索在生成过程的每一步维护一个包含 *k* 个最有可能的候选序列的集合（集束）。优先队列（使用堆实现）是高效管理此集束的理想选择。

理解如何使用堆高效地选择顶部或底部 *k* 个元素是一项有用的技能。它让你能够优化机器学习管道中涉及排序或选择的部分，避免在只需要极端值子集时进行完整排序的不必要开销。这项实践进一步说明了为当前任务选择正确数据结构所带来的效率提升。

## 参考资料

- Introduction to Algorithms — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press
  算法和数据结构的权威教材，涵盖堆、优先队列和选择问题。
- [heapq - Heap queue algorithm](https://docs.python.org/3/library/heapq.html) — Python Software Foundation (2024)
  Python 内置最小堆实现以及 `nlargest` 和 `nsmallest` 等函数的官方文档。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons
  讲解堆和优先队列的教材，包含 Python 实现和实际示例。
