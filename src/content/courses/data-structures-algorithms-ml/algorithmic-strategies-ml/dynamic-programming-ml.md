---
course: "data-structures-algorithms-ml"
chapter: "algorithmic-strategies-ml"
lesson: "dynamic-programming-ml"
sourceId: 5345
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-6-algorithmic-strategies-ml/dynamic-programming-ml"
title: "动态规划原理"
description: "介绍动态规划的思想及其在序列比对或强化学习问题中的应用。"
order: 2
plots: []
sourceHash: "1c36f84c3dd6810cf4742c5e11c64c2e6d97d7e7723841de55016d7cd9318b19"
sourceCorrections: []
---

动态规划（DP）提供了一种有效的方法，用于解决可以分解为重叠子问题的问题。与子问题通常独立的分治策略不同，当计算过程中多次遇到相同的子问题时，动态规划表现出色。通过在首次解决这些子问题时存储其解，动态规划避免了重复计算，通常能带来显著的性能提升。

可以将其想象为计算斐波那契数列。朴素的递归方法会多次重新计算中间的斐波那契数。动态规划提供了一种系统方法来存储和重用这些中间结果。

### 动态规划的核心准则

两个基本特性表明一个问题可能适合采用动态规划方法：

1. **最优子结构：** 较大问题的最优解可以由其组成子问题的最优解构建而成。如果您最优地解决较小的部分，就可以组合这些结果来最优地解决整个问题。
2. **重叠子问题：** 问题的递归方法涉及重复解决相同的子问题。动态规划采用这一点，通过只计算每个子问题一次并存储其结果。

### 动态规划的实现方法

实现动态规划解法主要有两种方式：

1. **记忆化（自顶向下）：** 这种方法保持了递归解法的结构，但增加了一个机制（通常是字典或数组）来存储子问题的结果。在计算一个子问题之前，算法会检查结果是否已存储。如果已存储，则返回存储的值；否则，计算结果，存储并返回。
2. **填表法（自底向上）：** 这种方法通过迭代方式解决问题，从最小的子问题开始。它通常涉及填充一个表格（因此称为“填表法”），其中每个条目代表一个子问题的解。通过以特定顺序解决子问题，算法确保在解决某个特定子问题时，其所依赖的任何较小子问题的解都已在表格中可用。

### 例子：斐波那契数列

我们以经典的斐波那契数列来说明动态规划，其中每个数是前面两个数的和，通常从0和1开始。数列如下：0, 1, 1, 2, 3, 5, 8, 13, ... 定义为：
$F(n) = F(n-1) + F(n-2)$，其中 $F(0) = 0$ 且 $F(1) = 1$。

**朴素递归：** 直接递归实现如下：

```python
def fib_recursive(n):
    if n <= 1:
        return n
    else:
        # 分别计算 F(n-1) 和 F(n-2)
        return fib_recursive(n-1) + fib_recursive(n-2)

# 例子：计算 fib_recursive(5) 涉及多次重复调用
# fib(5) -> fib(4) + fib(3)
# fib(4) -> fib(3) + fib(2)
# fib(3) -> fib(2) + fib(1)
# ... 注意 fib(3) 和 fib(2) 被计算了多次
```

这种朴素方法具有指数时间复杂度，大致为 $O(2^n)$，因为它重复计算相同的值。例如，计算 `fib_recursive(5)` 需要计算 `fib_recursive(3)` 两次，`fib_recursive(2)` 三次，等等。

> 图表显示了斐波那契(5)递归计算中的重叠子问题。F(3)和F(2)等节点被多次展开。

**记忆化（自顶向下）：** 我们使用缓存（例如，字典）来存储结果。

```python
def fib_memoized(n, cache={}):
    if n in cache:
        return cache[n]
    if n <= 1:
        result = n
    else:
        result = fib_memoized(n-1, cache) + fib_memoized(n-2, cache)

    cache[n] = result # 返回前存储结果
    return result

# 例子：fib_memoized(5)
# 现在，fib(3) 只计算一次，然后存储并重用。
# fib(2), fib(1) 等也是如此。
```

**填表法（自底向上）：** 我们从基本情况向上构建解。

```python
def fib_tabulated(n):
    if n <= 1:
        return n

    # 初始化一个表（列表）来存储到 n 的斐波那契数
    table = [0] * (n + 1)
    table[1] = 1 # 基本情况 F(1)

    # 迭代填充表
    for i in range(2, n + 1):
        table[i] = table[i-1] + table[i-2]

    return table[n]

# 例子：fib_tabulated(5)
# table[0]=0, table[1]=1
# i=2: table[2] = table[1] + table[0] = 1 + 0 = 1
# i=3: table[3] = table[2] + table[1] = 1 + 1 = 2
# i=4: table[4] = table[3] + table[2] = 2 + 1 = 3
# i=5: table[5] = table[4] + table[3] = 3 + 2 = 5
```

记忆化和填表法都将时间复杂度降至线性，为 $O(n)$，因为到 $n$ 的每个斐波那契数都只计算一次。缓存或表的空间复杂度也是 $O(n)$（尽管填表版本可以通过只记录最后两个值来优化到使用 $O(1)$ 空间）。

### 动态规划在机器学习 (machine learning)中的应用

尽管您可能不会直接使用动态规划从头实现线性回归或支持向量 (vector)机等算法，但其基本原理出现在机器学习的几个重要方面：

- **强化学习 (reinforcement learning)（RL）：** 许多基础强化学习算法，例如价值迭代和策略迭代，都非常依赖动态规划。它们根据后续状态的值迭代更新状态（或状态-动作对）的估计值，旨在找到最优策略。这个过程本身采用最优子结构（一个状态的最优策略取决于后续状态的最优策略），并处理重叠子问题（状态会被再次访问）。
- **序列比对和编辑距离：** 像Needleman-Wunsch（全局比对）和Smith-Waterman（局部比对）这样的算法使用动态规划来找到两个序列（例如，DNA、蛋白质或文本）之间的最佳比对。计算字符串之间的编辑距离（例如，Levenshtein距离），在自然语言处理中用于拼写校正或相似度度量，也是一个经典的动态规划问题。这些算法构建一个表，其中每个单元格 $(i, j)$ 存储了对于直到索引 $i$ 和 $j$ 的子序列的最优分数/距离。
- **隐马尔可夫模型（HMM）：** 维特比算法用于找到在隐马尔可夫模型中生成给定观察序列的最可能的隐藏状态序列，它是动态规划的一个典型示例。该算法广泛用于语音识别、生物信息学和自然语言处理（例如，词性标注）。
- **最优控制和规划：** 涉及随时间寻找最优动作序列或决策的问题，有时可以用动态规划来建模和解决。

识别问题何时显示最优子结构和重叠子问题，可以帮助您理解为什么某些机器学习算法（尤其是在强化学习和序列建模中）以它们的方式构建，并理解它们的计算特性。尽管库通常会抽象这些实现，但理解动态规划的根本对于分析以及有时根据特定需求定制算法很有价值。

## 参考资料

- [Introduction to Algorithms, Fourth Edition](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press; DOI: [10.7551/mitpress/14068.001.0001](https://doi.org/10.7551/mitpress/14068.001.0001)
  一本基础教材，全面阐述了动态规划原理和算法。
- [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/book/the-book-2nd.html) — Richard S. Sutton, Andrew G. Barto (2018)
  Publisher: MIT Press
  本书详细介绍了动态规划如何作为解决强化学习中最优控制问题的基础。
- [Design and Analysis of Algorithms (6.046J)](https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/) — Erik Demaine, Srini Devadas, Nancy Lynch (2015)
  Publisher: MIT OpenCourseWare
  一门高级算法课程，提供深入讲解动态规划的讲座视频和材料。
- [Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids](https://www.cambridge.org/core/books/biological-sequence-analysis/9780521629713) — Richard Durbin, Sean R. Eddy, Anders Krogh, Graeme Mitchison (1998)
  Publisher: Cambridge University Press
  一本经典著作，将动态规划应用于序列比对和隐马尔可夫模型（例如维特比算法）。
