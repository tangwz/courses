---
course: "data-structures-algorithms-ml"
chapter: "trees-for-ml"
lesson: "tree-traversal-techniques"
sourceId: 5266
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-2-trees-for-ml/tree-traversal-techniques"
title: "树遍历技术"
description: "学习访问树中节点的各种方法，可用于模型检查或特征提取。"
order: 6
plots: []
sourceHash: "6bfa8fc0fc4df1e22aca13df1019d4c1235e5317ce251b9b7a35996ffbc90ca6"
sourceCorrections: []
---

一旦树结构建立起来，无论是用于索引的二叉搜索树还是用于分类的决策树，我们通常都需要一种系统地访问每个节点的方法。这个过程称为树遍历。不同的遍历方法以不同的顺序访问节点，使它们适用于各种任务，例如提取信息、评估树表示的表达式，或者简单地处理所有元素。

遍历树有两种主要方法：深度优先搜索 (DFS) 和广度优先搜索 (BFS)。在DFS内部，有三种常见变体。

> 一个用于演示遍历方法的简单二叉树。'L' 表示左子节点，'R' 表示右子节点。

### 深度优先搜索 (DFS) 遍历

DFS 尽可能深入一条分支，然后回溯。三种主要的DFS遍历方法在访问当前节点相对于其左右子树的顺序上有所不同。

1. **中序遍历 (左、节点、右)**

   - 访问左子树。
   - 访问当前节点。
   - 访问右子树。
   - **示例输出：** 3, 5, 7, 10, 12, 15, 18
   - **用途：** 对于二叉搜索树（如我们的例子，如果被解释为二叉搜索树），中序遍历按升序访问节点。这对于高效地获取排序数据非常有用。
2. **先序遍历 (节点、左、右)**

   - 访问当前节点。
   - 访问左子树。
   - 访问右子树。
   - **示例输出：** 10, 5, 3, 7, 15, 12, 18
   - **用途：** 先序遍历常用于创建树的副本或序列化它（保存其结构）。它也是在前缀表示法（波兰表示法）表达式中评估节点的顺序。在决策树中，从根节点遍历路径遵循相对于该特定路径中已访问节点的先序逻辑。
3. **后序遍历 (左、右、节点)**

   - 访问左子树。
   - 访问右子树。
   - 访问当前节点。
   - **示例输出：** 3, 7, 5, 12, 18, 15, 10
   - **用途：** 后序遍历通常用于删除树中的节点（需要先删除子节点，再删除父节点）或评估后缀表示法（逆波兰表示法）表达式。

DFS遍历通常递归实现，这种方法通常优雅且符合定义。它们也可以使用显式栈数据结构进行迭代实现。

### 广度优先搜索 (BFS) 遍历 (层序遍历)

BFS 逐层访问节点，从上到下，在每层中从左到右。

1. **层序遍历**
   - 访问深度为0的节点（根节点）。
   - 访问深度为1的节点。
   - 访问深度为2的节点，依此类推。
   - **示例输出：** 10, 5, 15, 3, 7, 12, 18
   - **用途：** BFS 用于需要逐层查看树的场景。在边数方面，它找到从根节点到任何其他节点的最短路径。它使用队列数据结构进行迭代实现。

### 机器学习 (machine learning)背景下的遍历

理解树遍历在机器学习中很重要，原因如下：

- **决策树预测：** 使用已训练的决策树进行预测本身就涉及树遍历。从根节点开始，根据输入特征值沿着分支前进，直到到达包含预测的叶节点。这本质上是一种特定路径的遍历，由数据引导。
- **模型检查与解释：** 遍历决策树（例如，先序或层序）可让您查看其结构、理解其分割，并可能提取人类可读的规则。您可能希望找到特定深度的所有节点（BFS）或打印到每个叶节点的规则路径（DFS）。
- **特征提取：** 在一些特定情况中，穿过树的路径或树结构本身可能被用作另一个模型的特征，需要遍历来获取这些信息。
- **二叉搜索树操作：** 如果使用二叉搜索树或平衡树来索引数据点（例如，在某些最近邻预计算方案中），中序遍历提供排序访问，而其他遍历可能用于维护。

### 实现与复杂度

在Python中实现这些遍历可以递归完成（对于DFS），也可以迭代完成（使用列表作为DFS的栈，或使用`collections.deque`作为BFS的队列）。

- **时间复杂度：** 所有标准遍历方法（中序、先序、后序、层序）都需要精确访问每个节点一次。因此，对于一个有 $n$ 个节点的树，时间复杂度是 $O(n)$。
- **空间复杂度：**
  - 对于递归DFS，空间复杂度由递归栈的最大深度决定，对应于树的高度 ($h$)。在最佳情况（平衡树）下，$h = O(\log n)$。在最坏情况（斜树）下，$h = O(n)$。
  - 对于使用栈的迭代DFS，空间复杂度也与高度相关，$O(h)$。
  - 对于使用队列的BFS，空间复杂度由任意单个级别的最大节点数（树的宽度 $w$）决定。在最坏情况（完全二叉树）下，最后一层可以包含大约 $n/2$ 个节点，导致 $O(n)$ 的空间复杂度。

选择正确的遍历方法取决于您希望通过树结构实现什么，无论是提取排序数据、理解模型结构，还是实现核心树操作。

## 参考资料

- [Introduction to Algorithms, 4th Edition](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: MIT Press
  一本广受认可的算法与数据结构教材，提供了树遍历的全面解释。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser (2013)
  Publisher: Wiley
  提供了数据结构的详细解释和Python实现，包括树遍历算法。
- [6.006 Introduction to Algorithms, Fall 2011](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/) — Erik Demaine, Srini Devadas (2011)
  Publisher: MIT OpenCourseWare
  提供教学材料，包括讲座和笔记，涵盖树遍历等基本算法和数据结构。
- [An Introduction to Statistical Learning: with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, and Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38971-1](https://doi.org/10.1007/978-3-031-38971-1)
  介绍了机器学习的统计方法，解释了决策树的结构和预测机制，这些机制本质上使用了树遍历概念。
