---
course: "data-structures-algorithms-ml"
chapter: "trees-for-ml"
lesson: "tree-fundamentals"
sourceId: 5251
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-2-trees-for-ml/tree-fundamentals"
title: "树结构的基本知识"
description: "回顾树的基本术语：节点、边、根、叶子、高度、深度以及常见树类型。"
order: 1
plots: []
sourceHash: "1573342a6314924a2b81afb974c56e00dba8e5538c7a13b0bc7770668d4bd21d"
sourceCorrections: []
---

在介绍特定基于树的算法（例如二叉搜索树或决策树）之前，我们先为树结构本身确立一套通用术语。本质上，树是由被称为**节点**的实体通过被称为**边**的连接组成的一种集合，用于表示层次关系。与数组或链表等线性结构不同，树是非线性的。

可以想象一下族谱或组织结构图；这些都是树结构在日常中的实例。在计算机科学中，树提供了一种高效的方法来存储具有内在层次结构或需要快速查找和排序的数据。

### 术语

这里是一些你在使用树时会遇到的基本术语:

- **节点:** 树的基本组成部分。它包含数据，并可能连接到其他节点。
- **边:** 两个节点之间的连接或链接。如果节点A和节点B之间存在一条边，A可能是父节点，B可能是子节点，反之亦然。
- **根节点:** 树中唯一的、最顶层的节点。它是唯一没有入边（即没有父节点）的节点。每棵树都只有一个根节点。
- **父节点:** 一个通过边连接到层次结构中其下方节点的节点。一个节点只能有一个父节点。根节点没有父节点。
- **子节点:** 一个通过边连接到其上方节点的节点。一个节点可以有零个、一个或多个子节点。
- **兄弟节点:** 共享同一个父节点的节点。
- **叶节点 (或外部节点):** 没有子节点的节点。这些是位于树最底部的节点。
- **内部节点:** 任何至少有一个子节点的节点（即，不是叶节点的任何节点）。
- **子树:** 树的一部分，由一个节点及其所有后代组成。它本身也是一棵以该节点为根的有效树。

### 树的度量: 深度、高度和层

我们经常需要描述树的大小和形态:

- **节点深度:** 从根节点到该节点的路径上的边数。根节点的深度为0。
- **节点高度:** 从该节点到其子树中任意叶节点的最长路径上的边数。叶节点的高度为0。
- **树的高度:** 其根节点的高度。这表示从根节点到任意叶节点的最长路径的长度。
- **层:** 通常与深度互换使用。相同深度的节点被认为在同一层。根节点在第0层，其子节点在第1层，以此类推。

我们来通过图示看看这些术语:

> 一张说明树术语的图示。节点A是根节点。节点B和C是A的子节点，并且是兄弟节点。节点E、F和G是叶节点（外部节点）。节点A、B、C和D是内部节点。以B为根的子树包含节点B、D、E和G。树的高度是3（最长路径: A -> B -> D -> G）。节点D的深度为2，高度为1。

### 树的类型

虽然存在许多专门的树结构（例如堆，稍后会讲解），但一种基本类型是**二叉树**。在二叉树中，每个节点最多可以有两个子节点，通常称作*左子节点*和*右子节点*。

这项约束既简单又强大。它支撑了二叉搜索树（BSTs）的构建，使得后者能够实现高效的查找、插入和删除（平均时间复杂度通常为$O(\log n)$，其中$n$是节点数量），以及机器学习 (machine learning)模型中使用的决策树。我们将在后续章节中讨论这些具体应用。

理解这种基本结构和术语，是有效使用树来组织数据和构建机器学习模型的第一步。树的层次特性允许对数据或决策进行划分，而像高度这样的属性则直接关系到树操作的潜在效率（或低效率）。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein (2022)
  Publisher: The MIT Press; Pages: Chapter 10
  计算机算法和数据结构的权威参考书，为树结构提供了详细的理论基础。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser (2013)
  Publisher: Wiley; Pages: Chapter 8
  一本实用且易懂的教科书，通过清晰的例子介绍了各种数据结构，包括不同类型的树。
- [Introduction to Algorithms (MIT OpenCourseWare 6.006)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/) — Erik Demaine, Srini Devadas (2011)
  Publisher: MIT OpenCourseWare
  提供免费课程材料，包括讲义和作业，是理解树等基本数据结构的优秀教育资源。
- [Machine Learning](http://www.cs.cmu.edu/~tom/mlbook.html) — Tom M. Mitchell (1997)
  Publisher: McGraw Hill Education; Pages: Chapter 3
  机器学习领域的经典教科书，提供决策树学习算法的基础细节，将树结构与其在预测模型中的应用联系起来。
