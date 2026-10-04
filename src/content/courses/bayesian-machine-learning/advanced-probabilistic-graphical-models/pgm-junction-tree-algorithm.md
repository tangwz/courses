---
course: "bayesian-machine-learning"
chapter: "advanced-probabilistic-graphical-models"
lesson: "pgm-junction-tree-algorithm"
sourceId: 3614
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-5-advanced-probabilistic-graphical-models/pgm-junction-tree-algorithm"
title: "概率图模型中的高级推断：联结树算法"
description: "了解联结树算法在复杂图模型中进行精确推断的方法。"
order: 3
plots: []
sourceHash: "9afe56a32ebb292333ed0e50cd7e81de6adefdbc559e3ea1bde4c74c2d85deb4"
sourceCorrections: []
---

虽然像马尔可夫链蒙特卡洛（MCMC）和变分推断这样的近似方法是处理大型或复杂概率图模型（PGMs）的必要工具，但在某些情况下，精确推断既合乎期望又在计算上可行。联结树算法（也称为团树算法或连接树算法）是执行精确推断的一种主要方法，特别是用于计算单连接或中等复杂多连接图中变量的边际概率。它巧妙地将原始图转换为由变量簇（团）构成的树状结构，从而使信念传播的消息传递得以高效进行。

该算法建立在变量消去的原理之上，但它组织计算的方式可以避免冗余，尤其是在需要多个边际查询时。其核心思想是凭借图的结构来高效执行计算。

### 从图到树：三角化与团识别

构建联结树始于图的准备。如果从贝叶斯网络等有向图开始，第一步是**道德化**。这涉及通过在每个节点的父节点之间添加无向边来“连接”它们，然后将所有原始有向边变为无向。此过程生成道德图，这是一个捕获与推断有关的条件依赖关系的无向表示。

下一个重要步骤是**三角化**（或使图成为弦图）。如果一个无向图中每个长度为四或更长的环都有一条弦（连接环中两个不相邻节点的边），那么该无向图是三角化的。三角化涉及向道德图添加边以消除无弦环。尽管找到最优的三角化（最小化所得团大小的三角化）是NP难问题，但实践中通常使用启发式方法。为什么要进行三角化？因为三角化图有一个特性：它们的极大团（节点子集，其中每对节点都连接，且不属于更大的团）可以排列成满足*运行交集性质*的树状结构。

一旦图被三角化，我们识别其所有**极大团**。这些团将成为我们联结树的节点。

### 构建联结树

联结树由三角化图的极大团构建而成。联结树的节点是极大团 $C_1, C_2, \dots, C_k$。如果团 $C_i$ 和 $C_j$ 共享变量，则在它们之间添加边。共享变量集 $S_{ij} = C_i \cap C_j$ 称为**分离器**。

有效联结树的要求是**运行交集性质**（或联结树性质）：对于树中任意两个团 $C_i$ 和 $C_j$，它们之间唯一路径上的所有团都必须包含交集 $C_i \cap C_j$。此性质保证关于共享变量的信息在树中一致流动。构建满足此性质的树通常可以使用最大生成树算法在图上完成，其中图的节点是团，边权重 (weight)是分离器 $|C_i \cap C_j|$ 的大小。

> 联结树的一个简单例子。节点表示三角化图中的极大团。边连接共享变量的团，并指明了共享变量（分离器）。这种树状结构满足运行交集性质。

### 消息传递的信念传播

联结树构建完成后，推断通过消息传递进行。每个团 $C_i$ 最初都与一个**势函数** $\psi_i$ 相关联。此势通常通过将原始概率图模型中变量完全包含在 $C_i$ 内的条件概率表（CPTs）或因子相乘而形成。仔细分配可保证每个原始因子都恰好分配给一个团。

该算法包括两个阶段：

1. **收集证据（向上/向内传递）：** 选择任意一个团作为根节点。消息从树的叶子节点向根节点传递。团 $C_j$ 仅在其收到所有*其他*邻居的消息之后，才向其邻居 $C_i$ 发送消息。
2. **分发证据（向下/向外传递）：** 一旦根节点收到所有邻居的消息，消息将从根节点向下传回叶子节点。团 $C_i$ 仅在其收到来自其父节点（在收集阶段它接收消息的那个邻居）的消息之后，才向其邻居 $C_j$ 发送消息。

**什么是消息？** 从团 $C_j$ 发送给团 $C_i$ 的消息 $m_{ji}$ 基本是对以 $C_j$ 为根的子树（从 $C_i$ 的角度看）中收集到的信息（信念或势）的总结，并边际化到分离器 $S_{ij} = C_i \cap C_j$ 中的变量上。

**计算消息：** 令 $\beta_j$ 为与团 $C_j$ 相关联的当前势（最初是 $\psi_j$，然后由传入消息更新）。从 $C_j$ 发送给 $C_i$ 的消息 $m_{ji}$ 计算如下：


$$
m_{ji}(S_{ij}) = \sum_{C_j \setminus S_{ij}} \beta_j^*(C_j)
$$


其中 $\beta_j^*$ 表示势 $\beta_j$，可能包含 $C_j$ 在当前传递过程中从 $C_i$ 以外的邻居接收到的消息。具体而言，在收集阶段，当 $C_j$ 发送给 $C_i$ 时，$\beta_j^*$ 包含 $\psi_j$ 以及来自所有邻居 $k \in N(j) \setminus \{i\}$ 的消息。求和运算边际化掉仅存在于 $C_j$ 但不在分离器 $S_{ij}$ 中的变量。

当团 $C_i$ *收到*消息 $m_{ji}(S_{ij})$ 时，它会更新自己的势：


$$
\beta_i(C_i) \leftarrow \beta_i(C_i) \cdot m_{ji}(S_{ij})
$$


请注意，消息 $m_{ji}$ 仅是分离器变量 $S_{ij}$ 的函数，但它被乘入团 $C_i$ 中所有变量的函数势 $\beta_i$ 中。此操作有时被称为“势乘法”。

**最终信念：** 在收集和分发两个阶段完成后，与每个团 $C_i$ 相关联的势 $\beta_i(C_i)$ 与该团中变量的真实联合边际概率分布成比例，考虑到初始势中包含的任何证据：


$$
\beta_i(C_i) \propto P(C_i | \text{evidence})
$$


根据这些最终的团信念，我们可以通过对 $\beta_i(C_i)$ 中剩余变量求和来轻松计算团 $C_i$ 中任意单个变量 $X \in C_i$ 或变量子集的边际概率。运行交集性质保证从包含相同变量的不同团计算出的边际将是一致的。

### 计算复杂度与适用性

联结树算法的效率主要取决于联结树中最大团的大小。最大团的大小被称为图的**树宽**（更准确地说，原始图的树宽是在所有可能的三角化中，最大团的最小大小减一）。

消息传递的计算成本涉及对团上定义的势进行运算（乘法、求和）。如果最大团大小（树宽 + 1）为 $w_{max}$，并且变量最多有 $d$ 个状态，则复杂度大致呈 $w_{max}$ 的指数级（例如，$O(k \cdot d^{w_{max}})$，其中 $k$ 是团的数量）。

因此，联结树算法对于可以找到具有小极大团的图（即树宽较低的图）是高效的。对于树宽较高的图（如密集网格或高度连接的网络），团会变得过大，计算将变得难以处理。这时，近似推断方法就变得必要了。

### 优点与缺点

**优点：**

- 提供所有变量和团的**精确**边际概率。
- 对于树宽较低的图是高效的。
- 在初始消息传递阶段完成后，通过边际化最终的团信念，可以快速回答多个边际查询。
- 为理解推断复杂度提供理论支持。

**缺点：**

- 找到最优三角化是NP难问题。
- 联结树本身的构建可能很复杂。
- 对于树宽较高的图，计算上变得**难以处理**。
- 主要为离散变量设计，尽管存在高斯图模型的扩展。

总而言之，联结树算法代表着概率图模型中进行精确推断的一个重要的理论和实践工具。它将潜在复杂图上的推断问题转换为树上更结构化的消息传递过程。其适用性受图的结构属性支配，特别是其树宽，这强调了图结构与推断复杂性之间的密切关联。当树宽可控时，它提供了一种有效方式来获取精确的概率查询。当树宽不可控时，我们则转向其他章节讨论的近似方法。

## 参考资料

- [Probabilistic Graphical Models: Principles and Techniques](https://mitpress.mit.edu/books/probabilistic-graphical-models) — Daphne Koller and Nir Friedman (2009)
  Publisher: The MIT Press
  提供对概率图模型理论的全面涵盖，包括对连接树算法的详细解释。
- [Bayesian Networks and Decision Graphs](https://link.springer.com/book/10.1007/978-0-387-68282-2) — Finn V. Jensen, Thomas D. Nielsen (2007)
  Publisher: Springer; DOI: [10.1007/978-0-387-68282-2](https://doi.org/10.1007/978-0-387-68282-2)
  一本经典教材，对贝叶斯网络中的精确推理（包括连接树算法）提供了全面的论述。
- [CS228: Probabilistic Graphical Models (Autumn 2023-2024 Course Materials - Lecture 10: Junction Tree Algorithm)](https://ermongroup.github.io/cs228-notes/inference/jt/) — Volodymyr Kuleshov, Stefano Ermon (2022)
  提供作为一门备受推崇的概率图模型课程内容之一的连接树算法的清晰解释和算法细节。
- [Local Computations with Probabilities on Graphical Structures and their Application to Expert Systems](http://www.jstor.org/stable/2345762) — Steffen L. Lauritzen and David J. Spiegelhalter (1988)
  Journal: Journal of the Royal Statistical Society, Series B (Methodological); Publisher: Blackwell Publishing for the Royal Statistical Society; Volume: 50; Pages: 157-224; DOI: [10.1111/j.2517-6161.1988.tb01720.x](https://doi.org/10.1111/j.2517-6161.1988.tb01720.x)
  介绍图模型上局部计算的基础论文，局部计算是连接树算法进行精确推理的基础。
