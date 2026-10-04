---
course: "compiler-runtime-optimization-ml"
chapter: "tensor-level-polyhedral-optimizations"
lesson: "polyhedral-scheduling-transformations"
sourceId: 4500
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations/polyhedral-scheduling-transformations"
title: "调度变换 (倾斜变换, 分块)"
description: "利用多面体调度器实现先进的循环变换，以提升并行性和局部性。"
order: 4
plots: []
sourceHash: "6c1680304d475ab7e09c0e601a7b0c462017f90edc6c79c5dfe4a49d93a21b84"
sourceCorrections: []
---

在多面体模型中表示张量操作及其数据依赖，能够系统地改变循环迭代的执行顺序。这种重排，称为调度，是由作用于迭代向量 (vector)的数学变换指导的。主要目标是改善数据局部性，提升缓存利用率，并显现可供多核CPU或GPU利用的并行性。

调度会为循环嵌套中语句的每个动态实例分配一个逻辑执行时间戳（通常是多维的）。对于在迭代向量$\vec{x}$处执行的语句实例$S_i$，其调度表示为$T(S_i, \vec{x})$。一个基本要求是调度必须遵守所有数据依赖。如果语句实例$(S_i, \vec{x})$因数据依赖（写作$(S_i, \vec{x}) \rightarrow (S_j, \vec{y})$）而必须在$(S_j, \vec{y})$之前执行，那么调度必须保证它们的逻辑时间戳能体现这个顺序：
$T(S_i, \vec{x}) \prec T(S_j, \vec{y})$
此处，$\prec$表示字典序。找到一个满足所有依赖并同时优化局部性或并行性的调度$T$，是核心工作。在多面体模型中，调度通常被限制为循环迭代器($\vec{x}$)和循环边界或访问中存在的任何符号常量（参数 (parameter)）的*仿射函数*。

两种基本且广泛使用的调度仿射变换是分块和倾斜变换。

### 循环分块 (阻塞)

分块，也称为阻塞，可以说是现代内存层级结构中对性能影响最显著的变换。它将迭代空间划分为更小、规则的块（瓦片），这些块以原子方式执行。

**动机**：在瓦片内执行迭代，能将数据复用置于首位。如果一个瓦片访问总体数据的一小部分，这些数据更有可能适合内存层级结构中更快的级别（寄存器、L1/L2缓存）。这提升了*时间局部性*（复用相同数据元素）和*空间局部性*（访问附近数据元素）。此外，如果依赖允许，瓦片本身通常可以并行执行，显现粗粒度并行性。以矩阵乘法$C = A \times B$为例。分块将大的乘法分解为更小的矩阵乘法，这些乘法作用于A、B和C的块，这些块更适合缓存。

**多面体分块**：在多面体框架中，通过将原始迭代向量 (vector)$\vec{x}$转换为更高维的向量来实现分块，该向量将瓦片坐标与瓦片内坐标分离。对于具有迭代器$(i, j)$和方形瓦片大小$B$的二维循环嵌套，一种常见的分块变换将$(i, j)$映射到$(ii, jj, i', j')$，其中：

- $ii = \lfloor i / B \rfloor$ 和 $jj = \lfloor j / B \rfloor$ 是瓦片坐标。
- $i' = i \pmod B$ 和 $j' = j \pmod B$ 是瓦片内的点坐标。

结果调度可能看起来像$T(i, j) = (ii, jj, i', j')$。这将原始的2层循环嵌套转换为4层循环嵌套：两个外层循环迭代瓦片（$ii, jj$），两个内层循环迭代瓦片内的点（$i', j'$）。

> 将4x4迭代空间划分为2x2瓦片。像D1这样的依赖包含在一个瓦片内（如果瓦片大小足够大），或者跨越瓦片边界。像D2这样的依赖跨越瓦片边界（瓦片间依赖）。

**合法性**：只有当所选调度（包括瓦片循环和点循环）遵守所有原始数据依赖时，分块才是合法的。依赖可以分为：

- **瓦片内依赖**：依赖的源和目的地位于同一瓦片内。这些依赖通过瓦片内的执行顺序（$i', j'$循环）处理。
- **瓦片间依赖**：源和目的地位于不同的瓦片中。瓦片循环（$ii, jj$）的调度必须遵守这些依赖。

为了使分块能够实现瓦片间的并行（例如，并行执行$ii$循环），调度中该循环维度不得带有任何依赖。像Pluto调度器这样的算法旨在找到多维仿射调度，以实现合法且高效的分块，通常会最大化可以合法并行化的维度。

### 循环倾斜变换

Loop skewing是另一种仿射变换，常用于*使能*其他变换，如分块或向量 (vector)化，当依赖原本会阻止这些变换时。

**动机**：考虑一个循环嵌套，其依赖结构形成“波前”模式，这在模板计算或某些动态规划算法中很常见。例如，依赖$(i, j) \rightarrow (i+1, j+1)$。直接分块可能不合法，因为纯粹按列或按行执行瓦片会违反对角线依赖。倾斜变换改变了迭代空间相对于依赖的形状。

**机制**：倾斜变换会改变迭代器，从而改变依赖迭代的相对顺序。对于二维循环嵌套，一种常见的倾斜变换是：
$i' = i$
$j' = j + \alpha \cdot i$
其中$\alpha$是倾斜因子。这将原始的矩形迭代空间映射为一个平行四边形。主要影响在于依赖向量。如果原始依赖向量是$\vec{d} = (d_i, d_j)$，则在$(i', j')$空间中的新依赖向量$\vec{d}'$变为$(d_i, d_j + \alpha \cdot d_i)$。

> 倾斜变换 ($i'=i, j'=i+j$) 对二维迭代空间的影响。原始依赖$d=(1,1)$被转换为$d'=(1,2)$。此变化可能使能沿$i'$维度进行分块。

**使能分块**：通过改变依赖向量，倾斜变换可以使原本有问题的维度上的依赖在字典序上变为正向。对于依赖$d=(1,1)$，在通过$\alpha=1$进行倾斜变换后，依赖变为$d'=(1, 2)$。如果我们现在对倾斜空间$(i', j')$进行分块，依赖总是沿$i'$维度跨瓦片传递（因为$d'_{i'} = 1 > 0$）。这使得分块合法，而在原始空间上简单分块可能不合法。结果执行沿倾斜的波前进行。

### 组合变换与自动调度

Skewing和分块各自都很有用，但它们常被结合使用。一种常见策略是“先倾斜后分块”，即倾斜变换首先调整依赖，然后分块在倾斜空间中提升局部性和并行性。

寻找倾斜变换、分块以及其他潜在仿射变换（如置换或缩放）的最佳组合是复杂的。它很大程度上取决于具体的依赖模式和目标硬件架构。这正是基于多面体库（如`isl`，整数集库）的自动调度工具发挥作用的地方。这些工具分析多面体表示（域、依赖、访问函数），并采用算法（例如Pluto算法或Feautrier算法）来搜索合法的仿射调度$T$，以优化特定目标，例如最大化并行外层循环的数量或最小化数据移动，并根据需要纳入分块和倾斜变换。

一旦确定了合适的调度$T$（代表变换后的循环结构，可能包括分块和倾斜变换参数 (parameter)），下一步，在代码生成中讨论，是生成实现该调度的实际优化循环代码。

## 参考资料

- [A Practical and Complete Algorithm for Predicate Abstraction of Affine Programs](https://dl.acm.org/doi/10.1145/1458984.1459002) — Uday Bondhugula, Albert Hartono, J. Ramanujam, and P. Sadayappan (2008)
  Journal: Proceedings of the 17th International Conference on Parallel Architectures and Compilation Techniques (PACT '08); Publisher: ACM; Pages: 126-138; DOI: [10.1145/1458984.1459002](https://doi.org/10.1145/1458984.1459002)
  这篇论文介绍了Pluto算法，一个关键的自动多面体调度器，它通过遵守依赖关系生成多维仿射调度，有效实现分块和并行化。
- [Optimizing Compilers for Modern Architectures: A Dependence-Based Approach](https://books.google.com/books/about/Optimizing_Compilers_for_Modern_Architec.html?id=0s5oAAAAMAAJ) — Randy Allen and Ken Kennedy (2001)
  Publisher: Morgan Kaufmann
  这本经典且全面的教科书深入讲解了依赖分析和传统循环变换，这些是多面体模型的基本前置和补充技术。
