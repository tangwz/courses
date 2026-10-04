---
course: "compiler-runtime-optimization-ml"
chapter: "tensor-level-polyhedral-optimizations"
lesson: "polyhedral-modeling-intro"
sourceId: 4493
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations/polyhedral-modeling-intro"
title: "多面体建模简介"
description: "了解如何使用多面体代数表示循环嵌套和依赖关系的基本原理。"
order: 2
plots: ["plots/4493-0.json"]
sourceHash: "ebd3eb74e793ab423c5e4c87620496c27c154532574e0480dbfea166e31f4de4"
sourceCorrections: []
---

张量操作（如矩阵乘法或卷积）构成了许多机器学习 (machine learning)模型的计算核心。这些操作通常转化为嵌套循环，常伴有复杂的访问模式和依赖关系。例如，一个标准矩阵乘法 $C_{i,j} = \sum_{k} A_{i,k} \times B_{k,j}$ 的实现结构如下：

```c++
for (int i = 0; i < M; ++i) {
  for (int j = 0; j < N; ++j) {
    C[i][j] = 0; // 初始化累加器
    for (int k = 0; k < K; ++k) {
      C[i][j] += A[i][k] * B[k][j];
    }
  }
}
```

有效优化此类循环嵌套对性能来说非常重要。虽然传统编译器优化（如循环展开或循环交换）有所帮助，但它们常难以处理这些复杂情况中错综复杂的依赖关系和可能转换的搜索空间。它们可能以启发式方式应用转换，或未能找到可能高效但非显而易见的重排序。

为此，多面体模型提供了一种更为有效且系统化的方法。

### 循环的几何视角

多面体模型，亦称多胞体模型，是一个数学框架，用于分析和转换特定类型的循环嵌套，主要指那些循环边界和数组访问是外部循环迭代器和符号常数（参数 (parameter)）的仿射函数的循环。

多面体模型并非将循环纯粹视为顺序控制流结构，而是将语句的每个执行实例（循环迭代器值的一个特定组合）在嵌套中表示为多维空间中的一个整数点。一个语句的所有有效执行实例集合，构成了一个几何形状——多面体（或更准确地说，Z-多面体，因为我们关注整数点）内的整数集合。

考虑一个简化的二维循环嵌套：

```c++
for (int i = 0; i < 4; ++i) {
  for (int j = 0; j < i + 1; ++j) {
    S(i, j); // 在迭代 (i, j) 处执行的某个语句
  }
}
```

在多面体模型中，语句 `S` 的执行由满足循环边界约束的整数对 $(i, j)$ 集合定义：


$$
\text{迭代空间 } D_S = \{ (i, j) \in \mathbb{Z}^2 \mid 0 \le i < 4 \land 0 \le j < i+1 \}
$$


这组整数点 $(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)$ 构成了语句 `S` 的*迭代域*。它可以被形象地表示为二维 $(i, j)$ 平面中一个多边形内的整数坐标。



![示例循环嵌套的迭代空间](plots/4493-0.json)



> 整数点集合（蓝点）在嵌套循环示例中由语句 `S(i, j)` 执行。每个点代表一个独特的执行实例，由循环索引 $(i, j)$ 定义。

### 为何采用此表示方法？

使用仿射不等式定义的多面体来几何地表示循环，提供了多项益处：

1. **精准依赖分析：** 不同语句实例之间的数据依赖（流依赖、反依赖、输出依赖）可以利用这些整数集之间的关系进行精确描述。这有助于精确掌握哪些转换是合法的。
2. **转换的数学框架：** 循环转换，例如交换、反转、倾斜、分块和融合，对应于对这些多面体的数学操作（仿射转换）。这使得以结构化方式对复杂、组合的转换进行探究成为可能。
3. **统一优化：** 它提供了一个单一的框架，通过操纵迭代空间几何结构和其中点的调度（执行顺序），以同时考量并行性、局部性和向量 (vector)化。
4. **处理参数 (parameter)化边界：** 该模型自然适用于边界依赖于符号常数（参数，如矩阵乘法中的 `M`, `N`, `K`）的循环，从而实现对多种输入规模都有效的优化。

通过将循环嵌套抽象到这个数学域中，编译器可以摆脱纯粹语法驱动转换的限制。它们可以分析计算的基本特性，并根据优化迭代点执行调度来应用转换，常能找到从原始代码结构中不那么显而易见的优化。

接下来的章节将考察该模型的核心组成部分：迭代域、访问函数和依赖分析，之后将介绍它所支持的有效调度转换。

## 参考资料

- [Compiling for Massively Parallel Architectures: A Polyhedral Approach](https://link.springer.com/book/10.1007/978-1-4615-4676-1) — Alain Darte, Paul Feautrier, Georges-André Silber (2000)
  Publisher: Springer; DOI: [10.1007/978-1-4615-4676-1](https://doi.org/10.1007/978-1-4615-4676-1)
  介绍多面体模型及其在并行化编译器中应用的基础著作之一。
- [Pluto: A Practical and Effective Polyhedral Loop Optimizer](https://dl.acm.org/doi/10.1145/1375581.1375595) — Uday Bondhugula, Albert Cohen, Jagannathan Ramanujam, P. Sadayappan (2008)
  Journal: Proceedings of the 2008 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI); Publisher: ACM; Pages: 101-113; DOI: [10.1145/1375581.1375595](https://doi.org/10.1145/1375581.1375595)
  介绍了广泛使用的Pluto编译器，展示了多面体模型如何有效地应用于自动并行化和局部性优化。
- [The Polyhedral Model in Practice](https://www.sciencedirect.com/science/article/pii/B9780123855072000019) — Fabrice Rastello, Uday Bondhugula, Albert Cohen, Paul Feautrier, Sylvain Lelait (2011)
  Journal: Advances in Computers; Publisher: Elsevier; Volume: 80; Pages: 1-100; DOI: [10.1016/B978-0-12-385507-2.00001-9](https://doi.org/10.1016/B978-0-12-385507-2.00001-9)
  全面阐述多面体模型在现代编译器设计中的实践方面和影响的章节。
