---
course: "compiler-runtime-optimization-ml"
chapter: "tensor-level-polyhedral-optimizations"
lesson: "polyhedral-components"
sourceId: 4497
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations/polyhedral-components"
title: "迭代域、访问函数和依赖"
description: "定义并分析多面体模型的核心部分。"
order: 3
plots: []
sourceHash: "ecd432edb319895f7003014fc6c45a08e2e5170b068087193deacd73848af203"
sourceCorrections: []
---

为有效分析和转换张量计算中常见的循环嵌套，多面体模型采用了一个精确的数学框架。此框架依据三个基本组成部分：迭代域定义计算发生的时间，访问函数定义涉及的数据，以及依赖分析理解重排计算的限制。下面我们逐一查看这些内容。

### 迭代域：定义执行空间

在可能嵌套的循环结构中，语句的每一次执行都发生在由周围循环索引值定义的特定点。我们将这种特定执行称为*语句实例*。多面体模型捕获了语句执行时所有可能的整数索引向量 (vector)集合。这个集合被称为**迭代域**。

形式上，对于嵌套在$d$个循环中的语句$S$，其索引变量为$i_1, i_2, \ldots, i_d$，迭代域$D_S$是一个整数向量集合$iter = (i_1, i_2, \ldots, i_d) \in \mathbb{Z}^d$。这个集合由循环边界定义，循环边界必须可以表示为涉及循环索引和可能存在的符号常量（参数 (parameter)）的*仿射不等式*，这些常量代表未知的问题大小（如矩阵维度）。

迭代域$D_S$可以表示为：


$$
D_S = \{ iter \in \mathbb{Z}^d \mid A \cdot iter + b \ge 0 \}
$$


其中，$A$是一个整数矩阵，$b$是一个整数向量，表示从循环边界推导出的约束。不等式是按元素进行的。任何通过此类线性不等式定义的整数集合都会形成一个整数多面体。

**示例：矩阵乘法**

考虑矩阵乘法$C = A \times B$中的核心计算：

```c
// 假设C初始化为零
for (int i = 0; i < M; ++i) {      // 循环索引 i
  for (int j = 0; j < N; ++j) {    // 循环索引 j
    for (int k = 0; k < P; ++k) {  // 循环索引 k
      // 语句 S1:
      C[i][j] = C[i][j] + A[i][k] * B[k][j];
    }
  }
}
```

语句$S_1$嵌套在三个循环中。$S_1$的一个实例由向量$iter = (i, j, k)$标识。迭代域$D_{S1}$是所有由循环边界定义的有效$(i, j, k)$元组的集合：


$$
D_{S1} = \{ (i, j, k) \in \mathbb{Z}^3 \mid 0 \le i < M, 0 \le j < N, 0 \le k < P \}
$$


这可以使用仿射不等式来书写：
$i \ge 0$
$-i + (M-1) \ge 0$ （等价于$i \le M-1$，或$i < M$）
$j \ge 0$
$-j + (N-1) \ge 0$
$k \ge 0$
$-k + (P-1) \ge 0$

这在3D迭代空间$(i, j, k)$中形成一个有界整数多面体（一个长方体）。$M, N, P$是符号参数。

### 访问函数：迭代到数据的映射

在每次迭代$iter \in D_S$中，语句$S$通常会读写内存位置，这些位置常是数组（张量）的元素。一个**访问函数**描述了某个语句实例访问哪个内存位置。

多面体模型要求这些访问函数是迭代向量 (vector)$iter$的*仿射*函数。对于像`Array[index_expr_1]...[index_expr_n]`这样的数组访问，该函数将迭代向量$iter$映射到数组索引向量$(index\_expr\_1, \ldots, index\_expr\_n)$。这种映射必须采用以下形式：


$$
F(iter) = M \cdot iter + c
$$


其中$M$是一个整数矩阵，$c$是一个整数向量。

**示例：矩阵乘法访问**

继续矩阵乘法示例，语句$S_1$涉及四次访问：对`C`的读写，对`A`的读，以及对`B`的读。设迭代向量为$iter = (i, j, k)$。

1. **写入`C[i][j]`**：访问的索引是$(i, j)$。访问函数$F_{C, write}$将$(i, j, k)$映射到$(i, j)$。

   
   $$
   F_{C, write}(i, j, k) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix} \begin{pmatrix} i \\ j \\ k \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} i \\ j \end{pmatrix}
   $$
   

   这是仿射的。
2. **读取`C[i][j]`**：访问的索引是$(i, j)$。访问函数$F_{C, read}$与$F_{C, write}$相同。

   
   $$
   F_{C, read}(i, j, k) = \begin{pmatrix} i \\ j \end{pmatrix}
   $$
   
3. **读取`A[i][k]`**：访问的索引是$(i, k)$。访问函数$F_{A, read}$将$(i, j, k)$映射到$(i, k)$。

   
   $$
   F_{A, read}(i, j, k) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} i \\ j \\ k \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} i \\ k \end{pmatrix}
   $$
   

   这是仿射的。
4. **读取`B[k][j]`**：访问的索引是$(k, j)$。访问函数$F_{B, read}$将$(i, j, k)$映射到$(k, j)$。

   
   $$
   F_{B, read}(i, j, k) = \begin{pmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix} \begin{pmatrix} i \\ j \\ k \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} k \\ j \end{pmatrix}
   $$
   

   这也是仿射的。

由于所有迭代域都由仿射不等式定义，并且所有访问函数都是仿射的，因此该循环嵌套适用于在多面体模型中进行分析。

### 数据依赖：理解执行限制

循环顺序改变、分块或并行化等优化只有在保持程序语义的前提下才有效。这要求理解不同语句实例之间的数据依赖。如果两个语句实例访问相同的内存位置，并且其中至少一个访问是写入操作，那么它们之间存在数据依赖。

数据依赖的常见类型有：

1. **流（真）依赖（写后读，RAW）**：语句实例$S_1(iter_1)$写入的位置随后被$S_2(iter_2)$读取。这决定了$S_1(iter_1)$必须在$S_2(iter_2)$之前执行。
2. **反依赖（读后写，WAR）**：$S_1(iter_1)$从一个位置读取，该位置随后被$S_2(iter_2)$覆盖。$S_1(iter_1)$仍必须在$S_2(iter_2)$之前执行，以便读取原始值。
3. **输出依赖（写后写，WAW）**：$S_1(iter_1)$写入的位置随后被$S_2(iter_2)$覆盖。顺序对于存储的最终值有影响。$S_1(iter_1)$必须在$S_2(iter_2)$之前执行。
4. **输入依赖（读后读，RAR）**：$S_1(iter_1)$读取的位置随后被$S_2(iter_2)$读取。这不强制要求执行顺序，但可能与局部性优化有关。

为了使用多面体框架正式识别依赖，我们寻找满足以下条件的迭代对$(iter_1, iter_2)$：

- $iter_1 \in D_{S1}$且$iter_2 \in D_{S2}$（两个实例都是有效的执行点）。
- $iter_1$在$iter_2$之前执行。在原始代码中，这由迭代向量 (vector)的字典顺序决定。我们将其表示为$iter_1 \prec iter_2$。
- 它们访问相同的内存位置：$F_{S1, mem}(iter_1) = F_{S2, mem}(iter_2)$，其中$F_{S1, mem}$和$F_{S2, mem}$分别是$S_1$和$S_2$中相关内存引用的访问函数。
- 至少有一个访问（$S_1(iter_1)$或$S_2(iter_2)$）是写入操作。

所有满足这些特定类型依赖（例如流依赖）条件的迭代对$(iter_1, iter_2)$的集合定义了依赖关系。这种关系本身也常可以表示为另一个多面体。

**示例：简单循环中的依赖**

考虑这个循环：

```c
for (int i = 1; i < N; ++i) {
  A[i] = A[i-1] + B[i]; // 语句 S
}
```

- **迭代域**：$D_S = \{ i \in \mathbb{Z} \mid 1 \le i < N \}$
- **访问函数（针对数组 A）**：
  - 写入`A[i]`：$F_{write}(i) = i$
  - 从`A[i-1]`读取：$F_{read}(i) = i - 1$

我们来查找流依赖（写后读）。我们需要满足以下条件的$iter_1 = i_1$和$iter_2 = i_2$：

1. $i_1 \in D_S$, $i_2 \in D_S$
2. $i_1 \prec i_2$ （因为是单循环，所以意味着$i_1 < i_2$）
3. 在$i_1$处的写入访问对应于在$i_2$处的读取访问：$F_{write}(i_1) = F_{read}(i_2)$
   
   $$
   i_1 = i_2 - 1
   $$
   

组合这些条件：$1 \le i_1 < N$, $1 \le i_2 < N$, $i_1 < i_2$, 和$i_1 = i_2 - 1$。
这简化为：对于任何满足$2 \le i_2 < N$的$i_2$，都存在从迭代$i_1 = i_2 - 1$发出的流依赖。
对于所有从$1$到$N-2$的$i$，依赖关系从迭代$i$存在到迭代$i+1$。*依赖向量*是$d = iter_2 - iter_1 = i_2 - i_1 = 1$。这表明迭代$i+1$依赖于迭代$i$中计算的结果。

> 示例循环中的流依赖。每次迭代$i$写入`A[i]`，该值由下一次迭代$i+1$读取。依赖向量是常数，$d=(1)$。

**矩阵乘法中的依赖分析**

将此应用于矩阵乘法示例中对`C[i][j]`的访问：

- $S_1(i, j, k)$写入$C[i][j]$。
- $S_1(i, j, k+1)$读取$C[i][j]$（当$k < P-1$时）。
  这是一个流依赖：$iter_1 = (i, j, k)$，$iter_2 = (i, j, k+1)$。$iter_1 \prec iter_2$。访问函数$F_C(iter_1) = F_C(iter_2) = (i, j)$。依赖向量是$d = iter_2 - iter_1 = (0, 0, 1)$。此依赖由最内层循环$k$承载。
- 类似地，$S_1(i, j, k)$写入$C[i][j]$，$S_1(i, j, k+1)$也写入$C[i][j]$。这是一个输出依赖，依赖向量同样是$d=(0, 0, 1)$。

这些依赖告诉我们，对于给定的$(i, j)$，如果没有归约等机制，k循环的迭代不能随意重排或完全并行化，因为每次迭代都依赖于前一次在$C[i][j]$中存储的部分和。然而，具有不同$(i, j)$值的迭代在C数组访问方面是独立的。

通过精确定义迭代域、访问函数以及由此产生的数据依赖，多面体模型为数学推演循环嵌套并推导复杂且保持正确性的性能优化转换提供了依据。下一步涉及利用此依赖信息构建有效的*调度*，将迭代映射到执行时间步长，从而实现分块和并行化等优化。

## 参考资料

- [The Polyhedral Model and the Integer Set Library (ISL)](https://doi.org/10.1007/978-3-642-11814-2_16) — Sven Verdoolaege (2010)
  Journal: Compiler Construction (CC 2010); Publisher: Springer; Volume: 6049; Pages: 228-243; DOI: [10.1007/978-3-642-11814-2_16](https://doi.org/10.1007/978-3-642-11814-2_16)
  本文描述了多面体模型及其在整数集合库（ISL）中的实际实现，ISL是广泛用于表示和操作迭代域和访问函数的工具。
