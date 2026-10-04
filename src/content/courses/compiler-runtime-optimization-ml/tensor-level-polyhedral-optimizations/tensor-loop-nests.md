---
course: "compiler-runtime-optimization-ml"
chapter: "tensor-level-polyhedral-optimizations"
lesson: "tensor-loop-nests"
sourceId: 4491
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-4-tensor-level-polyhedral-optimizations/tensor-loop-nests"
title: "将张量计算表示为循环嵌套"
description: "了解卷积和矩阵乘法等张量操作如何对应为嵌套循环。"
order: 1
plots: []
sourceHash: "4ff4e790fcdeb8f600efaa0477cd115670c1c33f2b1489feb944fa0ff0e315be"
sourceCorrections: []
---

我们将重点从优化高级计算图（第3章）转向单个张量操作时，会遇到机器学习 (machine learning)工作负载中经常占据主要执行时间的计算核。矩阵乘法或卷积等数学定义的运算需要具体的实现才能执行。这种实现通常表现为嵌套循环，对输入和输出张量的维度进行迭代。了解如何将这些基本运算表示为循环嵌套，是运用强大的低级编译器优化技术的第一步。

### 矩阵乘法示例

考虑标准的矩阵乘法运算：
$C = A \times B$
其中 $A$ 是一个 $M \times K$ 矩阵，$B$ 是一个 $K \times N$ 矩阵，结果矩阵 $C$ 是 $M \times N$。输出矩阵中的每个元素 $C_{ij}$ 计算方式为 $A$ 的第 $i$ 行与 $B$ 的第 $j$ 列的点积：
$C_{ij} = \sum_{k=0}^{K-1} A_{ik} B_{kj}$
这个数学定义直接对应为三重嵌套循环结构。假设在类似 C 语言中矩阵采用标准的行主序存储，计算可以表示为：

```cpp
// 假设 C 已初始化为零
// A 是 M x K, B 是 K x N, C 是 M x N
for (int i = 0; i < M; ++i) {         // 遍历 C (和 A) 的行
    for (int j = 0; j < N; ++j) {     // 遍历 C (和 B) 的列
        for (int k = 0; k < K; ++k) { // 遍历归约维度
            C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}
```

这里：

- 外层循环遍历输出矩阵 $C$ 的行（由 `i` 索引）。
- 中层循环遍历输出矩阵 $C$ 的列（由 `j` 索引）。
- 内层循环遍历共享维度 $K$（由 `k` 索引），执行乘加操作。

这些循环的顺序（`i, j, k`）只是一种可能性。其他排列（`i, k, j`、`j, i, k` 等）计算结果相同，但由于内存访问模式和缓存行为，其性能表现可能大不相同。这种循环嵌套表示捕获了迭代空间（$0 \le i < M, 0 \le j < N, 0 \le k < K$）以及循环体内的内存访问，它成为本章后面讨论的优化技术（例如多面体建模）的主要优化对象。

### 卷积示例

卷积运算作为卷积神经网络 (neural network)（CNN）的核心，也很自然地对应为复杂的循环嵌套。典型的 2D 卷积将滤波器（核）应用于输入特征图以生成输出特征图。为了说明，我们考虑一个不带批处理或多通道的简化 2D 卷积：

输入 $I$ 的大小为 $H \times W$，核 $K$ 的大小为 $R \times S$，输出 $O$ 的大小为 $(H-R+1) \times (W-S+1)$。

输出元素 $O_{xy}$ 的计算涉及对核维度的求和：
$O_{xy} = \sum_{r=0}^{R-1} \sum_{s=0}^{S-1} I_{x+r, y+s} K_{rs}$

这对应为至少四层嵌套循环的结构：

```cpp
// 简化 2D 卷积（输出 O 已初始化为零）
// 输入 I: H x W, 核 K: R x S, 输出 O: P x Q (其中 P=H-R+1, Q=W-S+1)
for (int p = 0; p < P; ++p) { // 输出高度
    for (int q = 0; q < Q; ++q) { // 输出宽度
        for (int r = 0; r < R; ++r) { // 核高度
            for (int s = 0; s < S; ++s) { // 核宽度
                O[p * Q + q] += I[(p + r) * W + (q + s)] * K[r * S + s];
            }
        }
    }
}
```

"机器学习 (machine learning)模型中的卷积还会增加批处理大小 ($N$) 和输入/输出通道 ($C_{in}, C_{out}$) 的循环，从而得到 6 甚至 7 层嵌套循环。确切的索引表达式也很大程度上取决于所使用的数据布局（例如 NCHW 对比 NHWC）。NCHW（批次、通道、高度、宽度）在 PyTorch 等框架中很常见，而 NHWC 则常因性能原因被 TensorFlow 和硬件加速器优先选用。布局的选择直接影响最内层循环中计算的索引，并影响内存访问模式。"

### 优化的依据

将张量计算表示为这些显式循环嵌套，提供了一个编译器可以分析和操作的具体结构。循环定义了*迭代空间*，而循环体内的数组访问（$C[i][j]$、$A[i][k]$ 等）则定义了*数据依赖关系*和*内存访问模式*。

> 计算 2x2 矩阵乘法（$M=2, K=2, N=1$）第一列（$j=0$）的数据依赖关系。注意矩阵 B 中输入元素的复用。

这种明确、结构化的表示是多面体建模、循环切分、融合和向量 (vector)化等技术的必要起点。这些方法分析循环嵌套定义的依赖关系和迭代空间，将其转换为等效但效率明显更高的操作序列，并针对特定硬件目标进行优化。接下来的部分将说明多面体建模如何为这些有效的转换提供一个正式的体系。

## 参考资料

- [Polyhedral Optimizations for GPGPUs](https://dl.acm.org/doi/10.1145/2132896.2132902) — Uday Bondhugula, Albert Hartono, Jagannath Kannan, R. M. Ramanujam, Jay Hoeflinger, and Paul H. J. Kelly (2012)
  Journal: ACM Transactions on Architecture and Code Optimization (TACO); Publisher: ACM; Volume: 8; Pages: 1-28; DOI: [10.1145/2132896.2132902](https://doi.org/10.1145/2132896.2132902)
  探讨多面体模型在优化循环嵌套中的应用，尤其针对GPGPU架构，展示了平铺和融合等转换如何提高计算密集型内核的性能。
- [Compilers for Deep Learning](https://www.morganclaypool.com/doi/10.2200/S01021ED1V01Y202006CAV016) — Christophe Dubach, Oleksiy Telyatnikov, and Hugh Leather (2020)
  Journal: Synthesis Lectures on Computer Architecture; Publisher: Morgan & Claypool Publishers; Pages: 1-137; DOI: [10.2200/S01021ED1V01Y202006CAV016](https://doi.org/10.2200/S01021ED1V01Y202006CAV016)
  概述了深度学习工作负载特有的编译器技术和挑战，包括张量操作的下沉、图优化和针对特定硬件的代码生成。
- [BLIS: A Framework for Rapidly Instantiating High-Performance BLAS Operations](https://dl.acm.org/doi/10.1145/2765131) — Field G. Van Zee and Robert A. van de Geijn (2015)
  Journal: ACM Transactions on Mathematical Software (TOMS); Publisher: ACM; Volume: 41; Pages: 1-33; DOI: [10.1145/2765131](https://doi.org/10.1145/2765131)
  介绍了BLIS，一个用于系统地开发高性能BLAS操作（特别是矩阵乘法）实现的框架，通过结构化优化来提升内存层次和并行化效率。
- [TVM: An Automatic End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy (2018)
  Journal: 13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18); Publisher: USENIX Association; Pages: 578-594; DOI: [10.5555/3304128.3304170](https://doi.org/10.5555/3304128.3304170)
  介绍了TVM，一个用于深度学习的编译器栈，它自动优化张量计算，处理循环嵌套转换、内存管理和不同硬件后端的数据布局等挑战。
