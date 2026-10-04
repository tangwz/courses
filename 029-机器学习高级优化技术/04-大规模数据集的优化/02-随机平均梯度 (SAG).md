# 随机平均梯度 (SAG)

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/sag-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

随机梯度下降 (gradient descent) (SGD) 是处理大规模机器学习 (machine learning)任务的常用方法。但它依赖于从小批量数据计算出的噪声梯度，可能导致收敛缓慢或不稳定。每一步仅采样一小部分数据引入的方差，通常需要细致调整学习率和更新策略。

随机平均梯度 (SAG) 算法提供了一种有吸引力的替代方案，直接旨在降低这种方差。SAG 由 Schmidt、Le Roux 和 Bach 于 2012 年提出，它试图将 SGD 的低迭代成本与全批量梯度下降常有的更快收敛特性结合起来，特别是对于强凸问题。

### 主要思想：平均过去的梯度

SAG 背后的理念简单而巧妙：与其只使用当前小批量（或最简单情况下的单个数据点）的梯度，不如借鉴先前迭代中计算出的梯度信息？SAG 会为训练集中的*每个独立数据点*，记住*最近计算出的梯度*。

设目标函数是 $N$ 个数据点的平均损失：
$L(\theta) = \frac{1}{N} \sum_{i=1}^N L_i(\theta)$
其中 $L_i(\theta)$ 是与第 $i$ 个数据点关联的损失，$\theta$ 表示模型参数 (parameter)。

全批量梯度下降 (gradient descent)使用梯度 $\nabla L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla L_i(\theta)$。SGD 最简单形式（批量大小为 1）在迭代 $k$ 时选择一个随机索引 $i_k$，并使用 $\nabla L_{i_k}(\theta_k)$ 进行更新。

SAG 的工作方式不同。它维护一个包含 $N$ 个梯度向量 (vector)的表格，$g_1, g_2, \dots, g_N$。每个 $g_i$ 存储了当数据点 $i$ 在过去某个迭代中被选中时，为其计算出的最近梯度 $\nabla L_i(\theta)$。

### SAG 更新规则

在每次迭代 $k+1$ 时：

1. **采样：** 从 $\{1, 2, \dots, N\}$ 中随机选择一个索引 $i_k$。
2. **计算当前梯度：** 使用当前参数 (parameter) $\theta_k$ 计算所选数据点的梯度：$v_{i_k} = \nabla L_{i_k}(\theta_k)$。
3. **更新平均梯度估计：** SAG 的核心是使用存储梯度的平均值作为更新方向。这个平均值可以高效地更新。设 $G_k = \frac{1}{N} \sum_{j=1}^N g_j$ 是存储到迭代 $k$ 的梯度平均值。新的平均值 $G_{k+1}$ 通过在求和中用新计算的梯度 $v_{i_k}$ 有效地替换旧的存储梯度 $g_{i_k}$ 来计算：
   $G_{k+1} = G_k + \frac{1}{N} (v_{i_k} - g_{i_k})$
   这避免了在每一步重新求和所有 $N$ 个存储的梯度。
4. **更新参数：** 使用这个噪声较少的平均梯度估计更新参数：
   $\theta_{k+1} = \theta_k - \eta G_{k+1}$
   其中 $\eta$ 是学习率。
5. **更新记忆：** 将新计算的梯度 $v_{i_k}$ 存储到记忆表格中，替换掉 $g_{i_k}$ 的先前值：
   $g_{i_k} \leftarrow v_{i_k}$

最初，存储的梯度 $g_i$ 可以初始化为零向量 (vector)或根据初始参数猜测计算。

> 单次 SAG 迭代的流程。采样一个数据点 $i_k$，计算其当前梯度，这个新梯度同时更新运行中的平均梯度估计 ($G_{k+1}$) 和存储的梯度记忆 ($g_{i_k}$)。

### 收敛性质

SAG 的主要优点之一是其收敛速度。对于强凸且光滑的目标函数，SAG 实现了*线性*收敛速度。这相比于标准 SGD 在这类问题上通常的*次线性*速度是一个显著提升。简单来说，SAG 在达到一定精度所需的迭代次数方面收敛更快，接近全批量梯度下降 (gradient descent)的理想特性。与每迭代成本随 $N$ 线性增加的全批量梯度下降不同，SAG 的收敛速度不会随着数据集大小 $N$ 的增加而显著下降。

### 计算成本与内存

- **计算：** SAG 的每次迭代成本主要由单个梯度 $\nabla L_{i_k}(\theta_k)$ 的计算以及向量 (vector)加法/减法决定。这与 SGD（批量大小为 1）的每次迭代成本相当。它比全批量梯度下降 (gradient descent)便宜得多，后者每迭代需要计算 $N$ 个梯度。
- **内存：** SAG 的主要缺点是其内存需求。它需要存储 $N$ 个梯度向量，其中 $N$ 是训练集中的数据点总数。如果参数 (parameter)维度 $d$ 很大且 $N$ 数量庞大（数十亿样本），存储 $N \times d$ 个浮点数可能变得不切实际甚至不可能。

### SAG 与 SGD 对比

- **方差：** SAG 通过平均梯度显著降低了相比 SGD 的方差。这带来更稳定的更新和更快的收敛。
- **内存：** SGD 的内存开销最小（仅存储参数 (parameter)和当前小批量梯度），而 SAG 需要存储 $N$ 个梯度。
- **收敛速度（迭代次数）：** 对于强凸问题，SAG 通常比 SGD（次线性）收敛快得多（线性）。
- **实现复杂度：** 由于需要管理梯度记忆表，SAG 的实现略微复杂。

SAG 代表了弥合 SGD 可扩展性与批量方法快速收敛之间差距的重要一步。然而，其内存需求限制了它在 $N$ 极大情景下的适用性。这一限制推动了后续方差削减技术（如 SVRG，我们接下来会讨论）的发展，旨在保留快速收敛优势的同时减轻内存负担。

## 参考资料

- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics; Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  一篇关于大规模机器学习优化方法的全面综述，其中包含SAG。
- [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](http://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction) — Rie Johnson, Tong Zhang (2013)
  Journal: Advances in Neural Information Processing Systems 26 (NIPS 2013); Publisher: Curran Associates Inc.; Pages: 315-323; DOI: [10.55989/NIPS-2013-1070](https://doi.org/10.55989/NIPS-2013-1070)
  这篇论文介绍了随机方差缩减梯度 (SVRG)，它是SAG的后续算法，旨在解决其内存限制。

---

[上一节](01-%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E5%86%8D%E8%AE%A8%E8%AE%BA%EF%BC%9A%E6%96%B9%E5%B7%AE%E5%87%8F%E5%B0%8F.md) · [下一节](03-%E9%9A%8F%E6%9C%BA%E6%96%B9%E5%B7%AE%E7%BC%A9%E5%87%8F%E6%A2%AF%E5%BA%A6%20%28SVRG%29.md)
