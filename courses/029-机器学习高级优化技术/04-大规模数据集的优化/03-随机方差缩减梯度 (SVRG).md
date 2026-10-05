# 随机方差缩减梯度 (SVRG)

来源：[原文](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/svrg-algorithm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然随机梯度下降 (gradient descent) (SGD) 为大型数据集提供了扩展性，但其固有的梯度方差仍然是一个主要障碍，常阻碍快速收敛到高精度解。随机平均梯度 (SAG) 通过存储并平均历史梯度来处理此问题，但代价是可能需要大量内存。随机方差缩减梯度 (SVRG) 提出了一种替代的方差缩减方法，无需为每个数据点存储梯度。

SVRG背后的主要思想很精巧：它并非只使用有噪声的随机梯度，而是借助周期性计算的、噪声较小的全批量梯度信息来修正此梯度。这种修正旨在保持对真实梯度的无偏估计，同时明显缩减其方差，尤其是在优化过程接近最小值时。

### SVRG算法

SVRG以周期（epoch）为单位运行，每个周期包含两个阶段：全梯度计算和一系列由该全梯度优化的随机更新。

设 $f(w)$ 是我们想要最小化的目标函数，它通常是 $N$ 个数据点的平均值：$f(w) = \frac{1}{N} \sum_{i=1}^{N} f_i(w)$。

该算法步骤如下：

1. **初始化：** 从初始参数 (parameter)向量 (vector) $w_0$ 开始。
2. **外循环 (周期)：** 对于每个周期 $s = 0, 1, 2, ...$
   a. **快照：** 选择一个快照参数向量 $\tilde{w}$。一个常见的选择是前一个周期的最终参数向量，即 $\tilde{w} = w_m^{(s-1)}$（或者对于第一个周期是 $w_0$）。
   b. **全梯度计算：** 使用快照参数计算整个数据集的平均梯度：$\nabla f(\tilde{w}) = \frac{1}{N} \sum_{i=1}^{N} \nabla f_i(\tilde{w})$。这是计算量大的步骤，每个周期只执行一次。
   c. **内循环 (随机更新)：** 初始化内循环参数 $w_0^{(s)} = \tilde{w}$。然后，对于 $t = 0, 1, ..., m-1$：
   i. **采样：** 从 $\{1, ..., N\}$ 中随机选取一个索引 $i_t$（或一个mini-batch）。
   ii. **计算随机梯度：** 使用*相同*的样本 $i_t$ 计算两个随机梯度：一个在当前迭代 $w_t^{(s)}$ 处，另一个在快照 $\tilde{w}$ 处。
   \* $\nabla f_{i_t}(w_t^{(s)})$
   \* $\nabla f_{i_t}(\tilde{w})$
   iii. **构建方差缩减梯度：** 构成SVRG梯度估计 $v_t^{(s)}$：
   $v_t^{(s)} = \nabla f_{i_t}(w_t^{(s)}) - \nabla f_{i_t}(\tilde{w}) + \nabla f(\tilde{w})$
   iv. **更新参数：** 使用学习率 $\eta$ 更新参数：
   $w_{t+1}^{(s)} = w_t^{(s)} - \eta v_t^{(s)}$
   d. **周期更新：** 设置下一周期的起始点，例如 $w_0 = w_m^{(s)}$。另外，也可以使用内循环迭代的平均值。

> 流程图说明了SVRG算法的结构。外循环周期性地计算全梯度，然后将其用于内循环中，以修正用于参数更新的随机梯度估计。

### SVRG为何有效？

其奥妙在于梯度估计 $v_t^{(s)}$ 的构建方式。我们来分析它的特性：

- **无偏估计：** 对 $i_t$ 的选择取期望，我们发现 $v_t^{(s)}$ 是 $w_t^{(s)}$ 处真实梯度的无偏估计量：

  
  $$
  E_{i_t}[v_t^{(s)}] = E_{i_t}[\nabla f_{i_t}(w_t^{(s)})] - E_{i_t}[\nabla f_{i_t}(\tilde{w})] + \nabla f(\tilde{w})
  $$
  

  由于 $E_{i_t}[\nabla f_{i_t}(w)] = \nabla f(w)$，这简化为：

  
  $$
  E_{i_t}[v_t^{(s)}] = \nabla f(w_t^{(s)}) - \nabla f(\tilde{w}) + \nabla f(\tilde{w}) = \nabla f(w_t^{(s)})
  $$
  

  因此，平均而言，SVRG 更新方向与真实梯度方向一致，如同SGD或全批量梯度下降 (gradient descent)。
- **方差缩减：** $v_t^{(s)}$ 的方差是SVRG之所以有效的原因。考虑项 $\nabla f_{i_t}(w_t^{(s)}) - \nabla f_{i_t}(\tilde{w})$。随着周期内迭代 $w_t^{(s)}$ 收敛到快照参数 (parameter) $\tilde{w}$，这个差项会变小。由于 $\nabla f(\tilde{w})$ 在内循环中是常数，所以 $v_t^{(s)}$ 的方差主要取决于 $\nabla f_{i_t}(w_t^{(s)}) - \nabla f_{i_t}(\tilde{w})$ 的方差。当 $w_t^{(s)} \to \tilde{w}$ 时，这个方差趋近于零。这与标准SGD形成鲜明对比，在标准SGD中，即使接近最小值，方差 $Var[\nabla f_{i_t}(w_t)]$ 也大致保持不变。

这种方差缩减使得SVRG能够采取更积极的步长（更大的常数学习率），并比SGD更快地收敛，尤其是在需要高精度时。

### SVRG 与 SGD 和 SAG 对比

- **SVRG 与 SGD：** SVRG 需要周期性地计算全梯度，这比SGD增加了计算开销。然而，其明显缩减的方差常导致数据有效遍历次数上的更快收敛，可能需要更少的总计算量来达到目标精度。SGD通常需要递减的学习率，而SVRG常能使用恒定的学习率。
- **SVRG 与 SAG：** SVRG和SAG都通过缩减方差，提供了比SGD更快的收敛。SAG通过存储和平均每个数据点的历史梯度来实现此目的，需要 $O(Nd)$ 的内存（其中 $d$ 是参数 (parameter)维度）。SVRG通过周期性地重新计算全梯度来避免这种内存成本，这使得它在 $N$ 很大时更适用。计算上的权衡在于SAG的每次迭代内存访问/更新成本与SVRG的周期性全梯度计算之间。



[交互图表：收敛性：SVRG 与 SGD 对比](https://apxml.com/zh/courses/optimization-techniques-ml/chapter-4-large-scale-ml-optimization/svrg-algorithm#plot-dcyoo3)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "name": "SGD (递减学习率)",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20
      ],
      "y": [
        1.0,
        0.8,
        0.65,
        0.55,
        0.48,
        0.42,
        0.38,
        0.35,
        0.33,
        0.31,
        0.3,
        0.28,
        0.27,
        0.26,
        0.255,
        0.25
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#339af0",
        "dash": "dash"
      }
    },
    {
      "name": "SVRG (恒定学习率)",
      "x": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        12,
        14,
        16,
        18,
        20
      ],
      "y": [
        1.0,
        0.5,
        0.25,
        0.12,
        0.06,
        0.03,
        0.015,
        0.008,
        0.004,
        0.002,
        0.001,
        0.0005,
        0.0002,
        0.0001,
        5e-05,
        2e-05
      ],
      "type": "scatter",
      "mode": "lines",
      "line": {
        "color": "#f76707"
      }
    }
  ],
  "layout": {
    "title": "收敛性：SVRG 与 SGD 对比",
    "xaxis": {
      "title": "有效数据遍历次数"
    },
    "yaxis": {
      "title": "对数损失或误差",
      "type": "log"
    },
    "legend": {
      "x": 0.1,
      "y": 0.1,
      "bgcolor": "rgba(255,255,255,0.5)"
    },
    "margin": {
      "l": 60,
      "r": 10,
      "t": 50,
      "b": 40
    },
    "font": {
      "family": "sans-serif",
      "size": 12
    }
  }
}
```

</details>



> 图表对比了SVRG和SGD的收敛行为。SVRG因为方差缩减，常表现出线性收敛（在对数线性图上呈直线），而SGD呈现次线性收敛，并可能在较高的误差水平处停滞。

### 实现考量

- **内循环长度 ($m$)：** 这是一个重要参数 (parameter)。它决定了每次全梯度计算要执行多少次随机更新。理论上，$m$ 应该足够大以抵消全梯度计算的成本。常见的选择是将 $m$ 与数据集大小 $N$ 相关联，例如 $m=2N$ 或 $m=5N$。如果 $m$ 过小，全梯度的开销将占主导地位。如果 $m$ 过大，快照 $\tilde{w}$ 相对于内循环迭代 $w_t^{(s)}$ 可能会变得过时，可能阻碍收敛。
- **学习率 ($η$)：** SVRG的一个显著优点是它常能很好地使用恒定学习率，这与SGD不同，后者需要仔细调整递减策略。合适的恒定学习率取决于目标函数的特性（如平滑性和强凸性）。
- **快照更新策略：** 尽管将下一个周期的快照 $\tilde{w}$ 设置为内循环的最终迭代 $w_m^{(s)}$ 是常见做法，但另一个选择是使用内循环迭代的平均值 $\frac{1}{m} \sum_{t=0}^{m-1} w_t^{(s)}$。这种平均化有时能提供更稳定的性能。
- **Mini-batch SVRG：** 上述描述使用了单个数据点 $i_t$。在实际应用中，SVRG几乎总是为了效率而使用mini-batch来实现，类似于标准SGD。其原理保持不变：计算 $w_t^{(s)}$ 处的mini-batch梯度，使用*相同的mini-batch*计算 $\tilde{w}$ 处的mini-batch梯度，并使用预先计算好的全梯度 $\nabla f(\tilde{w})$ 作为修正项。

### 收敛性与适用性

对于强凸且平滑的目标函数，SVRG已被证明能达到线性收敛速度（对于某些 $\rho < 1$ 为 $O(\rho^k)$），这意味着误差随周期数呈指数级下降。这相对于标准SGD的次线性速度（$O(1/k)$）有很大提升。尽管理论保证通常是针对凸问题推导的，但SVRG在实践中也已证明对训练深度神经网络 (neural network)有效，尽管在非凸设置下的分析更为复杂。

总结来说，SVRG为大规模优化提供了一种引人注目的方法。通过周期性计算全梯度并用其修正有噪声的随机估计，它实现了快速收敛，而没有SAG等方法的内存负担，使其成为在海量数据集上训练模型的重要工具。主要的权衡是周期性全梯度计算的计算成本，这必须与内循环更新中实现的更快收敛进行权衡。

## 参考资料

- [Optimization Methods for Large-Scale Machine Learning](https://doi.org/10.1137/16M1080173) — Léon Bottou, Frank E. Curtis, and Jorge Nocedal (2018)
  Journal: SIAM Review; Publisher: Society for Industrial and Applied Mathematics (SIAM); Volume: 60; Pages: 223-311; DOI: [10.1137/16M1080173](https://doi.org/10.1137/16M1080173)
  一本涵盖机器学习各种优化技术的综合教材，详细介绍了SVRG及相关方差缩减方法。
- [Variance Reduction for Nonconvex Optimization](https://dl.acm.org/doi/10.5555/3157382.3157608) — Zeyuan Allen-Zhu, Yuanzhi Li (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS); Pages: 3394-3404; DOI: [10.5555/3157382.3157608](https://doi.org/10.5555/3157382.3157608)
  探讨了SVRG类方法在非凸优化中的理论性质和适用性，对深度学习领域有借鉴意义。
- [Variance Reduction for Stochastic Optimization: From Theory to Practice](https://doi.org/10.1561/2200000078) — Saeed Ghadimi and Guanghui Lan (2021)
  Journal: Foundations and Trends in Machine Learning; Publisher: Now Publishers; Volume: 14; Pages: 1-137; DOI: [10.1561/2200000078](https://doi.org/10.1561/2200000078)
  一项近期调查，广泛概述了包括SVRG在内的方差缩减技术的理论基础、算法和实践考量。

---

[上一节](02-%E9%9A%8F%E6%9C%BA%E5%B9%B3%E5%9D%87%E6%A2%AF%E5%BA%A6%20%28SAG%29.md) · [下一节](04-%E5%B0%8F%E6%89%B9%E9%87%8F%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%9A%84%E6%9D%83%E8%A1%A1.md)
