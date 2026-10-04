# Metropolis-Hastings 算法变体

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/metropolis-hastings-variants)

[返回章节目录](README.md) · [返回课程目录](../README.md)

标准的 Metropolis-Hastings (MH) 算法提供了一个通用框架，用于在能够评估与其成比例的函数（例如似然与先验的乘积 $p(D|\theta)p(\theta)$）时，从目标分布（通常是后验 $p(\theta | D)$）中进行采样。其基本思路是，从一个提议分布 $q(\theta' | \theta^{(t)})$ 中提议一个新状态 $\theta'$，并以如下概率接受它：


$$
\alpha(\theta', \theta^{(t)}) = \min \left( 1, \frac{p(\theta' | D) q(\theta^{(t)} | \theta')}{p(\theta^{(t)} | D) q(\theta' | \theta^{(t)})} \right)
$$


如果接受，则 $\theta^{(t+1)} = \theta'$。否则，$\theta^{(t+1)} = \theta^{(t)}$。尽管强大，但基本 MH 算法的效率和效用很大程度上取决于提议分布 $q$ 的选择。这导致出现了几种变体，旨在不同情境下提升性能。

### 随机游走 Metropolis (RWM)

随机游走 Metropolis 算法也许是最简单和最常用的变体。它使用对称提议分布，意思是，从 $\theta$ 开始提议 $\theta'$ 的概率与从 $\theta'$ 开始提议 $\theta$ 的概率相同。形式上， $q(\theta' | \theta) = q(\theta | \theta')$。

一个典型选择是以当前状态为中心的高斯分布：


$$
\theta' \sim \mathcal{N}(\theta^{(t)}, \Sigma)
$$


其中 $\Sigma$ 是一个协方差矩阵，通常是对角的，例如 $\sigma^2 I$。由于提议是对称的，提议项在接受率中抵消：


$$
q(\theta^{(t)} | \theta') = q(\theta' | \theta^{(t)})
$$


这将接受概率简化为：


$$
\alpha(\theta', \theta^{(t)}) = \min \left( 1, \frac{p(\theta' | D)}{p(\theta^{(t)} | D)} \right)
$$


这是原始的 Metropolis 算法（在 Hastings 泛化之前）。

**优点：**

- 易于实现。
- 在提议尺度（例如 $\sigma^2$）上无需复杂调整。

**缺点：**

- 如果目标分布具有复杂的相关性或维度间的尺度变化，则效率不高。随机游走行为可能导致参数 (parameter)空间的缓慢遍历。
- 性能对提议尺度的选择（$\sigma^2$ 或 $\Sigma$）非常敏感。

**提议尺度的调整：**
为随机游走选择合适的尺度很重要。

- **太小：** 大多数提议会被接受，但链移动非常慢，在空间中移动效率不高，导致样本之间的高度自相关。
- **太大：** 许多提议会落入低概率区域并被拒绝。链会卡住，再次导致混合不佳。

对于高维问题，最佳接受率经验上通常约为 0.234，对于一维问题约为 0.44，不过这只是一个指导原则，而非严格规定。调整通常涉及运行试点模拟，并调整提议尺度以达到合理的接受率。



[交互图表：不同步长的 RWM 遍历](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/metropolis-hastings-variants#plot-1mc9msr)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "xaxis": {
      "title": "参数 1",
      "range": [
        -4,
        4
      ]
    },
    "yaxis": {
      "title": "参数 2",
      "range": [
        -4,
        4
      ]
    },
    "title": "不同步长的 RWM 遍历"
  },
  "data": [
    {
      "x": [
        0,
        0.1,
        0.15,
        0.25,
        0.2
      ],
      "y": [
        0,
        0.05,
        -0.05,
        0.0,
        -0.1
      ],
      "mode": "lines+markers",
      "name": "小步长（高接受率，慢混合）",
      "line": {
        "color": "#1c7ed6"
      },
      "marker": {
        "color": "#1c7ed6"
      }
    },
    {
      "x": [
        0,
        0.8,
        1.5,
        1.2,
        2.1
      ],
      "y": [
        0,
        0.5,
        -0.2,
        0.9,
        0.7
      ],
      "mode": "lines+markers",
      "name": "良好步长（平衡）",
      "line": {
        "color": "#37b24d"
      },
      "marker": {
        "color": "#37b24d"
      }
    },
    {
      "x": [
        0,
        2.5,
        2.4,
        -1.8,
        -1.9
      ],
      "y": [
        0,
        -1.5,
        -1.6,
        2.2,
        2.1
      ],
      "mode": "lines+markers",
      "name": "大步长（低接受率，卡住）",
      "line": {
        "color": "#f03e3e",
        "dash": "dot"
      },
      "marker": {
        "color": "#f03e3e"
      }
    }
  ]
}
```

</details>



> 随机游走 Metropolis 路径的图示。小步长导致缓慢遍历，大步长导致频繁拒绝（停留在同一位置，由重叠标记 (token)指示），而调整得当的步长则平衡了接受和移动。

### 独立 Metropolis-Hastings

在此变体中，提议的状态 $\theta'$ 是从一个不依赖于当前状态 $\theta^{(t)}$ 的分布 $q(\theta')$ 中抽取的。


$$
\theta' \sim q(\theta')
$$


这意味着 $q(\theta' | \theta^{(t)}) = q(\theta')$。接受概率使用完整的 Hastings 比率，因为提议通常不对称：


$$
\alpha(\theta', \theta^{(t)}) = \min \left( 1, \frac{p(\theta' | D) q(\theta^{(t)})}{p(\theta^{(t)} | D) q(\theta')} \right)
$$


**要求：**
为了使此方法良好运行，提议分布 $q(\theta')$ 理想情况下应是目标后验 $p(\theta | D)$ 的良好近似。常见选择包括多元高斯分布或学生 t 分布，它们可以拟合后验众数（通过优化得到）或基于初始近似（如拉普拉斯近似）。

**优点：**

- 如果 $q(\theta')$ 是 $p(\theta | D)$ 的良好近似，采样器可以在参数 (parameter)空间中进行大步跳跃，可能导致比 RWM 快得多的混合。
- 如果后验的合理近似可用，则可能有效。

**缺点：**

- 寻找一个好的提议分布 $q(\theta')$ 可能具有挑战性。
- 如果 $q(\theta')$ 是不好的近似，特别是如果其尾部比目标后验 $p(\theta | D)$ 轻，采样器可能很少在状态空间的重要区域接受提议，导致非常差的性能。必须满足 $p(\theta|D) > 0 \implies q(\theta) > 0$ 的要求。
- 如果 $q$ 与 $p$ 不相似，接受率可能非常低。

### Metropolis-调整朗之万算法 (MALA)

MALA 结合了来自目标分布的梯度信息，以更智能地提议移动，将样本推向更高概率的区域。这在高维空间 (high-dimensional space)中特别有用，因为随机游走效率不高。

提议机制基于朗之万动力学，它涉及一个基于对数后验梯度的漂移项和一个随机噪声项。离散时间近似导致了以下提议：


$$
\theta' \sim \mathcal{N}\left( \theta^{(t)} + \frac{\epsilon^2}{2} \nabla \log p(\theta^{(t)} | D), \epsilon^2 I \right)
$$


这里，$\epsilon$ 是一个步长参数 (parameter)，$\nabla \log p(\theta | D)$ 是对数目标密度（对数似然 + 对数先验）的梯度。此提议将 $\theta'$ “向上”推向更高密度的区域。

因为提议依赖于当前点 $\theta^{(t)}$ 处的梯度，所以它不对称。提议密度为：


$$
q(\theta' | \theta^{(t)}) = \mathcal{N}\left(\theta' \mid \mu(\theta^{(t)}), \epsilon^2 I\right)
$$


其中 $\mu(\theta^{(t)}) = \theta^{(t)} + \frac{\epsilon^2}{2} \nabla \log p(\theta^{(t)} | D)$。必须使用完整的 Hastings 接受率：


$$
\alpha(\theta', \theta^{(t)}) = \min \left( 1, \frac{p(\theta' | D) q(\theta^{(t)} | \theta')}{p(\theta^{(t)} | D) q(\theta' | \theta^{(t)})} \right)
$$


**优点：**

- 通过使用梯度信息，它考察参数空间的效率通常比 RWM 高得多，特别是在更高维度。
- 比简单的随机游走更好地处理复杂的后验几何结构。

**缺点：**

- 需要计算梯度 $\nabla \log p(\theta | D)$，这对于所有模型来说可能无法解析获得或在计算上不可行。自动微分工具可以在这方面提供帮助。
- 需要调整步长 $\epsilon$。类似于 RWM，$\epsilon$ 需要在良好接受率和遍历之间取得平衡。MALA 的最佳接受率通常建议约为 0.574。
- 如果 $\epsilon$ 过大，离散时间近似仍然可能导致效率不高。

MALA 为哈密顿蒙特卡洛 (HMC) 搭建了桥梁，HMC 通过引入辅助动量变量和模拟哈密顿动力学，更有效地使用梯度信息。

### 其他变体

- **自适应 MCMC：** 这些算法在模拟期间自动调整提议参数 (parameter)（例如 RWM 中的协方差矩阵 $\Sigma$ 或 MALA 中的步长 $\epsilon$），通常在初始的“预热”或“热身”阶段进行。尽管强大，但必须小心确保适应性随时间停止或减弱，以保持马尔可夫链的正确平稳分布保证。
- **分量式 Metropolis-Hastings：** 不是一次性更新所有参数 $\theta = (\theta_1, ..., \theta_d)$，而是一次更新一个参数或一组参数，以其他参数的当前值为条件。这与 Gibbs 采样密切相关，后者将在下一节中讨论。

选择合适的 Metropolis-Hastings 变体涉及在实现复杂性、计算成本（尤其是梯度）、调整工作量和针对特定问题结构的采样效率之间进行权衡。对于复杂、高维的后验，结合梯度信息的方法，例如 MALA（以及后来的 HMC/NUTS），通常比基本的随机游走 Metropolis 明显更有效。

## 参考资料

- [Bayesian Data Analysis](https://www.taylorandfrancis.com/books/9780429113079) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  权威教材，涵盖贝叶斯建模的理论和实践，并广泛讨论了 MCMC 算法，包括 Metropolis-Hastings 及其高级变体。
- [Optimal Scaling of Discrete Approximations to Langevin Diffusions](https://academic.oup.com/jrsssb/article/60/1/255/7099238) — Gareth O. Roberts, Jeffrey S. Rosenthal (1998)
  Journal: Journal of the Royal Statistical Society: Series B (Statistical Methodology); Publisher: John Wiley & Sons; Volume: 60; Pages: 255-268; DOI: [10.1111/1467-9868.00123](https://doi.org/10.1111/1467-9868.00123)
  一篇开创性论文，介绍了 Metropolis-Adjusted Langevin 算法 (MALA) 并严谨分析了其在维度空间中的最佳接受率和尺度行为。
- [Monte Carlo Statistical Methods](https://doi.org/10.1007/978-1-4757-4145-1) — Christian P. Robert and George Casella (2004)
  Publisher: Springer; DOI: [10.1007/978-1-4757-4145-1](https://doi.org/10.1007/978-1-4757-4145-1)
  本教材为蒙特卡洛方法提供了全面的理论基础，其中有专门章节详细解释了 Metropolis-Hastings 及其变体等马尔可夫链蒙特卡洛算法。
- [Examples of adaptive MCMC](https://doi.org/10.1198/jcgs.2009.06134) — Gareth O. Roberts, Jeffrey S. Rosenthal (2009)
  Journal: Journal of Computational and Graphical Statistics; Publisher: Taylor & Francis; Volume: 18; Pages: 349-367; DOI: [10.1198/jcgs.2009.06134](https://doi.org/10.1198/jcgs.2009.06134)
  讨论了自适应 Metropolis-Hastings 算法的设计和特性，这些算法对于自动调整提议分布以提高效率至关重要。

---

[上一节](02-%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E9%93%BE%E7%90%86%E8%AE%BA%E4%B9%8B%E4%BA%8EMCMC.md) · [下一节](04-%E5%9F%BA%E4%BA%8E%E6%9D%A1%E4%BB%B6%E7%BB%93%E6%9E%84%E7%9A%84%E5%90%89%E5%B8%83%E6%96%AF%E9%87%87%E6%A0%B7.md)
