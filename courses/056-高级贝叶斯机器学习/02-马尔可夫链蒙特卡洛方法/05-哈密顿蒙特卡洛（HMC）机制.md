# 哈密顿蒙特卡洛（HMC）机制

来源：[原文](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/hmc-mechanics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管标准的Metropolis-Hastings和Gibbs采样提供了MCMC的基本方法，但它们在高维或强相关后验分布中可能表现不佳。它们的提议通常类似随机游走，导致采样效率低和样本间自相关性高。哈密顿蒙特卡洛（HMC）通过借鉴经典物理学思想，特别是哈密顿动力学，提供了一种更有效的方式来采样参数 (parameter)空间。

可以将负对数后验密度想象成一个势能曲面。我们希望高效地在这个曲面进行采样。HMC引入了一个假想粒子在这个曲面上移动。粒子的位置对应于我们希望采样的参数$\theta$，我们引入一个辅助动量变量$p$，其维度通常与$\theta$相同。

### 哈密顿框架

在物理学中，一个系统的总能量，即哈密顿量$H$，是势能$U$和动能$K$之和。我们将其应用于统计目的：

1. **势能 $U(\theta)$：** 这被定义为目标密度的负对数。由于MCMC只需要目标密度达到一个比例常数，我们可以使用联合概率（似然乘以先验）的负对数，忽略证据项：

   
   $$
   U(\theta) = -\log[p(D|\theta)p(\theta)] = -\log p(\theta, D)
   $$
   

   最小化势能对应于最大化后验概率。
2. **动能 $K(p)$：** 这取决于辅助动量变量$p$和一个“质量矩阵”$M$，为简化起见，该矩阵通常假定为单位矩阵（$M=I$），除非我们有理由对不同维度进行不同缩放。

   
   $$
   K(p) = \frac{1}{2} p^T M^{-1} p
   $$
   

   如果$M=I$，这简化为$K(p) = \frac{1}{2} \sum_{i} p_i^2$。这对应于从多元高斯分布中采样动量，通常是$p \sim \mathcal{N}(0, M)$。
3. **哈密顿量 $H(\theta, p)$：** 我们假想系统的总“能量”是：

   
   $$
   H(\theta, p) = U(\theta) + K(p)
   $$
   

核心思想是根据哈密顿动力学模拟粒子的运动。在理想物理系统中，总能量$H(\theta, p)$随时间保持不变。粒子的轨迹由哈密顿方程支配：


$$
\frac{d\theta_i}{dt} = \frac{\partial H}{\partial p_i} = (M^{-1}p)_i
$$


$$
\frac{dp_i}{dt} = -\frac{\partial H}{\partial \theta_i} = -\frac{\partial U(\theta)}{\partial \theta_i}
$$


直观地，第一个方程表明动量会引起位置（参数 (parameter)）改变。第二个方程表明作用在粒子上的力（势能的负梯度，即对数后验的梯度）引起动量改变。请注意，计算动量变化需要对数后验的梯度$\nabla_{\theta} U(\theta)$。这是HMC的一个必要条件。

### 使用蛙跳积分器模拟动力学

对于复杂的后验分布，解析求解哈密顿方程通常是不可能的。相反，我们使用数值积分。一种常用且有效的方法是**蛙跳积分器**。它交替更新位置和动量，与欧拉方法等简单积分器相比，提供了更好的稳定性并保持了几何特性。

蛙跳算法以大小为$\epsilon$的离散时间步进行。为了从时间$t$移动到$t + \epsilon$，它执行三次更新：

1. **动量的半步更新：**
   
   $$
   p(t + \epsilon/2) = p(t) - \frac{\epsilon}{2} \nabla_{\theta} U(\theta(t))
   $$
   
2. **位置的全步更新：**
   
   $$
   \theta(t + \epsilon) = \theta(t) + \epsilon M^{-1} p(t + \epsilon/2)
   $$
   
3. **动量的半步更新：**
   
   $$
   p(t + \epsilon) = p(t + \epsilon/2) - \frac{\epsilon}{2} \nabla_{\theta} U(\theta(t + \epsilon))
   $$
   

我们将这些步骤重复$L$次，以模拟总时长为$L\epsilon$的轨迹。

### HMC算法

HMC的一次迭代过程如下：

1. **开始**于当前参数 (parameter)状态$\theta_{current}$。
2. **动量采样：** 从高斯分布中抽取一个随机动量向量 (vector)$p$，例如，$p \sim \mathcal{N}(0, M)$。
3. **模拟轨迹：** 从$(\theta_{current}, p)$开始，运行蛙跳积分器$L$步，步长为$\epsilon$。令轨迹结束时的状态为$(\theta', p')$。
4. **Metropolis接受步骤：** 由于蛙跳积分器引入小的数值误差，哈密顿量$H$不会完美守恒。我们使用Metropolis接受步骤对此进行修正。计算接受概率：
   
   $$
   \alpha = \min\left(1, \frac{\exp(-H(\theta', p'))}{\exp(-H(\theta_{current}, p))}\right) = \min\left(1, \frac{p(\theta', D)\exp(-K(p'))}{p(\theta_{current}, D)\exp(-K(p))}\right)
   $$
   
   请注意，目标密度$p(\theta, D)$直接出现在这里。
5. **接受或拒绝：** 抽取$u \sim \text{Uniform}(0, 1)$。如果$u < \alpha$，接受提议的状态：$\theta_{next} = \theta'$。否则，拒绝提议：$\theta_{next} = \theta_{current}$。
6. **返回：** 马尔可夫链中的下一个样本是$\theta_{next}$。动量变量$p$被丢弃；它仅用于生成提议。在下一次迭代开始时会抽取一个新的动量。



[交互图表：HMC在势能曲面上的轨迹类比](https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/hmc-mechanics#plot-d9pcgz)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "HMC在势能曲面上的轨迹类比",
    "xaxis": {
      "title": "参数 θ₁",
      "range": [
        -3,
        3
      ]
    },
    "yaxis": {
      "title": "参数 θ₂",
      "range": [
        -3,
        3
      ]
    },
    "width": 500,
    "height": 450
  },
  "data": [
    {
      "type": "contour",
      "z": [
        [
          8,
          7,
          6,
          5,
          6,
          7,
          8
        ],
        [
          7,
          4,
          3,
          2,
          3,
          4,
          7
        ],
        [
          6,
          3,
          1,
          0,
          1,
          3,
          6
        ],
        [
          5,
          2,
          0,
          -1,
          0,
          2,
          5
        ],
        [
          6,
          3,
          1,
          0,
          1,
          3,
          6
        ],
        [
          7,
          4,
          3,
          2,
          3,
          4,
          7
        ],
        [
          8,
          7,
          6,
          5,
          6,
          7,
          8
        ]
      ],
      "x": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "y": [
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3
      ],
      "colorscale": "Blues",
      "reversescale": true,
      "contours": {
        "coloring": "heatmap"
      },
      "showscale": false
    },
    {
      "type": "scatter",
      "x": [
        -1.5,
        -1.2,
        -0.7,
        0.1,
        0.6,
        1.0,
        1.3,
        1.5
      ],
      "y": [
        1.5,
        1.8,
        1.9,
        1.7,
        1.3,
        0.8,
        0.2,
        -0.3
      ],
      "mode": "lines+markers",
      "line": {
        "color": "#f03e3e",
        "width": 2
      },
      "marker": {
        "color": "#f03e3e",
        "size": 6
      },
      "name": "HMC轨迹"
    }
  ]
}
```

</details>



> 一张HMC轨迹穿越势能曲面（代表负对数后验）等高线的示意图。轨迹在不同的参数值之间移动，同时倾向于停留在能量相似的区域。

### HMC为何有效

HMC使用梯度信息$\nabla_{\theta} U(\theta)$来智能地引导其提议。它不是进行随机游走，而是依据后验形状进行提议移动。通过模拟动力学进行多个步长$L$，HMC可以生成远离当前状态但仍具有高接受概率的提议，特别是如果$\epsilon$选择适当。这大幅降低了样本间的自相关性，并且与随机游走Metropolis或Gibbs采样相比，可以更快地采样目标分布，尤其是在处理大量参数 (parameter)或参数间存在强相关时。

### 实际考量

- **梯度计算：** HMC需要对数后验的梯度。现代概率编程语言（PPLs）如Stan、PyMC、NumPyro和TensorFlow Probability使用自动微分来高效准确地计算这些梯度，使HMC在复杂模型中变得实用。
- **调优参数 (parameter)：** HMC引入了两个主要调优参数：步长$\epsilon$和步数$L$。
  - $\epsilon$：控制离散化误差。$\epsilon$过大会导致哈密顿模拟不准确，从而接受率低。$\epsilon$过小会使模拟变慢，且需要大量步数$L$才能显著移动。
  - $L$：控制模拟轨迹的长度。$L$过小会导致类似随机游走的行为（高相关性）。$L$过大会导致轨迹折返，造成计算浪费。
    寻找最优的$\epsilon$和$L$通常需要一些实验或自适应方法。

HMC代表了MCMC技术的重要进展。它利用梯度进行高效采样的能力使其成为许多现代贝叶斯推断问题的核心算法。理解其机制有助于理解这些高级采样器的工作原理，并为理解自适应扩展（如No-U-Turn Sampler (NUTS)）奠定了基础，NUTS能自动化$L$的调优。

## 参考资料

- [Hybrid Monte Carlo](http://www.sciencedirect.com/science/article/B6TVN-46YSWPH-2XF/2/0f89cdc6cf214a2169b03df7414f3df4) — Simon Duane, A. D. Kennedy, Brian J. Pendleton, and Duncan Roweth (1987)
  Journal: Physics Letters B; Publisher: Elsevier; Volume: 195; Pages: 216-222; DOI: [10.1016/0370-2693(87)91197-X](https://doi.org/10.1016/0370-2693%2887%2991197-X)
  介绍混合蒙特卡洛的原始论文，是HMC的基础。
- [MCMC Using Hamiltonian Dynamics](https://arxiv.org/pdf/1206.1901.pdf) — Radford M. Neal (2011)
  Journal: Handbook of Markov Chain Monte Carlo; Publisher: Chapman & Hall/CRC; Pages: 113-162; DOI: [10.1201/b10905-6](https://doi.org/10.1201/b10905-6)
  一篇被广泛引用的章节，详细讨论了HMC的理论与实践。
- [Bayesian Data Analysis](https://doi.org/10.1201/b16018) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  一本贝叶斯统计学标准教材，全面涵盖包括HMC在内的MCMC方法。
- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://www.jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) — Matthew D. Hoffman and Andrew Gelman (2014)
  Journal: Journal of Machine Learning Research; Publisher: Microtome Publishing, LLC.; Volume: 15; Pages: 1593-1623; DOI: [10.1162/jmlr.2014.15.1593](https://doi.org/10.1162/jmlr.2014.15.1593)
  介绍了NUTS，一种自适应HMC方法，可自动调整模拟长度，应用于现代PPL中。

---

[上一节](04-%E5%9F%BA%E4%BA%8E%E6%9D%A1%E4%BB%B6%E7%BB%93%E6%9E%84%E7%9A%84%E5%90%89%E5%B8%83%E6%96%AF%E9%87%87%E6%A0%B7.md) · [下一节](06-%E6%97%A0U%E5%BD%A2%E8%BD%AC%E5%BC%AF%E9%87%87%E6%A0%B7%E5%99%A8%20%28NUTS%29.md)
