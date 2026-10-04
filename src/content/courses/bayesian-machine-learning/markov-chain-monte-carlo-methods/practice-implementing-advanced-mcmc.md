---
course: "bayesian-machine-learning"
chapter: "markov-chain-monte-carlo-methods"
lesson: "practice-implementing-advanced-mcmc"
sourceId: 3551
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-2-markov-chain-monte-carlo-methods/practice-implementing-advanced-mcmc"
title: "动手实践：实现高级MCMC"
description: "使用PyMC或Stan等库实现和诊断HMC或NUTS等MCMC采样器。"
order: 8
plots: ["plots/3551-0.json"]
sourceHash: "e41df7ccafb47fbf1ddc0bcd8288f696e1d2838e70c01d2a58ecbf62ccf28737"
sourceCorrections: []
---

高级马尔可夫链蒙特卡洛（MCMC）方法，如哈密顿蒙特卡洛（HMC）和无U形转弯采样器（NUTS），通常比 Metropolis-Hastings 或 Gibbs 采样等简单方法更高效，特别是对于复杂、高维的后验分布。这种效率源于其底层原理。使用现代概率编程库来实现和诊断这些采样器将被逐步演示。

我们将使用 Python 和流行的 PyMC 库，它为定义贝叶斯模型提供了直观的界面，并在内部使用强大的采样器（包括 NUTS）。我们还将使用 ArviZ 进行全面的 MCMC 诊断和可视化。

### 环境设置

首先，请确保已安装必要的库。通常可以使用 pip 进行安装：

```bash
pip install pymc arviz numpy pandas matplotlib seaborn
```

我们将使用这些库来定义模型、运行 NUTS 采样器并分析结果。

### 定义贝叶斯模型：一个分层示例

我们来处理一个高级MCMC表现出色的常见问题：一个分层模型。假设我们想对多个不同组间的关系（例如，线性回归）进行建模，允许回归参数 (parameter)因组而异，但假设这些组特定参数来自一个共同的总体分布。这种“部分合并”方法通常比为每个组拟合完全独立的模型或将所有数据合并在一起提供更好的估计。

以氡数据集为例，这是贝叶斯统计中的一个经典实例。它涉及测量明尼苏达州不同县房屋的室内氡气水平。我们可能希望将对数氡气水平建模为测量是否在地下室进行的函数，同时允许截距和斜率因县而异。

以下是我们可能在 PyMC 中指定此类分层线性模型的方式：

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

# 假设 'radon_data' 是一个 pandas DataFrame，包含以下列：
# 'log_radon'：氡测量值的对数
# 'floor'：0 表示地下室，1 表示一层
# 'county_idx'：表示县的整数索引
# 'n_counties'：唯一县的总数

# 加载或模拟数据（替换为实际数据加载）
# 示例占位符结构：
n_counties = 5 # 示例
n_obs_per_county = 20 # 示例
np.random.seed(42)
county_idx = np.repeat(np.arange(n_counties), n_obs_per_county)
floor = np.random.randint(0, 2, size=n_counties * n_obs_per_county)
# 模拟一些潜在的县级效应
true_county_intercept = np.random.normal(1.5, 0.5, size=n_counties)
true_county_slope = np.random.normal(-0.6, 0.2, size=n_counties)
log_radon = (true_county_intercept[county_idx] +
             true_county_slope[county_idx] * floor +
             np.random.normal(0, 0.3, size=n_counties * n_obs_per_county))

radon_data = pd.DataFrame({
    'log_radon': log_radon,
    'floor': floor,
    'county_idx': county_idx
})
n_counties = radon_data['county_idx'].nunique()
county_idx = radon_data['county_idx'].values
floor_measure = radon_data['floor'].values
log_radon_obs = radon_data['log_radon'].values

# 定义分层模型
with pm.Model() as hierarchical_model:
    # 县截距的超先验
    mu_a = pm.Normal('mu_a', mu=0., sigma=1.)
    sigma_a = pm.HalfCauchy('sigma_a', beta=1.)

    # 县斜率的超先验
    mu_b = pm.Normal('mu_b', mu=0., sigma=1.)
    sigma_b = pm.HalfCauchy('sigma_b', beta=1.)

    # 县特定截距（非中心化参数化）
    a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=n_counties)
    a = pm.Deterministic('a', mu_a + a_offset * sigma_a)

    # 县特定斜率（非中心化参数化）
    b_offset = pm.Normal('b_offset', mu=0, sigma=1, shape=n_counties)
    b = pm.Deterministic('b', mu_b + b_offset * sigma_b)

    # 模型误差项
    eps = pm.HalfCauchy('eps', beta=1.)

    # 基于县和楼层的期望值
    radon_est = a[county_idx] + b[county_idx] * floor_measure

    # 数据似然
    radon_like = pm.Normal('radon_like', mu=radon_est, sigma=eps, observed=log_radon_obs)
```

注意其结构：

- `mu_a`、`sigma_a`、`mu_b`、`sigma_b` 是管理县级参数分布的超先验。
- `a` 和 `b` 表示每个县的截距和斜率。我们使用非中心化参数化（`a_offset`、`b_offset`），这通常比直接从 `Normal(mu_a, sigma_a)` 采样 `a` 和 `b` 更能提升分层模型中采样器的效率。
- `eps` 是观测噪声标准差。
- `radon_like` 定义了似然，将模型预测（`radon_est`）与观测数据（`log_radon_obs`）关联起来。

### 运行 NUTS 采样器

模型定义好后，在 PyMC 中运行 NUTS 采样器非常简单。`pm.sample()` 函数会为连续模型智能地选择 NUTS。

```python
# 运行 NUTS 采样器
with hierarchical_model:
    idata = pm.sample(draws=2000, tune=1000, chains=4, cores=4, target_accept=0.9)
    # draws: 每条链要保留的样本数
    # tune: 每条链要丢弃的初始“预热”步数（用于调整采样器）
    # chains: 要运行的独立马尔可夫链数
    # cores: 要使用的 CPU 核心数（并行运行链）
    # target_accept: NUTS 步的期望平均接受概率（较高的值有时有助于处理复杂后验）
```

PyMC 执行采样过程，显示进度条和潜在警告（例如，关于发散或有效样本量低）。结果 `idata` (InferenceData) 是一个根据 ArviZ 约定构造的对象，包含后验样本、采样器统计数据、观测数据等。

### 分析采样器性能和收敛性

仅仅运行采样器是不够的。我们需要检查链是否收敛到目标分布，以及采样是否高效。ArviZ 为此提供了出色的工具。

#### 摘要统计：$\hat{R}$ 和 ESS

`az.summary()` 函数提供了每个参数 (parameter)的重要统计表：

```python
summary = az.summary(idata, var_names=['mu_a', 'sigma_a', 'mu_b', 'sigma_b', 'eps', 'a', 'b'])
print(summary)
```

请仔细看两列：

- `r_hat` ($\hat{R}$): Gelman-Rubin 诊断。它比较链内方差与链间方差。接近 1.0 的值（例如，< 1.01）表明收敛。远大于 1 的值表明链没有很好地混合，可能没有收敛到相同的分布。
- `ess_bulk` 和 `ess_tail`：有效样本量 (ESS)。它估计与 MCMC 获得的自相关样本等效的独立样本数量。低 ESS 值（相对于预热后总样本数）表示该参数的自相关性高和采样效率低。Bulk ESS 与分布中心相关（例如，用于估计均值），而 tail ESS 与尾部相关（例如，用于估计分位数）。为了获得可靠的估计，ESS 值应达到数百或数千。

#### 视觉诊断：轨迹图

轨迹图显示了每个链在迭代过程中参数的采样值。它们对于视觉检查收敛和混合非常有价值。

```python
az.plot_trace(idata, var_names=['mu_a', 'sigma_a', 'mu_b', 'sigma_b', 'eps'])
plt.tight_layout()
plt.show()
```

对于每个绘制的参数，你将看到两个面板：

1. \*\*左侧面板（轨迹）：\*\*显示所有链在迭代过程中的采样值。收敛的链应看起来像模糊的毛毛虫混合在一起，在没有长期趋势的情况下查看相同的取值范围。
2. \*\*右侧面板（后验密度）：\*\*显示参数的估计边际后验密度分布，结合了所有链的样本。

这里是轨迹图的可视化示例：



![轨迹图示例 (mu_a)](plots/3551-0.json)



> 参数 `mu_a` 的轨迹图示例。左侧显示两条链的迭代值；右侧显示组合的后验密度估计。

#### 其他有用图表

- `az.plot_posterior`：直接绘制带有可信区间的后验分布。
- `az.plot_pair`：为选定参数创建成对散点图矩阵，有助于找出后验中的相关性。
- `az.plot_energy`：HMC/NUTS 特有的图表，绘制能量转换诊断图，有助于找出采样困难。

### 结果解读与问题处理

一旦对收敛性满意，您可以使用 `idata.posterior` 中的后验样本来：

- 计算参数 (parameter)的后验均值、中位数和可信区间。
- 进行预测（使用 `pm.sample_posterior_predictive`）。
- 比较模型（将在课程后续部分介绍）。

#### 处理发散

NUTS 通常是可靠的，但您可能会遇到“发散”或“发散转换”。这些警告表明采样器遇到了数值难题，通常发生在后验中曲率高或几何形状复杂的区域。发散意味着样本可能存在偏差。常见策略包括：

- **增加 `target_accept`：** 将 `target_accept` 设置得更接近 1（例如，0.95，0.99）会强制使用更小的步长，可能更好地处理复杂的几何结构，但代价是采样速度变慢。
- **重新参数化：** 就像我们对 `a` 和 `b` 使用非中心化参数化那样，改变模型的数学指定方式（不改变底层模型）可以显著改善采样器的后验几何形状。
- **更强的先验：** 非常弱的先验有时可能导致参数空间中难以采样的区域。稍微更有信息的先验可能有助于限制采样器。
- **模型错配：** 持续的发散可能表明模型本身与数据不符或存在结构性问题。

### 结论

本次实践练习展示了如何使用 PyMC 为一个非平凡的分层贝叶斯模型实现高级 MCMC 方法（NUTS）。我们涵盖了模型定义、采样器运行，以及使用 ArviZ 执行重要的收敛性和性能诊断。请记住，MCMC 不是一个黑箱；需要通过摘要统计和视觉图表仔细分析输出，以确保后验推断的可靠性。掌握这些工具使您能够自信地将复杂的贝叶斯模型应用于难题。

## 参考资料

- [The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo](http://www.jmlr.org/papers/v15/hoffman14a.html) — Matthew D. Hoffman and Andrew Gelman (2014)
  Journal: Journal of Machine Learning Research; Publisher: Journal of Machine Learning Research; Volume: 15; Pages: 1593-1623; DOI: [10.5598/jmlr.v15.14-250](https://doi.org/10.5598/jmlr.v15.14-250)
  介绍了无U形转弯采样器（NUTS），这是本节讨论的一个重要的先进MCMC算法，提供了其理论基础。
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/abs/1701.02434) — Michael Betancourt (2017)
  Journal: arXiv preprint arXiv:1701.02434; DOI: [10.48550/arXiv.1701.02434](https://doi.org/10.48550/arXiv.1701.02434)
  提供了对哈密顿蒙特卡洛（HMC）及其NUTS等现代变体的清晰直观解释，侧重于其背后的几何原理。
- [Bayesian Data Analysis](https://www.crcpress.com/Bayesian-Data-Analysis-Third-Edition/Gelman-Carlin-Stern-Dunson-Vehtari-Rubin/p/book/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman and Hall/CRC
  一本全面的教科书（第三版），涵盖了贝叶斯推断、分层模型以及MCMC方法（包括诊断）的理论基础和应用。
- [PyMC Documentation](https://www.pymc.io/pymc-docs/en/stable/) — PyMC Developers (2024)
  PyMC的官方文档，这是一个开源的概率编程库，对于实践中讨论的MCMC方法的实现至关重要。
- [ArviZ Documentation](https://arviz.readthedocs.io/en/stable/) — ArviZ Developers (2024)
  ArviZ的官方文档，这是一个用于MCMC诊断、可视化和推断数据处理的Python库，对于分析采样器性能至关重要。
