---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "mean-field-approximation-details"
sourceId: 3561
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/mean-field-approximation-details"
title: "平均场近似具体内容"
description: "分析变分推断中平均场近似的假设和影响。"
order: 3
plots: ["plots/3561-0.json"]
sourceHash: "7c21741f2aa704a752783d60e2c0c183efcc7235ee235ea02706a644fce84487"
sourceCorrections: []
---

变分推断 (VI) 将计算后验分布 $p(\mathbf{z}|\mathbf{x})$ 的难题转化为一个优化问题。我们的目标是找到一个在选定的分布族 $\mathcal{Q}$ 中最能近似真实后验分布的 $q(\mathbf{z})$，通常通过最大化证据下界 (ELBO) 来实现：


$$
\mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]
$$


主要步骤在于选择分布族 $\mathcal{Q}$。如果我们允许 $\mathcal{Q}$ 包含*所有*可能的分布，那么最优的 $q^*(\mathbf{z})$ 将是真实后验分布 $p(\mathbf{z}|\mathbf{x})$ 本身，这会使我们回到最初难以处理的问题。因此，实用变分推断的要点在于为 $q(\mathbf{z})$ 选择一个受限且易于处理的分布族。

## 平均场假设：因子化复杂性

最常用的策略之一是**平均场变分族**。这种方法通过一个重要的简化，即假设近似分布 $q(\mathbf{z})$ 中的隐变量 $\mathbf{z} = (z_1, z_2, ..., z_M)$ 相互独立。在数学上，我们强制执行完全因子化结构：


$$
q(\mathbf{z}) = \prod_{j=1}^{M} q_j(z_j; \lambda_j)
$$


这里，联合变分分布 $q(\mathbf{z})$ 被分解为 $M$ 个独立因子的乘积，其中每个因子 $q_j(z_j; \lambda_j)$ 仅支配单个隐变量 $z_j$（有时也支配一组变量，尽管完全因子化更常见）。每个因子 $q_j$ 本身都是一个概率分布，通常属于简单的参数 (parameter)族（如高斯分布或狄利克雷分布），并由其自身的变分参数集 $\lambda_j$ 参数化。变分推断的目标于是变为找到最优参数 $\{\lambda_j\}_{j=1}^M$ 以最大化 ELBO。

## 简化ELBO

这种因子化假设对 ELBO 项有直接影响。熵项 $\mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$ 变得特别易于处理：


$$
\mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})] = \mathbb{E}_{q(\mathbf{z})} \left[ \sum_{j=1}^{M} \log q_j(z_j) \right] = \sum_{j=1}^{M} \mathbb{E}_{q_j(z_j)} [\log q_j(z_j)]
$$


和的期望变为期望的和，并且由于每个 $q_j$ 仅依赖于 $z_j$，期望 $\mathbb{E}_{q(\mathbf{z})}$ 对于第 $j$ 项简化为 $\mathbb{E}_{q_j(z_j)}$。这意味着总熵就是各个因子单独熵的和。如果因子 $q_j$ 属于标准指数族，它们的熵通常具有闭合形式的表达式，使得这一项易于计算。

第一项，$\mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})]$，也得到简化。期望是在因子化分布 $q(\mathbf{z})$ 上进行计算的，这意味着我们根据每个 $z_j$ 的因子 $q_j(z_j)$ 对其进行积分或求和。具体的计算在很大程度上取决于模型联合概率 $p(\mathbf{x}, \mathbf{z})$ 的结构。

## 推导最优因子

我们如何找到每个因子 $q_j(z_j)$ 的最优形式？我们可以相对于一个因子 $q_j$ 优化 ELBO，同时保持其他因子（$q_k$ 对于 $k \neq j$）固定。这是诸如坐标上升变分推断 (CAVI) 等算法的中心思想，我们将在下一节中进行讨论。

让我们分离出 ELBO 中依赖于特定因子 $q_j(z_j)$ 的项。我们可以将 ELBO 改写为：


$$
\mathcal{L}(q) = \int q_j(z_j) \left( \int \prod_{k \neq j} q_k(z_k) \log p(\mathbf{x}, \mathbf{z}) dz_{-j} \right) dz_j - \int q_j(z_j) \log q_j(z_j) dz_j + \text{不依赖于 } q_j \text{ 的项}
$$


其中 $dz_{-j}$ 表示对除 $z_j$ 之外的所有变量 $\mathbf{z}$ 进行积分。第一个积分中括号内的项是对除 $q_j$ *之外*所有因子求对数联合概率的期望：


$$
\mathbb{E}_{q_{-j}} [\log p(\mathbf{x}, \mathbf{z})] \triangleq \int \log p(\mathbf{x}, \mathbf{z}) \prod_{k \neq j} q_k(z_k) dz_{-j}
$$


这个期望 $\mathbb{E}_{q_{-j}} [\log p(\mathbf{x}, \mathbf{z})]$ 得到一个依赖于 $z_j$（以及 $\mathbf{x}$，和固定的 $q_k$ 的参数 (parameter)）的函数。我们记作 $\log \tilde{p}_j(z_j) = \mathbb{E}_{q_{-j}} [\log p(\mathbf{x}, \mathbf{z})]$。那么 ELBO 中依赖于 $q_j(z_j)$ 的项为：


$$
\mathcal{L}_j = \int q_j(z_j) \log \tilde{p}_j(z_j) dz_j - \int q_j(z_j) \log q_j(z_j) dz_j
$$


这个表达式看起来很熟悉。它等于 $q_j(z_j)$ 与一个非归一化 (normalization)分布（与 $\tilde{p}_j(z_j)$ 成比例）之间的负 Kullback-Leibler (KL) 散度，加上一个常数（$\tilde{p}_j(z_j)$ 的对数归一化常数）：


$$
\mathcal{L}_j = -KL(q_j(z_j) || \frac{\tilde{p}_j(z_j)}{C_j}) + \log C_j \quad \text{其中 } C_j = \int \tilde{p}_j(z_j) dz_j
$$


最大化 $\mathcal{L}_j$（从而在保持其他因子固定的情况下最大化相对于 $q_j$ 的完整 ELBO）等同于最小化 KL 散度 $KL(q_j(z_j) || \frac{\tilde{p}_j(z_j)}{C_j})$。当 $q_j(z_j)$ 恰好等于归一化分布 $\frac{\tilde{p}_j(z_j)}{C_j}$ 时，KL 散度的最小值（零）被达到。因此，因子 $q_j^*(z_j)$ 的最优解满足：


$$
q_j^*(z_j) \propto \exp(\mathbb{E}_{q_{-j}} [\log p(\mathbf{x}, \mathbf{z})])
$$


或者，等价地：


$$
\log q_j^*(z_j) = \mathbb{E}_{q_{-j}} [\log p(\mathbf{x}, \mathbf{z})] + \text{常数}
$$


这个重要结果为找到每个因子 $q_j^*(z_j)$ 的最优形式提供了一种方法，假设所有其他因子 $q_k(z_k)$（$k \neq j$）是固定的。它表明 $z_j$ 的最优对数密度是通过取模型联合概率 $p(\mathbf{x}, \mathbf{z})$ 的对数，然后根据其他变量 $z_k$（$k \neq j$）当前的变分分布 $q_k(z_k)$ 对它们进行平均而获得的。这构成了像 CAVI 这样的迭代更新方案的依据，其中我们循环遍历变量，根据其他变量的当前估计值更新每个 $q_j$。

## 平均场的影响与局限性

平均场近似的主要优势是**计算上的易处理性**。通过打破变分近似 $q$ 中隐变量之间的依赖结构，我们将一个潜在的复杂高维分布优化问题转化为一系列针对低维因子 $q_j$ 的潜在更简单的优化问题。如果模型结构和 $q_j$ 族的选择是兼容的（例如，在模型中使用共轭先验通常会导致 $q_j^*$ 具有可识别的形式），这些更新有时可以解析推导出来。

然而，这种简化是有代价的。主要的假设是变分后验因子是独立的：$q(\mathbf{z}) = \prod_j q_j(z_j)$。如果真实后验分布 $p(\mathbf{z}|\mathbf{x})$ 表现出变量 $z_j$ 之间明显的关联，那么平均场近似 $q(\mathbf{z})$ 根据定义将无法捕捉这些依赖关系。

考虑一个简单的二维高斯分布例子。如果真实后验分布在 $z_1$ 和 $z_2$ 之间显示出强的负相关，最好的平均场近似 $q(z_1, z_2) = q_1(z_1) q_2(z_2)$ 将是一个零相关的高斯分布（其等高线将与坐标轴对齐 (alignment)），即使它能正确地定中心并近似边际方差。



![平均场与真实后验的比较](plots/3561-0.json)



> 图像比较具有相关性的真实后验分布 $p(z_1, z_2|\mathbf{x})$（橙色虚线等高线）及其最优平均场近似 $q(z_1)q(z_2)$（蓝色实线等高线）。该近似能够正确地定中心，但强制独立性，因此未能捕捉真实后验分布中存在的关联结构。

这种无法捕捉后验相关性的情况是平均场变分推断的一个众所周知的特点。由于 KL 散度 $KL(q || p)$ 会惩罚 $q$ 在 $p$ 没有概率质量的地方放置概率质量，因此生成的 $q(\mathbf{z})$ 往往比真实后验分布 $p(\mathbf{z}|\mathbf{x})$ 更“紧凑”或更集中在众数附近。这通常会导致**后验方差的低估**以及潜在的过度自信的不确定性估计。

尽管存在这些局限性，平均场假设在变分推断中具有基础地位。其计算优势显著，使得贝叶斯推断对许多复杂模型和大型数据集变得可行，而 MCMC 方法可能在收敛时间或内存需求方面遇到困难。理解平均场近似、通过最优因子更新的推导及其在后验依赖性和方差估计方面的固有局限性，对于有效应用变分推断并批判性地解释其结果非常重要。我们现在将研究坐标上升变分推断 (CAVI) 算法，该算法直接实现从平均场假设推导出的迭代更新。

## 参考资料

- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 10
  这本标准教材对变分推断进行了完整呈现，涵盖了ELBO、均场近似及其数学推导。
- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  这篇综述对变分推断进行了易于理解且精确的讨论，涵盖了均场框架、其理论背景和各种用途。
- [An Introduction to Variational Methods for Graphical Models](https://doi.org/10.1023/A:1007665905271) — Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola, Lawrence K. Saul (1999)
  Journal: Machine Learning; Publisher: Kluwer Academic Publishers; Volume: 37; Pages: 183-233; DOI: [10.1023/A:1007665905271](https://doi.org/10.1023/A%3A1007665905271)
  这篇论文将变分方法（包括均场近似）确立为复杂概率模型中进行推断的框架。
