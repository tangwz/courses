---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "bbvi-algorithm"
sourceId: 3571
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/bbvi-algorithm"
title: "黑箱变分推断 (BBVI)"
description: "当 ELBO 梯度难以解析计算时，可使用 BBVI 来处理模型。"
order: 6
plots: []
sourceHash: "d8fd70b5e48b22d4cae8770c498277ffec4a0eb6d5dbdf37a0831b82d444a1ac"
sourceCorrections: []
---

变分推断方法在推导证据下界 (ELBO) 梯度的模型特定解析表达式时，经常面临困难甚至不可能的情况。例如，坐标上升变分推断 (CAVI) 在使用共轭模型和平均场近似时，能为变分参数 (parameter) $\lambda$ 提供解析更新。随机变分推断 (SVI) 则通过随机梯度实现可伸缩性。然而，这两种方法都通常依赖于此类解析梯度推导。这种情形经常出现在以下情况：

- **非共轭模型：** 当先验和似然不构成共轭对时，CAVI 中变分参数的更新可能没有闭式解。
- **复杂的模型结构：** 涉及复杂依赖关系、模拟器或生成过程中函数的模型，常常使得联合概率 $p(\mathbf{x}, \mathbf{z})$ 难以对变分参数 $\lambda$ 进行解析求导。

黑箱变分推断 (BBVI) 提供了一种通用的变分推断方法，可以克服这一局限。它使得我们即使无法解析计算 ELBO 的梯度 $\nabla_{\lambda} \mathcal{L}(\lambda)$，也能进行变分推断。主要思想是使用蒙特卡洛估计来近似此梯度。BBVI 将模型本身视为一个“黑箱”，仅要求能够评估对数联合概率 $\log p(\mathbf{x}, \mathbf{z})$ 和对数变分密度 $\log q(\mathbf{z}|\lambda)$，以及在特定条件下的它们的梯度。

### 估计 ELBO 梯度

回顾 ELBO：
$\mathcal{L}(\lambda) = \mathbb{E}_{q(\mathbf{z}|\lambda)} [\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\lambda)]$

我们需要计算 $\nabla_{\lambda} \mathcal{L}(\lambda)$。难题在于期望是针对 $q(\mathbf{z}|\lambda)$ 计算的，而 $q(\mathbf{z}|\lambda)$ 本身依赖于 $\lambda$。

#### 评分函数梯度估计器

一种可行的方法是使用对数导数技巧，也称为评分函数恒等式：$\nabla_{\lambda} q(\mathbf{z}|\lambda) = q(\mathbf{z}|\lambda) \nabla_{\lambda} \log q(\mathbf{z}|\lambda)$。将此应用于 ELBO 梯度，得到：

$\nabla_{\lambda} \mathcal{L}(\lambda) = \nabla_{\lambda} \int q(\mathbf{z}|\lambda) [\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\lambda)] d\mathbf{z}$
$= \int \nabla_{\lambda} q(\mathbf{z}|\lambda) [\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\lambda)] d\mathbf{z} + \int q(\mathbf{z}|\lambda) \nabla_{\lambda} [\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\lambda)] d\mathbf{z}$

假设模型的联合概率 $p(\mathbf{x}, \mathbf{z})$ 不依赖于变分参数 (parameter) $\lambda$，则 $\log p(\mathbf{x}, \mathbf{z})$ 关于 $\lambda$ 的梯度为零。在应用对数导数技巧和进一步的推导后，表达式简化为：

$\nabla_{\lambda} \mathcal{L}(\lambda) = \mathbb{E}_{q(\mathbf{z}|\lambda)} [ (\nabla_{\lambda} \log q(\mathbf{z}|\lambda)) (\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z}|\lambda)) ]$

这个期望可以使用蒙特卡洛样本近似：抽取 $S$ 个样本 $\mathbf{z}^{(s)} \sim q(\mathbf{z}|\lambda)$ 并计算：

$\nabla_{\lambda} \mathcal{L}(\lambda) \approx \frac{1}{S} \sum_{s=1}^S (\nabla_{\lambda} \log q(\mathbf{z}^{(s)}|\lambda)) (\log p(\mathbf{x}, \mathbf{z}^{(s)}) - \log q(\mathbf{z}^{(s)}|\lambda))$

这是评分函数梯度估计器，有时在强化学习 (reinforcement learning)背景下被称为 REINFORCE 估计器。要使用它，我们只需能够从 $q(\mathbf{z}|\lambda)$ 采样，评估 $\log p(\mathbf{x}, \mathbf{z})$、$\log q(\mathbf{z}|\lambda)$，并计算 $\log q(\mathbf{z}|\lambda)$ 关于 $\lambda$ 的梯度。

评分函数估计器的一个主要缺点是它常表现出高方差，需要大量样本 $S$ 才能获得稳定估计，这会减慢收敛速度。

#### 重参数化技巧 (路径导数估计器)

当变分分布 $q(\mathbf{z}|\lambda)$ 适用于*重参数化技巧*时，可以使用另一种通常方差更小的梯度估计器。这涉及将随机变量 $\mathbf{z}$ 表示为辅助随机变量 $\epsilon$ 的一个确定性且可微分的变换 $g(\cdot, \lambda)$，其中 $\epsilon$ 具有一个不依赖于 $\lambda$ 的固定分布 $p(\epsilon)$。

$\mathbf{z} = g(\epsilon, \lambda), \quad \text{其中 } \epsilon \sim p(\epsilon)$

例如，如果 $q(z|\lambda)$ 是高斯分布 $N(\mu, \sigma^2)$，其中 $\lambda = \{\mu, \sigma\}$，我们可以将 $z$ 重参数化为 $z = \mu + \sigma \epsilon$，其中 $\epsilon \sim N(0, 1)$。

通过这种重参数化，我们可以重写 ELBO 期望：
$\mathcal{L}(\lambda) = \mathbb{E}_{p(\epsilon)} [\log p(\mathbf{x}, g(\epsilon, \lambda)) - \log q(g(\epsilon, \lambda)|\lambda)]$
现在，期望是关于 $p(\epsilon)$ 的，它独立于 $\lambda$。我们可以将梯度移到期望内部：
$\nabla_{\lambda} \mathcal{L}(\lambda) = \mathbb{E}_{p(\epsilon)} [\nabla_{\lambda} (\log p(\mathbf{x}, g(\epsilon, \lambda)) - \log q(g(\epsilon, \lambda)|\lambda))]$

该梯度可以使用蒙特卡洛样本近似：抽取 $S$ 个样本 $\epsilon^{(s)} \sim p(\epsilon)$，计算 $\mathbf{z}^{(s)} = g(\epsilon^{(s)}, \lambda)$，并估计：
$\nabla_{\lambda} \mathcal{L}(\lambda) \approx \frac{1}{S} \sum_{s=1}^S \nabla_{\lambda} (\log p(\mathbf{x}, \mathbf{z}^{(s)}) - \log q(\mathbf{z}^{(s)}|\lambda))$
使用链式法则，$\nabla_{\lambda} f(\mathbf{z}^{(s)})$ 变为 $(\nabla_{\mathbf{z}} f(\mathbf{z}^{(s)}))^T (\nabla_{\lambda} g(\epsilon^{(s)}, \lambda))$。这需要对数联合概率和对数变分密度关于潜在变量 $\mathbf{z}$ 的梯度，以及变换 $g$ 关于 $\lambda$ 的梯度。现代自动微分库能够轻松处理这些计算。

与评分函数方法相比，重参数化技巧通常产生方差低得多的梯度估计，通常导致更快、更稳定的收敛。然而，它并非普遍适用；它需要找到一个合适的函数 $g(\epsilon, \lambda)$。

> 这是一个基本流程图，展示了 BBVI 中的重参数化技巧。来自基础分布 $\epsilon$ 的样本通过使用变分参数 $\lambda$ 进行变换，以获得样本 $\mathbf{z}$。这些样本用于评估对数概率，它们的梯度关于 $\lambda$ 构成 ELBO 梯度估计，然后将其输入优化器以更新 $\lambda$。

### BBVI 算法

BBVI 在随机优化框架内使用这些梯度估计器，类似于 SVI。

1. **初始化**变分参数 (parameter) $\lambda$。
2. **重复**直到收敛：
   a. **采样：** 抽取 $S$ 个蒙特卡洛样本。
   \* 如果使用评分函数：对于 $s=1,...,S$，从 $q(\mathbf{z}|\lambda)$ 采样 $\mathbf{z}^{(s)}$。
   \* 如果使用重参数化：对于 $s=1,...,S$，采样 $\epsilon^{(s)} \sim p(\epsilon)$ 并计算 $\mathbf{z}^{(s)} = g(\epsilon^{(s)}, \lambda)$。
   b. **估计梯度：** 使用评分函数或重参数化公式计算 $\nabla_{\lambda} \mathcal{L}(\lambda)$ 的蒙特卡洛估计 $\hat{\mathbf{g}}$。如果执行 SVI，则使用当前数据小批量的梯度估计。
   c. **更新参数：** 使用随机优化算法（例如，Adam、RMSprop）并结合估计梯度 $\hat{\mathbf{g}}$ 更新 $\lambda$：
   $\lambda_{t+1} \leftarrow \text{OptimizerUpdate}(\lambda_t, \hat{\mathbf{g}})$

当与数据子采样（如 SVI 中所示）结合时，步骤 2.b 也包含了来自数据小批量的随机性。

### 优点与注意事项

- **通用性：** BBVI 的主要优势在于它适用于广泛的模型，包括那些具有非共轭结构或难以处理的解析梯度的模型。您只需评估模型的对数概率（以及重参数 (parameter)化所需的梯度）。
- **使用自动微分：** 现代概率编程语言 (PPL) 如 Pyro、NumPyro、TensorFlow Probability 和 Stan 高度依赖自动微分，使得 BBVI 的实现（尤其是在结合重参数化技巧时）对用户来说相对简单。PPL 处理梯度计算。
- **方差：** 梯度估计器的选择很重要。评分函数估计器可能受到高方差的影响，可能阻碍收敛。重参数化技巧在适用时，通常因其较低的方差而受到青睐。高级方差削减技术（例如，控制变量）有时可以进一步提高这两种估计器的稳定性。
- **调优：** 与其他随机优化方法类似，BBVI 可能需要仔细调整学习率和优化器参数。每次梯度步的蒙特卡洛样本数 $S$ 也会影响梯度准确性与计算成本之间的权衡。

BBVI 大幅扩展了适用于变分推断的模型的范围。通过用蒙特卡洛估计替代解析梯度推导，它提供了一个强大且灵活的工具，用于近似贝叶斯推断，尤其是在结合随机优化的可伸缩性和自动微分的便利性时。

## 参考资料

- [Black Box Variational Inference](http://proceedings.mlr.press/v33/ranganath14.pdf) — Rajesh Ranganath, Sean Gerrish, David Blei (2014)
  Journal: Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics (AISTATS); Volume: 33; Pages: 812-820
  这篇基础论文介绍了黑盒变分推断（BBVI），提供了一个通用的变分推断框架，避免了分析梯度推导的需要，转而依赖蒙特卡洛估计。它是本节核心主题的直接来源。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P. Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR) Workshop Track; DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这项开创性工作在引入变分自编码器的同时，推广了重参数化技巧（又称路径导数估计器），作为一种用于变分推断中低方差梯度估计的方法，与文中对该技术的详细说明高度相关。
- [Automatic Differentiation Variational Inference](http://www.jmlr.org/papers/volume18/16-107/16-107.pdf) — Alp Kucukelbir, Dustin Tran, Robert R. Ma, Adeline Yu, Andrew Gelman (2017)
  Journal: Journal of Machine Learning Research (JMLR); Publisher: Journal of Machine Learning Research; Volume: 18; Pages: 1-48; DOI: [10.5555/3122009.3122010](https://doi.org/10.5555/3122009.3122010)
  这篇论文详细阐述了自动微分变分推断（ADVI），它是BBVI的一种实际实现，它利用了Stan等概率编程语言中的现代自动微分工具。它展示了通用的BBVI框架如何在实践中应用。
