---
course: "bayesian-machine-learning"
chapter: "variational-inference-techniques"
lesson: "advanced-variational-families"
sourceId: 3575
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques/advanced-variational-families"
title: "进阶变分族"
description: "审查比平均场近似更具表达能力的变分族。"
order: 7
plots: []
sourceHash: "a03a4cb4ac0bbe3d694fc7bf9359042f201cc419fa4c6000e5b014defe8e8bff"
sourceCorrections: []
---

真实后验分布中变量之间常存在复杂依赖性，这使得直接推断变得具有挑战性。平均场近似是解决这种复杂性的常用方法。该方法假设近似分布采取因子分解形式，$q(\mathbf{z}) = \prod_i q_i(z_i)$，这极大地简化了优化过程，并使得坐标上升变分推断（CAVI）等算法易于实现。然而，这种因子分解强制了近似分布 $q$ 中隐变量 $z_i$ 之间的独立性。这一假设通常是一种强烈的简化，因为真实的后验分布 $p(\mathbf{z}|\mathbf{x})$ 经常表现出变量之间强的关联性。

设想一个关于两个高度相关的变量 $z_1$ 和 $z_2$ 的后验分布，它在二维图中可能看起来像一个倾斜的椭圆。平均场近似被限制为轴对齐 (alignment)的椭圆（因为 $q(z_1, z_2) = q_1(z_1)q_2(z_2)$ 暗示着独立性），它将根本无法捕获这种关联结构。

> 平均场近似（黄色，轴对齐）无法捕获真实后验分布（蓝色，倾斜）中存在的关联性。

这种不匹配可能导致一些问题：

1. **方差低估：** 为了适应受限分布类别， $q_i(z_i)$ 的边际方差可能被迫小于真实后验的边际方差。
2. **估计偏差：** 虽然在某些情况下均值可以被较好地近似，但无法建模依赖关系会扭曲整体形状，可能导致预测或参数 (parameter)解释出现偏差。
3. **ELBO 松散：** 较不准确的 $q$ 分布会导致更大的 KL 散度 $KL(q(\mathbf{z}) || p(\mathbf{z}|\mathbf{x}))$，这意味着证据下界 $\mathcal{L}(q)$ 与真实对数模型证据 $\log p(\mathbf{x})$ 之间距离更远。

为了解决这些局限，研究人员已经开发出更具表达能力的变分族，这些变分族放宽了平均场变分推断（VI）中严格的独立性假定。

### 结构化平均场

一个相对简单的扩展是**结构化平均场**（在某些情况下也称为分块平均场或变分消息传递）。它不假定完全因子分解，而是将隐变量 $\mathbf{z}$ 划分为不相交的集合 $\mathbf{z}_1, \dots, \mathbf{z}_M$，并假定这些集合*之间*存在因子分解，但允许每个集合*内部*存在依赖关系：
$q(\mathbf{z}) = \prod_{k=1}^M q_k(\mathbf{z}_k)$
这使得近似 $q_k(\mathbf{z}_k)$ 能够捕获第 $k$ 个组内变量间的关联。分区的选择十分重要，并且通常受概率模型本身结构的指引（例如，将联合分布 $p(\mathbf{x}, \mathbf{z})$ 因子中同时出现的变量进行分组）。

虽然比标准平均场更具灵活性，但优化结构化平均场近似可能更为复杂。CAVI 更新涉及对每个块内联合分布 $q_k(\mathbf{z}_k)$ 的期望，除非仔细选择块结构，否则这些期望可能没有简单的闭式解。

### 归一化 (normalization)流

一种构建高度灵活变分分布的强大且常用方法是使用**归一化流**。其核心思路是，从一个简单的基础分布 $q_0(\mathbf{z}_0)$（例如，一个标准多元高斯分布）开始，我们可以轻松计算其密度并从中采样。然后，我们通过一系列可逆函数 $f_1, \dots, f_K$ 来变换这个简单分布：
$\mathbf{z}_0 \sim q_0(\mathbf{z}_0)$
$\mathbf{z}_1 = f_1(\mathbf{z}_0)$
$\mathbf{z}_2 = f_2(\mathbf{z}_1)$
$\dots$
$\mathbf{z}_K = f_K(\mathbf{z}_{K-1})$
最终变量 $\mathbf{z} = \mathbf{z}_K$ 服从一个可能远更为复杂的分布 $q_K(\mathbf{z})$。由于每个变换 $f_k$ 都是可逆的，我们可以从最终样本 $\mathbf{z}$ 中恢复初始噪声 $\mathbf{z}_0$：$\mathbf{z}_0 = f_1^{-1}(f_2^{-1}(\dots f_K^{-1}(\mathbf{z})))$。

特别地，如果选择的变换 $f_k$ 使得其雅可比矩阵的行列式 $\det J_{f_k}$ 能够计算，我们就可以使用概率论中的变量变换公式来计算最终分布 $q_K(\mathbf{z})$ 的密度：
$\log q_K(\mathbf{z}) = \log q_0(\mathbf{z}_0) - \sum_{k=1}^K \log |\det J_{f_k}(\mathbf{z}_{k-1})|$
这里，$\mathbf{z}_0$ 和 $\mathbf{z}_{k-1}$ 是通过从 $\mathbf{z}$ 开始逆向变换得到的。函数 $f_k$ 通常是参数 (parameter)化的（例如，使用神经网络 (neural network)），并且这些参数被优化以最大化 ELBO。

流变换的例子包括：

- **平面流：** 简单的仿射变换后跟逐元素非线性变换。
- **径向流：** 围绕参考点收缩或扩展空间的变换。
- **自回归 (autoregressive)流（例如，MAF，IAF）：** 使用自回归神经网络来定义具有三角雅可比矩阵的变换，使得行列式易于计算。
- **耦合流（例如，RealNVP，NICE，Glow）：** 将变量分成两部分，并根据一部分变换另一部分，从而得到易于可逆的函数和可处理的雅可比矩阵。

归一化流使得 $q(\mathbf{z})$ 能够近似任意复杂的后验分布，前提是流足够深且富有表达能力。它们显著提升了相比平均场近似的灵活性，通常会产生更紧密的 ELBO，并更好地捕获后验依赖关系。

### 隐式变分分布

另一种高级分布类别涉及**隐式分布**。这些分布 $q(\mathbf{z})$ 易于*采样*，但难以或无法评估其概率密度函数 $q(\mathbf{z})$ 本身。通常，它们通过采样过程定义：
$\mathbf{z} = g(\boldsymbol{\epsilon}, \phi)$
这里 $\boldsymbol{\epsilon}$ 从一个简单的噪声分布（例如，高斯分布）中抽取，而 $g$ 是一个由 $\phi$ 参数 (parameter)化的复杂、不可逆函数（如深度神经网络 (neural network)）。

隐式分布面临的挑战是 ELBO 中的熵项 $\mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})]$，它要求评估密度 $q(\mathbf{z})$。有多种技术可绕过此问题：

- **密度比估计：** 直接估计比率 $p(\mathbf{x}, \mathbf{z}) / q(\mathbf{z})$，通常使用类似于生成对抗网络 (GAN)（GANs）的对抗训练技术。
- **替代散度：** 优化标准 ELBO 中使用的 KL 散度之外的其他散度，这可能不需要明确的密度评估。

当主要目标是生成与后验分布相似的样本时，隐式分布特别有用，即便对于下游任务而言，密度评估并非严格必需。

### 自回归 (autoregressive)模型

与某些类型的归一化 (normalization)流相关，**自回归模型**使用概率的链式法则显式地对变分分布进行因子分解，而无需要求每一步都易于可逆：
$q(\mathbf{z}) = \prod_{i=1}^D q(z_i | z_{1}, \dots, z_{i-1})$
每个条件分布 $q(z_i | z_{<i})$ 可以使用灵活的函数进行建模，通常是一个神经网络 (neural network)，它将前面的变量 $z_{<i}$ 作为输入，并输出 $z_i$ 分布的参数 (parameter)（例如，如果 $q$ 是高斯分布，则为均值和方差）。例子包括应用于隐变量的 MADE（Masked Autoencoder for Distribution Estimation）或 PixelCNN/RNN 等模型。

这些模型可以捕获任意依赖关系，因为它们不通过链式法则顺序施加条件独立性假设。采样需要顺序生成，而密度评估通常通过计算每个条件项来直接进行。

### 权衡与考量

从平均场 VI 转向这些方法，引入了近似准确性和计算成本/复杂性之间的权衡：

- **准确性：** 结构化平均场、归一化 (normalization)流和自回归 (autoregressive)模型相比标准平均场，可以为复杂后验提供明显更准确的近似，从而产生更紧密的 ELBO 和更好的不确定性估计。
- **计算成本：**
  - 结构化平均场需要仔细选择块，并且更新步骤可能更为复杂。
  - 归一化流涉及计算雅可比矩阵及其行列式，这可能成本较高，特别是对于深度流或高维隐空间。其复杂性在很大程度上取决于所使用的流变换类型。
  - 隐式模型可能需要对抗训练或复杂的密度比估计。
  - 自回归模型在采样或密度评估时需要顺序计算（尽管密度评估有时可以并行化）。
- **优化：** 优化这些更复杂分布类别（例如，归一化流中变换的参数 (parameter)）的参数可能比优化简单的平均场参数更具挑战性。优化问题可能是非凸的，存在许多局部最优解。

变分族的选取是一个重要的建模决策。虽然平均场 VI 提供了一种可扩展且通常有效的基线，但在后验依赖性很强或需要更高精度后验近似以实现准确的不确定性量化 (quantization)和模型性能时，理解并运用这些更高级的族是必不可少的。模型检查和比较不同族之间 ELBO 的紧密程度可以帮助指导这一选择。

## 参考资料

- [Variational Inference: A Review for Statisticians](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1285773) — David M. Blei, Alp Kucukelbir, Jon D. McAuliffe (2017)
  Journal: Journal of the American Statistical Association; Publisher: Taylor & Francis Ltd.; Volume: 112; Pages: 859-877; DOI: [10.1080/01621459.2017.1285773](https://doi.org/10.1080/01621459.2017.1285773)
  全面概述变分推断，包括平均场近似的局限性和更高级方法的介绍。
- [Variational Inference with Normalizing Flows](https://doi.org/10.48550/arXiv.1505.05770) — Danilo Jimenez Rezende, Shakir Mohamed (2015)
  Journal: International Conference on Machine Learning (ICML); Pages: 1530-1538; DOI: [10.48550/arXiv.1505.05770](https://doi.org/10.48550/arXiv.1505.05770)
  引入了使用归一化流构建灵活变分分布的概念，并详细介绍了平面流和径向流变换。
- [Normalizing Flows for Probabilistic Modeling and Inference](https://www.jmlr.org/papers/v22/20-1361.html) — George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan (2021)
  Journal: Journal of Machine Learning Research; Publisher: JMLR, Inc.; Volume: 22; Pages: 1-64; DOI: [10.5555/3540209.3540212](https://doi.org/10.5555/3540209.3540212)
  全面回顾归一化流，涵盖其理论基础、各种架构（包括自回归流和耦合流）以及在概率建模和推断中的应用。
