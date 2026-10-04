---
course: "bayesian-machine-learning"
chapter: "revisiting-bayesian-foundations"
lesson: "bayes-theorem-high-dimensions"
sourceId: 3511
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-1-revisiting-bayesian-foundations/bayes-theorem-high-dimensions"
title: "贝叶斯定理在高维空间中"
description: "分析贝叶斯定理在高维空间中的应用及其影响。"
order: 2
plots: []
sourceHash: "37806d22be1b62fba050030bf89cb92755a9768edf2e226171ed3dfeb5401296"
sourceCorrections: []
---

正如之前所提，贝叶斯定理提供了一种系统化的方式，可以根据新证据（数据 $\mathcal{D}$）更新我们的信念（先验 $P(\theta)$），从而形成更新后的信念（后验 $P(\theta | \mathcal{D})$）：


$$
P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) P(\theta)}{P(\mathcal{D})}
$$


尽管其形式优雅，但当参数 (parameter)空间 $\theta$ 维度很高时，应用此定理会遇到很大的实际困难。这在现代机器学习 (machine learning)中是常态，模型可能拥有数千、数百万甚至数十亿个参数（例如，深度神经网络 (neural network)、复杂的图模型）。我们来审视高维度如何影响定理的每个组成部分以及由此产生的后验。

### 概率空间中的维度灾难

“维度灾难”一词指的是在分析高维空间 (high-dimensional space)数据时出现的各种现象。在贝叶斯推断中：

1. **体积集中：** 高维空间的表现与直觉不符。大多数高维超球体的体积都集中在其表面附近，而不是中心附近。同样，高维高斯分布的质量集中在一个远离均值的薄“壳”中。这表示从高维分布中随机抽取的样本不太可能接近众数。
2. **稀疏性：** 随着维度增加，可用数据变得相对稀疏。为了在高维增长时保持相同的数据点密度，所需数据量呈指数级增长。在高维参数 (parameter)空间中，我们的数据 $\mathcal{D}$ 可能只提供关于一小部分子空间的信息，使得许多参数维度未被似然很好地约束。
3. **计算复杂度：** 随着维度数量的增加，积分和优化等操作的计算要求变得很高。

### 对先验 $P(\theta)$ 的影响

在高维度中，指定一个有意义的先验 $P(\theta)$ 变得困难很多。

- **直觉失效：** 我们主要基于二维和三维的几何直觉会失效。在低维度中看似合理的先验（例如超立方体上的均匀先验）在高维度中可能会产生意想不到的后果，将大部分概率质量放置在角落或表现出异常。
- **指定困难：** 选择真正“无信息”或“客观”的先验是很有挑战性的。像宽泛高斯分布这样的标准选择，当扩展到许多维度时，仍然可能在无意中编码了强烈的假设。分层先验，其中先验参数 (parameter)（超参数 (hyperparameter)）本身被赋予了分布，对于跨维度借用信息变得更为重要，但增加了复杂性。
- **先验敏感性：** 在似然提供信息很少的方向上，后验可能对先验选择更加敏感，这在高维度中经常发生。

### 对似然 $P(\mathcal{D} | \theta)$ 的影响

似然函数 $P(\mathcal{D} | \theta)$，当数据 $\mathcal{D}$ 固定时，将其视为 $\theta$ 的函数，也会表现出不同的特点。

- **复杂的几何形态：** 高维似然曲面可能是高度多峰的，或者呈现复杂的脊状结构，使得查验变得困难。找到最大似然估计（MLE）本身就可能很有挑战性。
- **计算成本：** 对于复杂的模型（例如深度网络或大型图模型），在给定高维 $\theta$ 的情况下评估似然可能会带来高昂的计算成本，对于推断算法的每一步都是如此。

### 对后验 $P(\theta | \mathcal{D})$ 的影响

高维先验和似然的结合导致了复杂的后验。

- **几何复杂性：** 后验分布 $P(\theta | \mathcal{D})$ 继承了先验和似然的复杂性。它通常集中在参数 (parameter)空间的一小部分中，可能是在难以找到或描述的区域。它可能展现出参数之间的强相关性，使得简单的概述产生误导。
- **难以处理的归一化 (normalization)常数：** 证据 $P(\mathcal{D}) = \int P(\mathcal{D} | \theta) P(\theta) d\theta$ 是计算挑战的根本。这个积分是在整个高维参数空间 $\theta$ 上进行的。除了共轭先验-似然对（这在复杂模型中很少见）之外，这个积分在分析上是难以处理的，并且使用像求积法这样的标准数值积分技术准确计算在计算上是不可行的，其复杂度随维度呈指数增长。

考虑一个简单的可视化，说明概率质量如何在高维度中以不同方式集中。想象从 $d$ 维标准多元高斯分布 $N(0, I)$ 中进行采样。

> 概率质量集中情况的示意。在低维度（左侧）中，高斯分布的样本通常靠近众数（中心）。在高维度（右侧）中，大部分概率质量集中在远离众数的薄壳中。

- **近似需求：** 因为由于难以处理的归一化常数 $P(\mathcal{D})$，通常不可能计算出精确的后验 $P(\theta | \mathcal{D})$，所以我们必须诉诸近似方法。这是第2章（马尔可夫链蒙特卡洛）和第3章（变分推断）中所介绍的高级推断技术的主要原因。这些方法绕过了 $P(\mathcal{D})$ 的直接计算。

理解这些高维特征不仅仅是理论练习。它直接指导我们如何在实际的大规模机器学习 (machine learning)应用中构建模型、选择先验、选择推断算法以及解释结果。贝叶斯定理看似简单，但在应用于AI中常见的复杂高维问题时，它包含了很大的深度。

## 参考资料

- [Bayesian Data Analysis](https://www.taylorfrancis.com/books/9781439840955) — Andrew Gelman, John B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin (2013)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/b16018](https://doi.org/10.1201/b16018)
  贝叶斯统计学的标准参考书，涵盖理论基础和推理中的实际挑战。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; DOI: [10.1007/978-0-387-45528-0](https://doi.org/10.1007/978-0-387-45528-0)
  全面介绍了机器学习中的贝叶斯方法，讨论了高维空间和近似推理的必要性。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://doi.org/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2017)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  讨论了维度灾难及其在高维设置中对统计建模和机器学习的影响。
- [Probabilistic Machine Learning: An Introduction](https://mitpress.mit.edu/books/probabilistic-machine-learning-introduction-0) — Kevin P. Murphy (2022)
  Publisher: MIT Press; DOI: [10.7551/mitpress/12170.001.0001](https://doi.org/10.7551/mitpress/12170.001.0001)
  涵盖概率模型的基础概念，解决复杂高维模型中精确推理的挑战，并阐明近似方法的必要性。
