---
course: "calculus-essentials-machine-learning"
chapter: "gradient-descent-algorithms"
lesson: "stochastic-gradient-descent"
sourceId: 1396
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-4-gradient-descent-algorithms/stochastic-gradient-descent"
title: "随机梯度下降 (SGD)"
description: "了解SGD如何每次只使用一个数据点更新参数。"
order: 5
plots: ["plots/1396-0.json"]
sourceHash: "d68aa9dbe6fcc9b519747e152d08a795dc83322a8cb8d63a785457fb7f224813"
sourceCorrections: []
---

机器学习 (machine learning)模型在处理非常大的数据集时，参数 (parameter)更新会产生高昂的计算成本。例如，通过对整个数据集的梯度取平均的方法（如批量梯度下降 (gradient descent)），可能因为每次更新都需要计算数百万甚至数十亿数据点的贡献而变得成本过高，甚至无法承受。随机梯度下降 (SGD) 旨在解决这一挑战，提供了一种实用且高效的替代方法，能够大幅减少每次参数更新所需的计算量。

### SGD方法：每次一个样本

SGD不使用整个数据集，而是在每次迭代中，利用从**仅一个**随机选择的训练样本 $(x^{(i)}, y^{(i)})$ 计算出的梯度来更新参数 (parameter)。

核心思想很简单：取一个样本，*只为该样本*计算成本和梯度，然后立即更新参数。接着，选择另一个随机样本并重复此过程。

从数学上看，单个参数 $\theta_j$ 的更新规则如下：

$\theta_j := \theta_j - \alpha \nabla_{\theta_j} J( \theta; x^{(i)}, y^{(i)} )$

此处，$J(\theta; x^{(i)}, y^{(i)})$ 表示为单个训练样本 $(x^{(i)}, y^{(i)})$ 计算的成本函数，而 $\nabla_{\theta_j} J(\theta; x^{(i)}, y^{(i)})$ 是该成本相对于参数 $\theta_j$ 的梯度，仅使用该单个样本进行评估。注意，这里没有批量梯度下降 (gradient descent)中出现的求和项和 $\frac{1}{m}$ 项。

### 为何称作“随机”？

“随机”一词指的是每次更新选择单个数据点所涉及的随机性。由于每次更新仅基于一个样本，因此梯度计算 $\nabla_{\theta_j} J( \theta; x^{(i)}, y^{(i)} )$ 是真实梯度 $\nabla_{\theta_j} J(\theta)$ （后者将对所有样本取平均值）的一个“有噪声”的估计。

这意味着SGD趋向最小值的路径不如批量梯度下降 (gradient descent)平滑。SGD不会直接下坡，而是呈锯齿状并波动。虽然这可能看起来效率不高，但恰恰是这种有噪声的行为提供了一些优点。



![梯度下降路径（批量与随机）](plots/1396-0.json)



> 简单的二次成本函数上的优化路径比较。批量梯度下降采用更平滑、更直接的路线，而随机梯度下降则呈现出一种有噪声、波动的趋向最小值的路径。

### SGD的优点

1. **计算效率：** 最显著的优点。每次参数 (parameter)更新都非常快，因为它只处理一个样本。这使得SGD适用于批量梯度下降 (gradient descent)不可行的大型数据集。
2. **摆脱局部最小值的能力：** 有噪声的更新有时可以帮助算法“跳出”浅层局部最小值，或比批量梯度下降更有效地通过鞍点（后者可能会陷入停滞）。
3. **在线学习：** SGD天然适合数据按顺序到达的场景（在线学习），因为模型可以随着每个新样本进行更新。

### 缺点与注意事项

1. **高方差更新：** 有噪声的梯度导致成本函数在迭代之间显著波动，而不是平稳下降。趋向最小值的路径远没有那么直接。
2. **收敛速度较慢（按轮数计）：** 尽管每次更新速度快，但由于其不规则的路径，SGD通常需要更多迭代（数据遍历，即“轮数”）才能比批量梯度下降 (gradient descent)更接近最小值。
3. **无法保证达到精确最小值：** 由于噪声的存在，SGD通常不会收敛到精确最小值，而是在其附近波动。
4. **学习率调整：** 选择合适的学习率 $\alpha$ 非常重要。通常，会使用衰减学习率（开始时较大，并随时间减小），以帮助SGD更接近最小值。
5. **数据打乱：** 在每轮（完整遍历所有训练样本）之前随机打乱训练数据集是常见做法。这可以避免算法反复看到相同顺序的样本所形成的循环，从而提高收敛性。

SGD代表一种权衡：它牺牲了批量梯度下降的平滑收敛性，以换取更快的单个更新，从而使大规模机器学习 (machine learning)成为可能。其有噪声的特性，虽然有时会带来问题，但也可能有利。然而，更新中的高方差自然地引出了一种折中方案：小批量梯度下降，我们接下来会讨论。

## 参考资料

- [A Stochastic Approximation Method](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full) — Herbert Robbins and Sutton Monro (1951)
  Journal: The Annals of Mathematical Statistics; Publisher: Institute of Mathematical Statistics; Volume: 22; Pages: 400-407; DOI: [10.1214/aoms/1177729586](https://doi.org/10.1214/aoms/1177729586)
  这篇开创性论文介绍了随机逼近方法，它构成了随机梯度下降的理论基础，为通过噪声估计进行迭代优化奠定了基础。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  第8章“深度模型训练的优化”对随机梯度下降进行了广泛的讨论，涵盖了其机制、优点和机器学习中的实践考量。
- [Large-scale machine learning with stochastic gradient descent](https://link.springer.com/chapter/10.1007/978-3-7908-2604-2_14) — Léon Bottou (2010)
  Journal: Proceedings of COMPSTAT'2010; Publisher: Physica-Verlag, Heidelberg; Pages: 177-186; DOI: [10.1007/978-3-7908-2604-2_14](https://doi.org/10.1007/978-3-7908-2604-2_14)
  这项工作为随机梯度下降处理大规模数据集的实际方面和有效性提供了有价值的见解，强调了其计算效率和收敛特性。
