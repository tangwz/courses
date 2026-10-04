---
course: "vae-representation-learning"
chapter: "vae-inference-techniques"
lesson: "mean-field-approximations-limitations"
sourceId: 6337
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/mean-field-approximations-limitations"
title: "平均场近似的局限性"
description: "理解变分后验中平均场假设所施加的限制。"
order: 2
plots: ["plots/6337-0.json"]
sourceHash: "4bc1d2a4b4efe16c7410f37079ff0d01b87e7bb62dacf19f52d48de4ed9cb809"
sourceCorrections: []
---

变分自编码器（VAE）的效果通常取决于其近似后验 $q_\phi(z|x)$ 的质量。在构建这种近似后验时，一种广泛采用的简化方法是**平均场近似**。该方法假设在给定输入 $x$ 的情况下，潜在向量 (vector) $z = (z_1, z_2, \dots, z_D)$ 中的潜在变量 $z_i$ 相互独立。从数学上讲，这意味着近似后验可以分解为各个独立分布的乘积：


$$
q_\phi(z|x) = \prod_{i=1}^{D} q_\phi(z_i|x)
$$


通常，每个 $q_\phi(z_i|x)$ 被建模为一个一元高斯分布，其均值 $\mu_i(x)$ 和方差 $\sigma_i^2(x)$ 由编码器网络输出。这个假设在计算上很方便。它简化了 ELBO 中 KL 散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 的计算，尤其当先验 $p(z)$ 也是一个因子化高斯分布时（例如，标准正态分布 $\mathcal{N}(0, I)$）。KL 散度随后分解为一元高斯分布之间的 KL 散度之和，这具有闭式解。

然而，这种简化带来弊端。平均场近似的主要局限在于**真实后验 $p_\theta(z|x)$ 通常复杂得多，并且不以这种方式因子化**。数据背后的生成因素，即潜在变量 $z$ 旨在捕获的那些因素，可能内在相关。例如，在人脸图像中，姿态和光照可能是相关因素。如果 $z_1$ 表示姿态，$z_2$ 表示光照，那么 $p_\theta(z_1, z_2 | x)$ 很有可能表现出强相关性。

通过强制条件独立性，平均场 $q_\phi(z|x)$ 被限制在一类无法捕获潜在变量之间任何相关性的分布中。这带来以下几项重要影响：

1. **真实后验拟合不佳：** 如果真实后验 $p_\theta(z|x)$ 具有显著相关性（即其协方差矩阵具有非零非对角线元素），则因子化的 $q_\phi(z|x)$（如果为高斯分布，则意味着 $z$ 的协方差矩阵是对角矩阵）将是一个拙劣的近似。这种不匹配意味着我们最大化的 ELBO，可能比使用更具表达力的 $q_\phi$ 所能达到的，对真实对数似然 $\log p_\theta(x)$ 的下界更松散。
2. **方差低估与不准确的不确定性：** 平均场近似倾向于低估真实后验的方差，或者更普遍地说，它无法捕获其概率密度的形状。这通常导致近似后验在潜在空间的某些区域过于紧凑或“过度自信”，尤其当真实后验沿着不与潜在空间坐标轴对齐 (alignment)的方向延伸时。
3. **对表示质量的影响：** 如果模型被迫用独立的潜在变量表示相关的真实因素，则学习到的表示可能意义不大或更难解释。如果 VAE 的推断机制甚至无法准确建模这些相关因素的联合后验分布，它可能难以有效地解耦它们。
4. **次优生成性能：** 尽管 VAE 的训练目标是最大化 ELBO，但最终目标通常是生成高质量样本。拙劣的后验近似会间接影响解码器。如果在训练期间，编码器持续向解码器提供误导性或过于简单化的后验表示，则解码器可能无法那么有效地学习真实数据流形。
5. **对后验坍塌的助推：** 尽管并非唯一原因，但非常简单的近似后验，如平均场高斯分布，有时会使 KL 散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 更容易通过使 $q_\phi(z|x)$ 几乎与先验 $p(z)$ 相同而被最小化。在这种情况下，潜在变量携带关于输入 $x$ 的信息很少，这个问题被称为后验坍塌。更灵活的 $q_\phi(z|x)$ 可能更善于编码来自 $x$ 的信息，同时仍在合理程度上匹配先验。

为了直观地看清差异，设想一个真实的二维后验，其中 $z_1$ 和 $z_2$ 高度相关，形成一个相对于坐标轴倾斜的椭圆形分布。平均场近似会尝试用一个轴对齐的椭圆（如果方差相等，则为圆形）来拟合它。



![真实后验与平均场近似的对比](plots/6337-0.json)



> 一个对比示意图，展示了 $z_1$ 和 $z_2$ 之间具有强相关性的真实后验 $p_\theta(z|x)$（蓝色，形成倾斜的椭圆状云团）与平均场近似 $q_\phi(z|x)$（红色，形成更圆形或轴对齐的云团）。平均场近似未能捕获协方差结构。阴影区域大致表示高密度区域。

“平均场”一词本身源于物理学，其中**具有复杂关系的系统**通过假设每个组件仅与所有其他组件的平均效应发生影响，而忽略特定的成对关系来简化。在 VAE 中，这意味着在近似后验中假设潜在变量 $z_i$ 之间相互独立，忽略它们潜在的直接关系。

理解这些局限性很重要，因为它促使发展和使用更精密的推断技术。本章后续部分将介绍旨在摆脱平均场假设的方法，旨在获得更具表达力的近似后验，这些后验能更好地捕获真实后验 $p_\theta(z|x)$ 的复杂性，从而得到更紧密的 ELBO 并可能提升 VAE 性能。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍变分自编码器（VAE）的基础论文，包括最初通常采用均值场高斯近似后验的公式。
- [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) — Danilo Jimenez Rezende and Shakir Mohamed (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning; Volume: 37; Pages: 1530-1538; DOI: [10.48550/arXiv.1505.05770](https://doi.org/10.48550/arXiv.1505.05770)
  引入了归一化流以构建更具表现力的变分推断近似后验，直接解决了VAE中简单、分解（均值场）后验的限制。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 778
  对变分推断和均值场近似提供了全面的学术概述，解释了它们在概率模型中的理论基础和普遍适用性。特别是，第10章涵盖了“近似推断”，包括“变分推断”和“均值场理论”。
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — Carl Doersch (2016)
  Journal: arXiv preprint arXiv:1606.05908; DOI: [10.48550/arXiv.1606.05908](https://doi.org/10.48550/arXiv.1606.05908)
  一份被广泛引用的教程，提供了对变分自编码器易于理解且全面的解释，包括对近似后验作用和限制以及常见均值场假设的讨论。
