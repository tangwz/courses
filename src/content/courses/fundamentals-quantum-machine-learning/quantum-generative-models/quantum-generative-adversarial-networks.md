---
course: "fundamentals-quantum-machine-learning"
chapter: "quantum-generative-models"
lesson: "quantum-generative-adversarial-networks"
sourceId: 1073
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-6-quantum-generative-models/quantum-generative-adversarial-networks"
title: "量子生成对抗网络 (QGANs)"
description: "QGANs 框架介绍，其生成器和可能的判别器使用量子电路。"
order: 3
plots: []
sourceHash: "78b027773879f95d4b750ccaca88b898d5d1f695b0da4221ff4296731ab3b0ed"
sourceCorrections: []
---

量子生成对抗网络 (GAN) (QGANs) 借鉴了经典生成对抗网络 (GANs) 的思想，在生成器和判别器这两个组件之间建立了一种竞争关系。然而，在 QGAN 框架中，一个或两个组件都可以运用量子计算。这种设置旨在比经典对应物或通过不同目标训练的 QCBMs 更有效地获取复杂数据分布。

## 量子背景下的对抗博弈

回顾经典的 GAN 设置：生成器 $G$ 试图创建能欺骗判别器 $D$ 的数据，而 $D$ 则试图区分真实数据和 $G$ 产生的伪造数据。这相当于一个极小极大博弈，其中 $G$ 最小化 $D$ 判断正确的概率，而 $D$ 则最大化该概率。

QGAN 框架采用这种对抗原理，但允许量子处理。主要有四种配置：

1. **经典生成器，经典判别器 (CC-GAN)：** 标准的经典 GAN。
2. **量子生成器，经典判别器 (QC-GAN)：** 参数 (parameter)化量子电路通过测量生成数据，然后经典神经网络 (neural network)试图将其分类为真实或伪造。由于经典判别器的相对成熟，这是一个常见的关注方向。
3. **经典生成器，量子判别器 (CQ-GAN)：** 经典生成器产生数据，而量子电路（例如，变分分类器或量子核方法）充当判别器。
4. **量子生成器，量子判别器 (QQ-GAN)：** 生成器和判别器都使用量子电路实现。这在理论上具有趣味性，但在当前硬件上面临显著的实现难题。

我们将主要侧重于 QC-GAN 架构，因为它代表了量子生成能力和已有的经典判别能力的一个实际结合点。

## 量子生成器

在 QC-GAN 中，生成器 $G(\theta)$ 通常是一个参数 (parameter)化量子电路 (PQC)。它接收一个输入，通常是随机经典噪声的向量 (vector) $z$（从高斯或均匀分布等简单分布中采样），并用它来设置 PQC 中某些门的参数，或根据 $z$ 准备一个初始状态。PQC 随后将一个固定的初始状态（例如 $|0...0\rangle$）演化为一个最终状态 $|\psi(z, \theta)\rangle$。


$$
|0...0\rangle \xrightarrow{U(z, \theta)} |\psi(z, \theta)\rangle
$$


这里，$U(z, \theta)$ 代表由 PQC 实现的幺正演化，其中包含可训练参数 $\theta$ 和输入噪声 $z$。为了生成经典判别器可以处理的经典数据样本 $x$，我们对最终状态 $|\psi(z, \theta)\rangle$ 在合适的基（通常是计算基）中进行测量。得到结果 $x$ 的概率由玻恩规则给出：


$$
p_{\theta}(x|z) = |\langle x | \psi(z, \theta) \rangle|^2
$$


生成器的目标是调整其参数 $\theta$，使得测量样本 $x$ 的分布模拟真实数据分布 $p_{\text{data}}(x)$。

## 经典判别器

判别器 $D(\phi)$ 通常是一个标准的经典神经网络 (neural network)（例如，多层感知机、CNN 或 RNN，取决于数据类型），带有可训练参数 (parameter) $\phi$。它接收数据样本 $x$ 作为输入（可以是来自数据集的真实样本，也可以是通过测量由 $G(\theta)$ 生成的伪造样本），并输出一个标量值，表示 $x$ 为真实的概率。


$$
D_{\phi}: x \mapsto [0, 1]
$$


它的目标是对于真实样本输出 $D_{\phi}(x) \approx 1$，对于由 $G(\theta)$ 生成的伪造样本输出 $D_{\phi}(x) \approx 0$。

## 训练动态与目标

训练过程遵循该方法。整体目标函数类似于经典 GAN 的目标：


$$
\min_{\theta} \max_{\phi} V(D_{\phi}, G_{\theta}) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D_{\phi}(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D_{\phi}(G_{\theta}(z)))]
$$


这里，$G_{\theta}(z)$ 表示运行带有参数 (parameter) $\theta$ 和输入 $z$ 的量子电路，测量结果以获得样本 $x_{\text{fake}}$，并将 $x_{\text{fake}}$ 输入判别器的过程。期望值 $\mathbb{E}_{z \sim p_z}$ 涉及采样噪声 $z$，运行量子生成器，测量，然后评估判别器项。

训练涉及交替更新：

1. **更新判别器：** 在保持 $\theta$ 不变的情况下，最大化 $V(D_{\phi}, G_{\theta})$ 关于 $\phi$ 的值。这通常涉及对 $\phi$ 进行梯度上升，使用当前 $G_{\theta}$ 生成的真实数据和伪造数据的批次。
   
   $$
   \nabla_{\phi} \left( \frac{1}{m} \sum_{i=1}^{m} \log D_{\phi}(x^{(i)}) + \frac{1}{m} \sum_{i=1}^{m} \log(1 - D_{\phi}(G_{\theta}(z^{(i)}))) \right)
   $$
   
2. **更新生成器：** 在保持 $\phi$ 不变的情况下，最小化 $V(D_{\phi}, G_{\theta})$ 关于 $\theta$ 的值。这通常涉及对 $\theta$ 进行梯度下降 (gradient descent)。一个常见的目标是最大化判别器的困惑度，通常通过最小化 $\mathbb{E}_{z \sim p_z}[-\log D_{\phi}(G_{\theta}(z))]$ 来实现。
   
   $$
   \nabla_{\theta} \left( \frac{1}{m} \sum_{i=1}^{m} \log(1 - D_{\phi}(G_{\theta}(z^{(i)}))) \right) \approx -\nabla_{\theta} \left( \frac{1}{m} \sum_{i=1}^{m} \log D_{\phi}(G_{\theta}(z^{(i)})) \right)
   $$
   

计算梯度 $\nabla_{\theta}$ 需要对量子电路的期望值进行微分。这通常涉及参数移位规则等技术，在应用时需要仔细考虑测量过程及其与判别器输入的关系。

> 量子-经典生成对抗网络 (GAN) (QC-GAN) 的流程图。随机噪声 `z` 作用于参数化量子电路 `PQC U(z, θ)`。测量产生 `伪造样本 x_fake`，它与 `真实样本 x_real` 一同输入经典神经网络 (neural network) `D(φ)`。判别器的 `真/假判决` 引导对判别器参数 `φ` 和生成器参数 `θ` 的更新。

## 架构与考量

生成器 $G(\theta)$ 的 PQC 架构选择很重要。浅层电路可能难以捕捉复杂关联，而非常深的电路可能受到贫瘠高原问题的影响，使得训练变得困难。通常会考虑设计用于最小化特定量子硬件上的深度和门数量的硬件高效结构。

经典判别器 $D(\phi)$ 可以是适用于数据类型的任何合适架构（例如，图像的 CNN，表格数据的 MLP）。其能力需要足以有效挑战量子生成器。

与通常使用基于似然的方法（如最小化 Kullback-Leibler 或其他散度）训练的 QCBMs 相比，QGANs 依赖对抗性损失。这有时可能带来样本质量的更快收敛，但稳定性较低，且易出现模式崩溃（即生成器只产生有限种类的样本）等问题，这与经典 GANs 类似。衡量 QGAN 性能通常涉及训练损失中的指标，例如比较生成样本与真实数据的统计特征，或在适用时使用 Frกchet Inception Distance (FID) 等指标。

## 参考资料

- [Quantum Generative Adversarial Networks for learning and loading quantum data](https://www.nature.com/articles/s41534-019-0220-9) — Christa Zoufal, Aurélien Lucchi, and Stefan Woerner (2019)
  Journal: npj Quantum Information; Publisher: Springer Nature; Volume: 5; Pages: 103; DOI: [10.1038/s41534-019-0220-9](https://doi.org/10.1038/s41534-019-0220-9)
  介绍并实验性展示了量子生成对抗网络（QGANs）的概念，包括用于学习和加载量子数据分布的QC-GAN架构。
- [Parameterized quantum circuits as machine learning models](https://iopscience.iop.org/article/10.1088/2058-9565/ab4eb5) — Marcello Benedetti, Erika Lloyd, Stefan Sack, and Mattia Fiorentini (2019)
  Journal: Quantum Science and Technology; Publisher: IOP Publishing; Volume: 4; Pages: 043007; DOI: [10.1088/2058-9565/ab4eb5](https://doi.org/10.1088/2058-9565/ab4eb5)
  全面概述了参数化量子电路（PQC）及其作为机器学习模型的应用，对于理解量子生成器有较高参考价值。
- [Evaluating gradients of parametrized quantum circuits with the parameter-shift rule](https://doi.org/10.1103/PhysRevA.101.032311) — Maria Schuld, Alexey I. Khazov, David Pérez-García, and Vedran Dunjko (2020)
  Journal: Physical Review A; Publisher: American Physical Society; Volume: 101; Pages: 032311; DOI: [10.1103/PhysRevA.101.032311](https://doi.org/10.1103/PhysRevA.101.032311)
  详细阐述了参数位移规则的基础论文，这是变分量子算法（包括QGANs）中计算梯度的重要技术。
- [Barren plateaus in quantum neural network training landscapes](https://doi.org/10.1038/s41467-018-07090-4) — Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven (2018)
  Journal: Nature Communications; Volume: 9; Pages: 4812; DOI: [10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)
  识别并解释了贫瘠高原现象，这是训练深度参数化量子电路的一个重要挑战，与生成器设计相关。
