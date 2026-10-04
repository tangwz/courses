---
course: "applied-autoencoders-feature-extraction"
chapter: "variational-autoencoders-structured-latent-spaces"
lesson: "using-vae-latent-representations-features"
sourceId: 6487
sourceUrl: "https://apxml.com/zh/courses/applied-autoencoders-feature-extraction/chapter-6-variational-autoencoders-structured-latent-spaces/using-vae-latent-representations-features"
title: "使用VAE潜在表示作为特征"
description: "描述了如何将VAE中学到的潜在变量（均值向量）用作后续任务的特征。"
order: 8
plots: []
sourceHash: "df00e99704dd71bcf421298868b406af33bd1c132432d6722178c3fada69468e"
sourceCorrections: []
---

变分自编码器（VAEs）不仅用于重构输入；它们学习到一种到结构化潜在空间的概率映射。这一特性使得VAE的潜在表示具有价值，不仅可以生成新的数据样本，还可以作为有用的特征，用于各种后续机器学习 (machine learning)任务。这里将介绍如何提取这些特征以及它们为何特别有效。

### 从潜在分布到确定性特征

回顾一下，VAE编码器与标准自编码器的编码器不同，它不会为输入 $x$ 输出一个单一、固定的潜在向量 (vector)。相反，它提供潜在空间内某个概率分布的参数 (parameter)。通常，这个分布是高斯分布，由均值向量 $\mu(x)$ 和对数方差向量 $\log(\sigma^2(x))$ 定义。

当生成新数据时，我们从这个学习到的分布 $\mathcal{N}(\mu(x), \sigma^2(x))$ 中采样一个潜在向量 $z$。这通常使用重参数化技巧完成：$z = \mu(x) + \sigma(x) \cdot \epsilon$，其中 $\epsilon$ 是来自标准正态分布 $\mathcal{N}(0, I)$ 的一个样本。

然而，如果我们的目标是为给定输入 $x$ 提取一个一致的、确定性的特征向量，以便用于后续的分类或聚类任务，我们通常会使用**均值向量 $\mu(x)$**。均值向量 $\mu(x)$ 代表了潜在空间中对应于输入 $x$ 的分布的中心点，也就是最可能的点。它有效捕获了VAE对 $x$ 的潜在表示的最佳估计，且不包含采样过程引入的随机性。

尽管方差 $\sigma^2(x)$ 提供了有关该潜在表示的不确定性或变异性的信息，但均值 $\mu(x)$ 通常作为更稳定和直接的特征，适用于许多后续应用。

### 特征提取：过程

从已训练的VAE中提取特征是一个直接的过程：

1. **训练你的VAE：** 首先，确保你的VAE模型已得到充分训练。这表示它应在重构质量和维持良好结构化潜在空间之间取得良好平衡，这通常通过损失函数 (loss function)中合理的Kullback-Leibler (KL)散度值来体现。
2. **隔离编码器：** VAE模型由编码器和解码器组成。为了特征提取，你只需要编码器部分。
3. **执行前向传播：** 将你的输入数据 $x$ 通过已训练的编码器网络。
4. **收集均值向量 (vector)：** 编码器将输出潜在分布的参数 (parameter)，即 $\mu(x)$ 和 $\log(\sigma^2(x))$。然后你将收集 $\mu(x)$ 向量。每个 $\mu(x)$ 都是一个向量，其维度与VAE潜在空间的维度相同。这些 $\mu(x)$ 向量就是你新的、学习到的特征。

> 使用已训练VAE编码器的特征提取流程。输入数据通过编码器，其结果——潜在分布的均值向量 $\mu(x)$ ——被用作后续任务的特征表示。

### VAE派生特征的属性

从VAE潜在分布的均值中提取的特征通常具有以下几种有利属性：

- **结构化和连续的表示：** VAE损失函数 (loss function)的一个重要组成部分是KL散度项 $D_{KL}(q(z|x) || p(z))$。这一项促使学习到的潜在分布 $q(z|x)$ 近似预定义的先验分布，通常是标准正态分布 $\mathcal{N}(0, I)$。这种正则化 (regularization)效应促成一个更有序、更连续、“更平滑”的潜在空间。因此，均值向量 (vector) $\mu(x)$ 倾向于以连贯的方式捕获数据点之间的语义关系。例如，语义相似的输入在潜在空间中其 $\mu(x)$ 向量可能也相近。
- **有效的降维：** 如果所选的潜在空间维度小于原始输入空间维度，$\mu(x)$ 向量提供数据的压缩表示。这与标准自编码器类似，但增加了VAE结构规则性的优势。这种压缩可以提高计算效率，并有助于减轻后续机器学习 (machine learning)模型中的“维度灾难”。
- **提高可分离性的潜力：** 因为VAE致力于以结构化的方式组织潜在空间，派生的特征 $\mu(x)$ 可能在分类或聚类等任务中表现出更好的可分离性。这可能比使用原始数据或来自缺少这种显式概率正则化的标准自编码器所提取的特征更有利。VAE潜在空间固有的平滑性有助于定义不同数据类别之间更清晰的决策边界。

### 在后续机器学习 (machine learning)任务中的应用

一旦你为你的数据集生成了均值向量 (vector)集合 $\{\mu(x_1), \mu(x_2), ..., \mu(x_N)\}$，这些可以作为输入特征，应用于各种机器学习算法中：

- **监督学习 (supervised learning)：**
  - **分类：** $\mu(x)$ 向量可以作为输入特征，用于训练各种分类模型，例如逻辑回归、支持向量机 (SVMs)、随机森林，甚至另一个神经网络 (neural network)。与输入数据关联的原始目标标签 $y$ 用于训练这些分类器。
  - **回归：** 同样地，对于回归问题，$\mu(x)$ 向量可以作为输入特征来预测连续目标变量。
- **无监督学习 (unsupervised learning)：**
  - **聚类：** K-Means、DBSCAN 或层次聚类等算法可以直接应用于 $\mu(x)$ 向量。VAE潜在空间的结构化特性通常会产生更连贯和可解释的聚类。
  - **可视化：** 如果潜在空间维度较低（例如2D或3D），$\mu(x)$ 向量可以直接绘制。这使得可以直观检查数据流形，识别自然分组，或理解数据点之间的关系。对于更高维的潜在空间，可以将t-SNE或UMAP等技术应用于 $\mu(x)$ 向量，以生成低维嵌入 (embedding)进行可视化。

### 使用VAE特征时的考量

尽管从VAE派生的特征可以非常有效，但有几点需要注意：

- **VAE训练中的平衡：** VAE的损失函数 (loss function)需要平衡两个主要目标：准确重构输入数据，并确保学习到的潜在分布 $q(z|x)$ 接近先验分布 $p(z)$。典型的VAE损失函数是：
  
  $$
  L_{VAE} = L_{\text{重构}} + \beta \cdot D_{KL}(q(z|x) || p(z))
  $$
  
  其中，$L_{\text{重构}}$衡量解码器从潜在空间重构输入的质量，而$\beta$是一个超参数 (parameter) (hyperparameter)，用于调整KL散度项的权重 (weight)（如在$\beta$-VAEs中所示，即使$\beta=1$，这种平衡也是固有的）。如果KL散度项过于主导（例如，$\beta$设置过高），模型可能会优先匹配先验分布，牺牲重构的保真度。这可能导致一个非常规则的潜在空间，但也可能使得 $\mu(x)$ 向量 (vector)丢失一些对判别任务重要的细粒度信息。相反，如果重构损失权重过大，潜在空间可能变得不那么规则，从而削弱VAE提供的一些结构性优势。因此，仔细调整VAE架构、训练参数以及KL项的相对权重非常重要。
- **潜在空间的维度：** 潜在空间维度的选择（以及因此$\mu(x)$的维度）是一个重要的超参数。维度过小可能导致显著的信息损失，从而造成欠拟合 (underfitting)。反之，维度过大可能无法提供足够的压缩，或者使VAE学习到良好正则化 (regularization)的潜在空间变得更具挑战性。这通常需要进行实证研究和验证。
- **与其他方法的比较：** 对于生成能力或潜在空间的概率性质并非主要考虑的纯判别任务，来自标准自编码器或其他降维技术（如PCA）的特征有时可能产生可比甚至更优的性能。如果数据集相对较小，或者主要目标是直接降维而无需施加特定的概率结构，则尤其如此。然而，当您认为数据中潜在的语义连续性或结构对当前任务很重要时，VAE特征的有序性提供了一个独特的优势。

本质上，使用VAE编码器生成的均值向量 $\mu(x)$ 提供了一种精巧的方法，用于获取结构化、通常是低维的特征表示。这些特征随后可以用于多种后续机器学习 (machine learning)模型，凭借VAE学习到的潜在空间的正则化和连续特性，可能提升它们的性能。随后的实践练习将指导你构建一个VAE，这将使你能够实践在实际场景中提取和使用此类特征。

## 参考资料

- [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/pdf?id=Sy2o-kpjl) — Irina Higgins, Loïc Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Demis Hassabis (2017)
  Journal: International Conference on Learning Representations (ICLR)
  介绍了 β-VAE 框架，强调控制重建精度与潜在空间正则化之间的平衡，这对于学习解耦和结构良好的特征至关重要。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本关于深度学习的综合性教科书，深入涵盖了自编码器、变分自编码器（第20章）及其在表示学习和特征提取中的作用。
