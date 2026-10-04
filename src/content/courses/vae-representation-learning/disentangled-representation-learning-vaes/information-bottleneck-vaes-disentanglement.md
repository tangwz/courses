---
course: "vae-representation-learning"
chapter: "disentangled-representation-learning-vaes"
lesson: "information-bottleneck-vaes-disentanglement"
sourceId: 6361
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-5-disentangled-representation-learning-vaes/information-bottleneck-vaes-disentanglement"
title: "信息瓶颈理论与用于解耦的VAEs"
description: "联系信息瓶颈理论与VAEs，着眼于解耦因素的学习。"
order: 4
plots: []
sourceHash: "46e00e21f7f2ce83114d3b412e9a079bfc03cb7a139df8b409ab50562a1174fc"
sourceCorrections: []
---

为使VAEs学习到解耦的表示，即潜在维度与独立的生成因素相对应，信息瓶颈（IB）理论提供了一个有力的指导框架。该理论最初用于信号处理和信息论，它为某些VAEs的修改，尤其是涉及KL散度项的修改，如何以及为何能够促成解耦提供了宝贵的解释。

### 信息瓶颈原理

信息瓶颈原理的核心在于处理一个基本权衡。假设你有一些输入数据$X$，并且希望为这些数据创建一个压缩表示$Z$。这个表示$Z$应该尽可能地“简单”或“紧凑”，这意味着它应该去除$X$中不相关的信息。然而，$Z$也必须保留关于$X$的足够信息，以便你仍然可以预测某个相关的目标变量$Y$（在自编码情境下，$Y$可以是$X$本身）。

信息瓶颈原理通过寻求一个表示$Z$来形式化这一点：该$Z$最小化输入$X$与表示$Z$之间的互信息$I(X;Z)$，同时最大化表示$Z$与目标$Y$之间的互信息$I(Z;Y)$。互信息$I(A;B)$衡量变量$A$包含关于变量$B$的多少信息。最小化$I(X;Z)$迫使$Z$成为$X$的压缩版本。最大化$I(Z;Y)$确保$Z$对于预测$Y$是有用的。

这种权衡通常通过拉格朗日目标函数表达：


$$
\mathcal{L}_{IB} = I(X;Z) - \lambda I(Z;Y)
$$


我们目标是最小化$\mathcal{L}_{IB}$。参数 (parameter)$\lambda > 0$（在其他场合常被记作$\beta$，注意不要与$\beta$-VAE的系数混淆）控制着平衡：较大的$\lambda$更注重$Z$预测$Y$的效果，而较小的$\lambda$则优先将$X$压缩成$Z$。

以下图表说明了这一流程：

> 数据$X$被编码为一个潜在表示$Z$，后者构成一个“瓶颈”。这个$Z$随后用于预测目标$Y$。目的是使$Z$既简洁又具有信息量。

### 连接信息瓶颈与VAEs

VAE目标函数，即证据下界（ELBO），包含两个与信息瓶颈原理高度契合的主要组成部分：


$$
\mathcal{L}_{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


1. **重构项**: $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$
   该项促使解码器$p_\theta(x|z)$在给定从近似后验$q_\phi(z|x)$中采样的潜在代码$z$的情况下，准确重构输入$x$。在目标$Y$就是输入$X$本身（自编码）的信息瓶颈情境中，此项类似于最大化$I(Z;X)$，以保证表示$Z$能为$X$提供充分信息。
2. **KL散度项**: $D_{KL}(q_\phi(z|x) || p(z))$
   该项将近似后验$q_\phi(z|x)$规整化，使其接近先验$p(z)$。正是在这里，“瓶颈”的特点变得明显。KL散度可以重写（在特定假设下并对数据分布$p_{data}(x)$进行平均）以与输入和潜在表示之间的互信息$I(X;Z)$相关联。
   具体而言，如果$p(z)$是一个简单的、可分解的先验（如$\mathcal{N}(0,I)$），则$D_{KL}(q_\phi(z|x) || p(z))$鼓励$Z$丢弃$X$中重构不需要的信息。通过使$q_\phi(z|x)$趋向于$p(z)$，VAE限制了潜在通道的“带宽”。

你会想起第3章提到，$\beta$-VAEs对ELBO进行了修改：


$$
\mathcal{L}_{\beta-VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) || p(z))
$$


当$\beta > 1$时，我们对KL散度施加更强的惩罚。从信息瓶颈的角度看，增加$\beta$等同于对“瓶颈”施加更大的压力来压缩信息，即进一步最小化$I(X;Z)$。假设是，通过强制$Z$成为$X$的一个高度压缩（但仍有用）的表示，VAE将被促使去识别最显著、潜在的变异因素，理想情况下以解耦的方式。如果这些真实的生成因素本身是独立的，那么一个捕捉了它们的高度压缩表示自然会试图使其自身的维度独立，以匹配$p(z)$的结构。

### 信息瓶颈与解耦的寻求

为什么这种压缩会促成解耦？直觉是，如果数据的真实生成因素相对独立，并且能够解释数据的不同方面，那么在潜在空间$Z$中表示数据最有效（即最压缩）的方式就是让每个潜在维度$z_j$对应其中一个因素。

- **效率压力**：信息瓶颈原理，通过$\beta$-VAEs中的$\beta$项放大，强制模型明智地选择其潜在编码。如果一个更简单、可分解的表示（由$p(z)$鼓励）足以进行重构，模型就无法承担编码冗余信息或潜在变量之间复杂依赖的开销。
- **可分解先验**：可分解先验的标准选择，$p(z) = \prod_j p(z_j)$（例如，各向同性高斯分布），起着重要作用。通过惩罚$q_\phi(z|x)$偏离此可分解先验的行为，VAE被促使学习一个同样可分解的聚合后验$q(z) = \mathbb{E}_{p_{data}(x)}[q_\phi(z|x)]$。$q(z)$中的可分解性意味着潜在维度之间的统计独立性，这是解耦的一个标志。
- **充分统计量**：信息瓶颈框架促使$Z$成为任务（重构）的最小充分统计量。如果潜在因素是描述数据变异的“真实”最小充分统计量，那么强大的瓶颈压力应该引导VAE去找到它们。

### 实际影响与思考

信息瓶颈理论为$\beta$-VAE等方法提供了有力的理论依据。它解释了为什么增加$\beta$可以使表示在解耦指标上获得更好的分数。模型被迫优先保留哪些信息，如果数据的潜在结构由相对独立的因素组成，那么从信息成本角度看，这些是“最便宜”的保留内容。

然而，有一些实际点需要考虑：

- **$\beta$权衡**：正如$\beta$-VAEs所示，存在直接的权衡。较高的$\beta$值通常能提高解耦分数，但可能导致重构质量下降。模型为了获得更压缩和可分解的潜在空间，可能会舍弃完美重构所需的细节。这正是信息瓶颈权衡的体现。
- **对数据的隐含假设**：信息瓶颈启发的解耦方法能够成功，依赖于数据*确实*拥有潜在的、相对独立的生成因素，且这些因素也是重构时最具信息量的组成部分。如果真实因素高度纠缠，仅凭信息瓶颈可能不足以实现解耦。
- **VAEs中的近似**：VAEs使用摊销变分近似$q_\phi(z|x)$并优化对数似然的下界。这些近似意味着真正的互信息项$I(X;Z)$和$I(Z;X)$并未被直接或完美优化。ELBO中的KL散度是控制容量$I(X;Z)$的替代手段。
- **无保证**：尽管信息瓶颈提供了令人信服的理由，但它不提供严格的解耦保证。学习到的表示仍然高度依赖于数据集、模型架构、$\beta$的选择以及其他超参数 (parameter) (hyperparameter)。“信息性”（即良好的重构）的定义可能并不总是与人类可解释的解耦完美契合。

总之，信息瓶颈理论提供了一个有益的视角，有助于理解VAEs中促成解耦的机制。它解释了为什么通过KL散度项（通常乘以$\beta$等因子）来规整潜在空间的容量，可以促使模型学习到每个维度捕获数据中独立变异因素的表示。尽管并非万能药，但这一观点为许多成功的解耦方法的设计与解释提供了指导。

## 参考资料

- [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057) — Naftali Tishby, Fernando C. Pereira, William Bialek (1999)
  Journal: Advances in Neural Information Processing Systems 12 (NIPS 1999); Volume: 12; DOI: [10.48550/arXiv.physics/0004057](https://doi.org/10.48550/arXiv.physics/0004057)
  引入信息瓶颈原理的开创性论文，为信息压缩和相关信息提取提供了理论框架。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2013)
  Journal: International Conference on Learning Representations (ICLR 2014); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAE）框架，是生成模型和表示学习的基础，也是本节讨论方法的基础。
- [β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=HyzQzM-Cg) — Irina Higgins, Loïc Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner (2017)
  Journal: International Conference on Learning Representations (ICLR 2017)
  提出了β-VAE模型，通过调整KL散度项来促进解缠结，直接与信息瓶颈原理相关。
- [Deep Variational Information Bottleneck](https://arxiv.org/abs/1612.00410) — Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, Kevin Murphy (2017)
  Journal: Proceedings of the International Conference on Learning Representations (ICLR) 2017; DOI: [10.48550/arXiv.1612.00410](https://doi.org/10.48550/arXiv.1612.00410)
  将信息瓶颈原理与深度学习相结合，提出了一种与VAE并行的变分近似方法，并明确了神经网络的IB目标。
