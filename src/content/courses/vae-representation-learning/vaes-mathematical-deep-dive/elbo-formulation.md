---
course: "vae-representation-learning"
chapter: "vaes-mathematical-deep-dive"
lesson: "elbo-formulation"
sourceId: 6287
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/elbo-formulation"
title: "证据下界 (ELBO) 公式"
description: "VAE 中证据下界 (ELBO) 的详细公式和阐释。"
order: 2
plots: []
sourceHash: "885bf398c4b59650ec95215b221e47b4cc2c50566bb01d3afbb84501067c8389"
sourceCorrections: []
---

变分自编码器 (VAE) 是使用变分推断的原理推导出来的。生成建模中的一个常见目标是估计观测数据 $p(x)$ 的概率分布。对于带有潜在变量 $z$ 的模型，这需要计算边际似然：
$p(x) = \int p(x,z) dz = \int p_\theta(x|z)p(z) dz$
其中 $p(z)$ 是潜在变量的先验分布，而 $p_\theta(x|z)$ 是给定潜在变量的数据似然，通常由一个参数 (parameter)为 $\theta$ 的解码器网络参数化。

然而，对于复杂模型和高维潜在空间，此积分通常难以计算。这种难以计算性也体现在真实的后验分布 $p(z|x) = p(x|z)p(z) / p(x)$ 上，因为其分母 $p(x)$ 正是我们无法计算的积分。变分推断通过引入真实后验的近似 $q_\phi(z|x)$ 来解决此问题。此近似后验通常由一个参数为 $\phi$ 的编码器网络参数化。

核心思想是使 $q_\phi(z|x)$ 尽可能接近真实的后验 $p(z|x)$。我们使用库尔巴克-莱布勒 (KL) 散度 $D_{KL}(q_\phi(z|x) || p(z|x))$ 来衡量这种“接近程度”。我们的目标是找到使此 KL 散度最小化的参数 $\phi$。

让我们从数据对数似然 $\log p(x)$ 开始，看看 $q_\phi(z|x)$ 和证据下界 (ELBO) 如何得到。


$$
\log p(x) = \log \int p(x,z) dz
$$


我们可以在积分号内乘以并除以 $q_\phi(z|x)$（假设当 $p(x,z) > 0$ 时 $q_\phi(z|x) > 0$）：


$$
\log p(x) = \log \int q_\phi(z|x) \frac{p(x,z)}{q_\phi(z|x)} dz
$$


这可以重写为关于 $q_\phi(z|x)$ 的期望的对数：


$$
\log p(x) = \log E_{q_\phi(z|x)} \left[ \frac{p(x,z)}{q_\phi(z|x)} \right]
$$


由于对数是一个凹函数，我们可以应用琴生不等式 ($\log E[Y] \ge E[\log Y]$) 将对数移到期望内部：


$$
\log p(x) \ge E_{q_\phi(z|x)} \left[ \log \frac{p(x,z)}{q_\phi(z|x)} \right]
$$


这个下界正是**证据下界 (ELBO)**，通常表示为 $L_{ELBO}$ 或简写为 $\mathcal{L}(\phi, \theta; x)$：


$$
L_{ELBO}(\phi, \theta; x) = E_{q_\phi(z|x)} [ \log p(x,z) - \log q_\phi(z|x) ]
$$


通过展开 $p(x,z) = p_\theta(x|z)p(z)$，我们得到另一个常用形式：


$$
L_{ELBO}(\phi, \theta; x) = E_{q_\phi(z|x)} [ \log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x) ]
$$


真实对数似然 $\log p(x)$ 和 ELBO 之间的差值正是近似后验与真实后验之间的 KL 散度：


$$
\log p(x) - L_{ELBO}(\phi, \theta; x) = E_{q_\phi(z|x)} \left[ \log q_\phi(z|x) - \log p(z|x) \right] = D_{KL}(q_\phi(z|x) || p(z|x))
$$


所以，我们得到以下基本关系：


$$
\log p(x) = L_{ELBO}(\phi, \theta; x) + D_{KL}(q_\phi(z|x) || p(z|x))
$$


由于 KL 散度始终非负 ($D_{KL} \ge 0$)，ELBO 确实是数据对数似然的下界。相对于 $\phi$ 和 $\theta$ 最大化 ELBO 有以下两个作用：

1. 它使 ELBO 更接近真实对数似然，有效地最小化 KL 散度 $D_{KL}(q_\phi(z|x) || p(z|x))$，从而使我们的近似后验 $q_\phi(z|x)$ 更好地近似真实的后验 $p(z|x)$。
2. 它间接最大化了我们的模型生成观测数据的对数似然 $\log p(x)$。

> 对数边际似然 $\log p(x)$ 分解为证据下界 (ELBO) 和近似后验 $q_\phi(z|x)$ 与真实后验 $p(z|x)$ 之间的 KL 散度。最大化 ELBO 有效地最大化了 $\log p(x)$，同时最小化了近似误差。

### 剖析 ELBO

ELBO 可以重排成一个更易于理解的形式，突出 VAE 的两个主要目标：
从 $L_{ELBO} = E_{q_\phi(z|x)} [ \log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x) ]$ 开始，我们可以将项分组：


$$
L_{ELBO}(\phi, \theta; x) = E_{q_\phi(z|x)} [ \log p_\theta(x|z) ] - E_{q_\phi(z|x)} [ \log q_\phi(z|x) - \log p(z) ]
$$


第二项是 $q_\phi(z|x)$ 和 $p(z)$ 之间 KL 散度的定义：


$$
D_{KL}(q_\phi(z|x) || p(z)) = E_{q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z)} \right] = E_{q_\phi(z|x)} [ \log q_\phi(z|x) - \log p(z) ]
$$


因此，ELBO 变为：


$$
L_{ELBO}(\phi, \theta; x) = \underbrace{E_{q_\phi(z|x)} [ \log p_\theta(x|z) ]}_{\text{重构似然}} - \underbrace{D_{KL}(q_\phi(z|x) || p(z))}_{\text{KL 正则项}}
$$


让我们分别分析这两个组成部分：

1. **期望重构对数似然：** $E_{q_\phi(z|x)} [ \log p_\theta(x|z) ]$
   此项衡量解码器 $p_\theta(x|z)$ 在给定从编码器近似后验 $q_\phi(z|x)$ 中采样的潜在编码 $z$ 时，能够多好地重构输入数据 $x$。它鼓励模型学习潜在表示 $z$，这些表示保留了足够的信息以重建 $x$。这是 VAE 的“自编码”部分。$\log p_\theta(x|z)$ 的具体形式取决于数据类型：

   - 对于二值数据（例如，黑白图像），$p_\theta(x|z)$ 通常建模为伯努利分布的乘积。最大化 $\log p_\theta(x|z)$ 对应于最小化输入 $x$ 和重构输出 $\hat{x} = \text{decoder}(z)$ 之间的二元交叉熵 (BCE) 损失。
   - 对于实值数据（例如，像素强度归一化 (normalization)到 [0,1] 的图像或连续信号），$p_\theta(x|z)$ 通常建模为高斯分布 $N(x | \mu_\theta(z), \sigma^2 I)$。如果方差 $\sigma^2$ 是固定的，最大化此项等同于最小化 $x$ 和解码器平均输出 $\mu_\theta(z)$ 之间的均方误差 (MSE)。
2. **KL 散度正则项：** $D_{KL}(q_\phi(z|x) || p(z))$
   此项充当潜在空间上的正则项。它衡量近似后验分布 $q_\phi(z|x)$（由编码器针对给定输入 $x$ 生成）与潜在变量的先验分布 $p(z)$ 之间的不相似性。先验 $p(z)$ 通常选择为简单、固定的分布，最常见的是标准多元高斯分布 $N(0, I)$。
   通过最小化此 KL 散度（请注意 ELBO 公式中的负号，这意味着我们通过最大化 ELBO 项 $-D_{KL}$ 来有效最小化 $D_{KL}$），我们鼓励编码器生成潜在分布 $q_\phi(z|x)$，使其平均而言接近先验 $p(z)$。这有若干好处：

   - **平滑性和连续性：** 它有助于组织潜在空间，使其更连续，并减少出现“空洞”或不相连区域的可能性。这对 VAE 的生成能力很重要，因为我们希望能够采样 $z \sim p(z)$ 并生成新颖、连贯的数据。
   - **正则化 (regularization)：** 它阻止编码器学习过于复杂或“作弊”的后验，这些后验可能只是简单地记住 $z$ 中的输入数据。

> ELBO 包含两个主要项。第一项是期望重构对数似然，它促使模型准确重构数据。第二项是 KL 散度项，通过促使近似后验 $q_\phi(z|x)$ 接近预设先验 $p(z)$ 来正则化潜在空间。

### 实践中的 KL 散度项

先验 $p(z)$ 和近似后验 $q_\phi(z|x)$ 的常见选择都是多元高斯分布。
设 $p(z) = N(z | 0, I)$，一个均值为零、协方差矩阵为单位矩阵的标准高斯分布。
设近似后验 $q_\phi(z|x)$ 也是一个高斯分布，但其均值为 $\mu_\phi(x)$，且协方差矩阵为对角矩阵 $\text{diag}(\sigma^2_{\phi,1}(x), ..., \sigma^2_{\phi,J}(x))$，其中 $J$ 是潜在空间的维度。编码器网络将为每个输入 $x$ 输出参数 (parameter) $\mu_\phi(x)$ 和 $\log(\sigma^2_\phi(x))$（或直接是 $\sigma_\phi(x)$）。

对于这些选择，KL 散度 $D_{KL}(q_\phi(z|x) || p(z))$ 有一个便捷的解析解：


$$
D_{KL}(N(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x))) || N(0, I)) = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_{\phi,j}(x)^2 + \sigma^2_{\phi,j}(x) - \log(\sigma^2_{\phi,j}(x)) - 1 \right)
$$


这个闭式表达式可以直接纳入 VAE 的损失函数 (loss function)中，并通过梯度下降 (gradient descent)进行优化。重参数化技巧（我们将在下一节中讨论）对于通过期望 $E_{q_\phi(z|x)}$ 所涉及的采样过程反向传播 (backpropagation)梯度至关重要。

总之，ELBO 为训练 VAE 提供了可计算的目标函数。它精妙地平衡了准确数据重构的需求与正则化 (regularization)、平滑且适合生成潜在空间的需求。通过最大化 ELBO，我们同时改进我们对数据 $p(x)$ 的模型，并优化我们对真实、难以计算的后验 $p(z|x)$ 的近似 $q_\phi(z|x)$。理解此公式对于理解 VAE 如何学习以及开发更进阶的 VAE 架构和技术而言是基础性的。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  引入变分自编码器 (VAE) 并详细阐述了 ELBO 在训练中的推导和应用的基础论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  在深度学习背景下，全面解释了变分自编码器和 ELBO 的公式。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 10
  一本经典教科书，为变分推断提供了基础且严谨的处理方法，变分推断是 ELBO 和 VAE 的理论基础。
- [Probabilistic Machine Learning: Advanced Topics](https://probml.github.io/pml-book/book2.html) — Kevin Patrick Murphy (2023)
  Publisher: MIT Press; Pages: Chapter 25
  一本现代且全面的教科书，在概率机器学习基础概念的基础上，介绍了 VAE 和 ELBO 的公式。
