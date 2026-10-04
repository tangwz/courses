---
course: "vae-representation-learning"
chapter: "advanced-vae-architectures"
lesson: "normalizing-flows-vaes"
sourceId: 6322
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-3-advanced-vae-architectures/normalizing-flows-vaes"
title: "灵活先验和后验的归一化流"
description: "将归一化流并入 VAE 中，以学习更具表达力的先验和近似后验。"
order: 5
plots: ["plots/6322-0.json"]
sourceHash: "cf2c9cae254460ed0952d21ac209bea5d29066f7f2333492c936c1474b4b402c"
sourceCorrections: []
---

标准变分自编码器对先验 $p(z)$ 和变分后验 $q_\phi(z|x)$ 都依赖相对简单的分布假设，通常是高斯分布。虽然这种选择确保了易处理性，但它会大幅限制模型的表达能力。复杂数据的真实后验 $p(z|x)$ 可能表现出多峰性或隐变量维度之间复杂的关联，远比分解高斯分布所能捕捉的要丰富。同样地，强制聚合后验 $q(z) = \int q_\phi(z|x) p_{data}(x) dx$ 去匹配一个简单、固定的先验 $p(z)$ 会限制模型学习表示的能力，并可能导致后验坍塌等问题。归一化 (normalization)流（NF）提供了一种有效且原则性的方法，用于构建更灵活、可学习的概率分布，使得 VAE 能够克服这些局限。

### 归一化 (normalization)流的要旨

其核心是，归一化流通过应用 $K$ 个可逆且可微分的变换序列 $f_1, \dots, f_K$，将一个简单的初始概率分布 $p_0(z_0)$（通常称为基分布，通常是标准高斯分布 $\mathcal{N}(0, I)$）变成一个更复杂的目标分布 $p_K(z_K)$。

设想从 $p_0(z_0)$ 抽取一个样本 $z_0$。该样本随后通过序列进行变换：
$z_1 = f_1(z_0)$
$z_2 = f_2(z_1)$
...
$z_K = f_K(z_{K-1})$

值得注意的是，我们可以精确计算最终变换变量 $z_K$ 的概率密度。这是通过概率论中的变量变换公式实现的。如果我们有一个变换 $z' = f(z)$，则 $z'$ 的密度与 $z$ 的密度之间的关系为 $p_{Z'}(z') = p_Z(f^{-1}(z')) \left| \det \left( \frac{\partial f^{-1}(z')}{\partial z'} \right) \right|$。对于一系列前向变换 $z_k = f_k(z_{k-1})$，通常更方便将最终输出 $z_K$ 的对数密度表示为初始 $z_0$ 的形式：


$$
\log p_K(z_K) = \log p_0(z_0) - \sum_{k=1}^K \log \left| \det J_{f_k}(z_{k-1}) \right|
$$


这里，$J_{f_k}(z_{k-1})$ 表示变换 $f_k$ 的雅可比矩阵（即所有一阶偏导数的矩阵），在输入 $z_{k-1}$ 处计算得到。表达式 $\left| \det J_{f_k}(z_{k-1}) \right|$ 描述了变换 $f_k$ 如何在局部拉伸或压缩空间。

为了使整个过程在计算上可行且有效，流中的每个变换 $f_k$ 必须满足三个条件：

1. 它必须是**可逆的**，这意味着我们可以通过 $f_k^{-1}$ 从 $z_k$ 恢复 $z_{k-1}$。
2. 它必须是**可微分的**，以便雅可比矩阵存在。
3. 最重要的是，其**雅可比行列式** $\det J_{f_k}$ 必须能够高效计算。这个限制严重影响了合适流层的设计。

> 一系列可逆变换 $f_1, \dots, f_K$ 将来自简单基分布 $p_0(z_0)$ 的样本 $z_0$ 映射到来自更复杂目标分布 $p_K(z_K)$ 的样本 $z_K$。这些变换的参数 (parameter)通常是学习得到的。

### 将归一化 (normalization)流融入 VAE

归一化流的灵活性可以在 VAE 框架内得到应用，以丰富变分后验 $q_\phi(z|x)$、先验 $p(z)$，甚至两者。

#### 更具表达力的变分后验

标准 VAE 通常对变分后验采用分解高斯分布，例如 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$。这是一种平均场近似，它假设给定 $x$ 时隐变量维度之间相互独立。如果真实后验 $p(z|x)$ 表现出复杂的关联或多峰性，这个假设可能会过于受限。

使用 NF，我们可以构建一个丰富得多的 $q_\phi(z|x)$：

1. 编码器网络（由 $\phi$ 参数 (parameter)化）输出一个简单基分布的参数，例如 $z_0 \sim q_{base}(z_0|x)$（例如，一个对角高斯分布，其均值和方差是 $x$ 的函数）。
2. 这个初始样本 $z_0$ 随后通过一系列 $K$ 个流变换 $f_1, \dots, f_K$。这些流层的参数也可以依赖于 $x$，或者可以是全局的、学习得到的参数。这会产生 $z_K = f_K(\dots f_1(z_0))$。
3. 得到的变分后验是 $q_\phi(z_K|x)$。其对数密度计算如下：
   
   $$
   \log q_\phi(z_K|x) = \log q_{base}(z_0|x) - \sum_{k=1}^K \log \left| \det J_{f_k}(z_{k-1}) \right|
   $$
   

这种更复杂的 $q_\phi(z_K|x)$ 随后取代了证据下界（ELBO）计算中较简单的后验。具体来说，KL 散度项 $\mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x) - \log p(z)]$ 现在包含了这种表达能力强的密度。$q_\phi(z|x)$ 更好地近似真实（通常难以处理的）后验 $p(z|x)$ 的能力可以使 ELBO 更紧密（一个更高的值，更接近真实对数似然 $\log p(x)$），从而获得更具信息量和实用价值的隐变量表示。

#### 可学习的灵活先验

在许多 VAE 实现中，隐变量的先验 $p(z)$ 是固定的，通常是标准正态分布 $\mathcal{N}(0, I)$。这种选择对隐空间的结构施加了强烈的假设。如果数据的内在流形在投影到隐空间时不能自然地符合各向同性高斯形状，模型可能难以有效学习。

归一化流提供了一种巧妙的方式来使先验 $p(z)$ 可学习且更具适应性：

1. 从一个非常简单的基分布开始，$z_0 \sim p_0(z_0)$（例如，$\mathcal{N}(0, I)$）。
2. 应用一系列 $M$ 个流变换 $g_1, \dots, g_M$，其参数 $\theta$ 是可学习的，以获得 $z_M = g_M(\dots g_1(z_0))$。
3. 这种构建方式定义了先验 $p_\theta(z_M)$，其对数密度为：
   
   $$
   \log p_\theta(z_M) = \log p_0(z_0) - \sum_{m=1}^M \log \left| \det J_{g_m}(z_{m-1}) \right|
   $$
   

这些先验变换流层 $g_m$ 的参数 $\theta$ 在训练期间与 VAE 的编码器和解码器参数一起优化。更灵活的先验允许模型找到更适合数据的隐空间几何结构。这对于缓解后验坍塌特别有帮助，后验坍塌是一种通过使 $q_\phi(z|x)$ 几乎与 $p(z)$ 相同来最小化 KL 散度项的现象，从而使隐变量失去信息。如果 $p(z)$ 本身可以适应，编码器将数据映射到有意义的隐编码可能“更简单”。

### 常见的流变换架构

NF 的实际效用取决于设计既具表达力又允许高效计算其雅可比行列式的变换层 $f_k$。几类这样的变换已被证实有效：

- **平面流**：这些应用变换 $f(z) = z + u h(w^T z + b)$，变量 $u, w \in \mathbb{R}^D$ 和 $b \in \mathbb{R}$ 是可学习参数 (parameter)，并且 $h$ 是一个平滑的逐元素非线性函数，如 $\tanh$。雅可比行列式相对简单：$\det J_f = 1 + u^T \psi(z)$，其中 $\psi(z) = h'(w^T z + b)w$。平面流直观易懂，但可能需要堆叠多层才能达到高表达力，因为每层基本上都是沿着一个超平面推拉密度。
- **径向流**：这些变换会修改围绕特定参考点 $z_{ref}$ 的密度：$f(z) = z + \beta (\alpha + ||z - z_{ref}||)^{-1} (z - z_{ref})$。参数包括 $z_{ref} \in \mathbb{R}^D$、$\alpha \in \mathbb{R}^+$ 和 $\beta \in \mathbb{R}$。径向流可以创建密度上更局部的变化。
- **耦合层（例如 RealNVP, NICE, Glow）**：这类变换特别强大且应用广泛，尤其适用于高维 $z$。其核心思想是将输入 $z$ 分为两部分（或更多部分），例如 $z_A$ 和 $z_B$。其中一部分根据另一部分进行变换，而另一部分可能保持不变或独立变换：
  $z'_A = z_A$ (第一部分的恒等变换)
  $z'_B = z_B \odot \exp(s(z_A)) + t(z_A)$ (第二部分进行缩放和平移，缩放函数 $s(\cdot)$ 和平移函数 $t(\cdot)$ 是复杂的映射，例如神经网络 (neural network)，它们*仅*依赖于 $z_A$。
  这种变换的雅可比矩阵是下三角矩阵（如果 $z'_B = z_B$ 则是上三角矩阵），这意味着其行列式简单地是其对角元素的乘积。对于上述形式，这表示为 $\prod_i \exp(s(z_A)_i) = \exp(\sum_i s(z_A)_i)$。逆变换在计算上也很高效：
  $z_A = z'_A$
  $z_B = (z'_B - t(z'_A)) \odot \exp(-s(z'_A))$
  通过堆叠许多这样的耦合层并交替变换 $z$ 的哪一部分（例如，使用置换或交换 $z_A$ 和 $z_B$ 的角色），可以建模非常复杂且表达力强的分布。
- **自回归 (autoregressive)流（例如 MAF, IAF）**：在这些流中，每个维度 $z_i$ 的变换都取决于先前的维度 $z_{<i} = (z_1, \dots, z_{i-1})$。具体而言，$z'_i = \tau(z_i; h_i(z_{<i}))$，其中 $\tau$ 是一个可逆的标量变换（例如仿射变换 $a z_i + b$），其参数 $h_i$（例如 $a$ 和 $b$）由 $z_{<i}$ 的函数生成。

  - **掩码自回归流（MAF）**：变换 $z_i$ 的参数是基于 $z_{<i}$ 生成的。这种结构使得密度评估 $\log p(z')$ 高效（可以一次完成），但样本 $z'$ 的生成是顺序的（先 $z'_1$，然后使用 $z'_1$ 生成 $z'_2$，以此类推），因此在高维情况下速度较慢。
  - **逆自回归流（IAF）**：设计为 MAF 的逆操作。样本 $z'$ 的生成可以并行进行且非常快（因为在逆变换过程中 $z_i$ 依赖于 $z'_{<i}$），但密度评估变得顺序且缓慢。
    MAF 和 IAF 都具有很强的表达力，特别是当条件函数 $h_i$ 本身由神经网络参数化时（例如，使用 MADE 架构）。



![一维归一化流：高斯到对数正态](plots/6322-0.json)



> 一个简单的一维高斯基分布（蓝色）通过函数 $z' = \exp(z)$ 变换为对数正态分布（橙色）。注意密度如何变化：变换扩展空间（大 $z$）的区域密度减小，而变换收缩空间（小 $z$）的区域密度增大，这由变换的雅可比矩阵决定。

### 归一化 (normalization)流对 VAE 性能的影响

将归一化流集成到 VAE 中可以带来显著的好处：

- **改进的密度建模和更紧密的 ELBO**：通过让 $q_\phi(z|x)$ 更好地近似真实后验，或让 $p(z)$ 更好地建模隐数据流形，NF 通常会带来更高的（更紧密的）ELBO。这表明 VAE 正在学习一个更好的数据分布模型。
- **增强的样本质量**：配备基于流的后验和/或先验的 VAE 经常生成样本（来自 $p_\theta(x) = \int p_\theta(x|z) p_\theta(z) dz$），这些样本比标准 VAE 生成的更清晰、更多样、更真实。这在处理高分辨率图像等复杂数据的方面尤其明显。
- **更丰富、更有意义的隐变量表示**：当不受过于简单的分布形式限制时，VAE 可以学习到捕捉数据中更复杂且语义上更有意义的变异因素的隐变量 $z$。如果可学习的基于流的先验允许隐空间调整其几何形状，则更是如此。
- **缓解后验坍塌**：更灵活的后验 $q_\phi(z|x)$ 不太可能变得平凡（即忽略 $x$ 并坍塌到先验 $p(z)$），因为它具有建模复杂条件依赖关系的能力。同样，灵活的先验可以适应聚合后验，减少有时导致这种坍塌的 KL 散度压力。

然而，这些优势伴随着一些权衡：

- **增加的计算需求**：归一化流中的每一层都会增加计算负担，主要原因是变换的前向传播及其雅可比行列式的计算。更深或更复杂的流架构会明显增加训练和推理 (inference)时间。
- **更高的模型复杂度与优化难题**：由于流网络中增加了参数 (parameter)，整个 VAE 模型变得更加复杂。优化这些更大的模型可能更困难，可能需要仔细的超参数 (hyperparameter)调整、复杂的优化算法或更长的训练计划。
- **流架构的选择**：改进的程度通常很大程度上取决于具体流架构的选择（例如，耦合流与自回归 (autoregressive)流，流层的数量，每层内神经网络 (neural network)的复杂度）。为给定问题选择最优的流设计并非总是简单直接的，并且是一个活跃的研究方向。

### 推进表达性分布

归一化 (normalization)流标志着 VAE 工具集中的一项重要进步，直接解决了与分布假设相关的一些基本局限。它们使得 VAE 能够学习模型中推断（后验）和生成（先验）两方面的复杂概率分布。当你为挑战性数据集设计 VAE 或追求顶尖的生成性能和表示质量时，评估 NF 增加的表达能力是否值得计算投入是一个重要考量。它们成功集成到许多前沿生成模型中，突显了它们在现代深度学习 (deep learning)中的价值。

## 参考资料

- [Variational Inference with Normalizing Flows](http://proceedings.mlr.press/v37/rezende15.pdf) — Danilo Jimenez Rezende, Shakir Mohamed (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Pages: 1530-1538
  介绍使用归一化流进行变分推断，提出平面流和径向流以构建 VAE 中更具表现力的近似后验分布。
- [Improving Variational Inference with Inverse Autoregressive Flows](https://arxiv.org/pdf/1606.04934) — Diederik P. Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 29; DOI: [10.48550/arXiv.1606.04934](https://doi.org/10.48550/arXiv.1606.04934)
  提出逆自回归流（IAF），用于创建灵活的后验分布，实现高效采样和精确对数概率计算。
- [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) — Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1605.08803](https://doi.org/10.48550/arXiv.1605.08803)
  介绍 Real NVP，一种偶合层流，可同时实现高效的密度估计和采样，适用于高维数据。
- [Masked Autoregressive Flow for Density Estimation](https://papers.nips.cc/paper/2017/hash/8b3e8103a0a30b0559a44ccf190e5138-Abstract.html) — George Papamakarios, Theo Pavlakou, Iain Murray (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; DOI: [10.48550/arXiv.1705.07057](https://doi.org/10.48550/arXiv.1705.07057)
  详细描述用于密度估计的掩码自回归流（MAF），其对数概率评估高效，但采样是顺序进行。
