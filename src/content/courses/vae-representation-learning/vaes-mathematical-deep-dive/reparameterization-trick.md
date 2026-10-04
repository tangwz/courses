---
course: "vae-representation-learning"
chapter: "vaes-mathematical-deep-dive"
lesson: "reparameterization-trick"
sourceId: 6289
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-2-vaes-mathematical-deep-dive/reparameterization-trick"
title: "重参数化技巧"
description: "了解重参数化技巧在变分自编码器中实现梯度优化的作用。"
order: 3
plots: []
sourceHash: "5d4eb3cdd7084d0190a2b0d7869d625b574151008c99e5b59786742f68602627"
sourceCorrections: []
---

在我们对变分自编码器的考察中，我们确认了ELBO，即 $L_{ELBO}$，作为我们的目标函数。此目标包含对近似后验分布 $q_\phi(z|x)$ 的期望：


$$
L_{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


为了使用随机梯度下降 (gradient descent)（SGD）等基于梯度的方法来优化此目标，我们需要计算相对于编码器（$\phi$）和解码器（$\theta$）参数 (parameter)的梯度。KL散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 通常可以进行解析计算（正如我们将看到的关于高斯分布的情况），并且可以求得其相对于 $\phi$ 的梯度。

然而，重构项 $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ 带来了一个难题。具体而言，我们如何通过采样过程 $z \sim q_\phi(z|x)$ 反向传播 (backpropagation)梯度以更新 $\phi$？从 $q_\phi(z|x)$ 采样的行为是一种随机操作，简单地看，它在我们的计算图中制造了一个不可导点。这就是重参数化技巧发挥作用之处。

### 问题：采样中断梯度流

设想编码器网络输出分布 $q_\phi(z|x)$ 的参数 (parameter)，例如，如果 $q_\phi(z|x)$ 是高斯分布，则输出均值 $\mu_\phi(x)$ 和标准差 $\sigma_\phi(x)$。接着，我们从这个分布中采样 $z$。解码器接收这个 $z$ 并尝试重构 $x$。

如果我们尝试反向传播 (backpropagation)重构损失，梯度需要从解码器，通过 $z$，然后到达编码器的参数 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$。采样步骤 $z \sim q_\phi(z|x)$ 本身存在问题，因为从分布中抽取随机样本的操作，相对于该分布的参数而言，不是直接可导的。梯度流的路径被实际阻断了。

考虑一个简化视图：
$x \rightarrow \text{编码器}(\phi) \rightarrow q_\phi(z|x) \text{ 的参数} \xrightarrow{\text{采样 } z} \text{解码器}(\theta) \rightarrow \log p_\theta(x|z)$。
梯度 $\nabla_\phi \log p_\theta(x|z)$ 是重构项所必需的，但采样步骤阻断了这一点。

### 解决办法：将随机性与参数 (parameter)分离

重参数化技巧通过重新构想采样过程，提供了一种巧妙的解决办法。我们不再直接从 $q_\phi(z|x)$ 采样 $z$，而是引入一个辅助噪声变量 $\epsilon$，该变量从一个简单的固定分布 $p(\epsilon)$ 中采样（例如，标准正态分布 $\mathcal{N}(0, I)$）。然后，我们将 $z$ 表示为一个确定性函数 $g_\phi(x, \epsilon)$，该函数使用编码器的参数来变换此噪声 $\epsilon$。

因此，$z = g_\phi(x, \epsilon)$，其中 $\epsilon \sim p(\epsilon)$。
重点在于 $g_\phi(x, \epsilon)$ 是一个确定性函数，其参数（源自 $\phi$）是该函数的一部分。现在，随机性被隔离到 $\epsilon$，而 $\epsilon$ 不依赖于 $\phi$。

这使我们能够改写期望：


$$
\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] = \mathbb{E}_{p(\epsilon)}[\log p_\theta(x|g_\phi(x, \epsilon))]
$$


当我们使用来自 $p(\epsilon)$ 的单个样本 $\epsilon^{(l)}$ 来进行蒙特卡洛估计时，该项变为 $\log p_\theta(x|g_\phi(x, \epsilon^{(l)}))$。现在，我们可以计算梯度 $\nabla_\phi \log p_\theta(x|g_\phi(x, \epsilon^{(l)}))$，因为在给定 $x$ 和 $\epsilon^{(l)}$ 的情况下，$g_\phi(x, \epsilon^{(l)})$ 是 $\phi$ 的一个确定性函数。梯度可以通过函数 $g$ 回流到参数 $\phi$。

### 高斯潜在变量的重参数 (parameter)化

当 $q_\phi(z|x)$ 是一个多元高斯分布时，这种技巧最常被阐明和使用。我们假设一个对角协方差结构：


$$
q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \text{diag}(\sigma^2_{\phi,1}(x), ..., \sigma^2_{\phi,J}(x)))
$$


在此，由 $\phi$ 参数化的编码器网络，为每个输入 $x$ 输出均值向量 (vector) $\mu_\phi(x)$ 和方差向量（或标准差向量）$\sigma^2_\phi(x)$。

为了进行重参数化，我们首先采样一个噪声向量 $\epsilon \sim \mathcal{N}(0, I)$，其中 $I$ 是单位矩阵，并且 $\epsilon$ 与 $z$ 具有相同的维度。然后，我们可以生成 $z$ 如下：


$$
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
$$


其中 $\sigma_\phi(x)$ 是标准差向量（方差的逐元素平方根），$\odot$ 表示逐元素乘积。

现在，$z$ 仍然是根据 $q_\phi(z|x)$ 分布的随机变量，但它被表示为 $\mu_\phi(x)$、$\sigma_\phi(x)$ 和独立噪声 $\epsilon$ 的确定性变换。损失相对于 $\mu_\phi(x)$ 和 $\sigma_\phi(x)$（进而相对于 $\phi$）的梯度可以通过该变换的标准反向传播 (backpropagation)来计算。

在实践中，编码器通常输出 $\mu_\phi(x)$ 和 $\log \sigma^2_\phi(x)$（对数方差）。这是为了数值稳定性，并确保方差 $\sigma^2_\phi(x)$ 始终为正。如果网络输出 $\log \sigma^2_\phi(x)$，那么 $\sigma_\phi(x) = \exp(0.5 \cdot \log \sigma^2_\phi(x))$。

### 梯度流图示

下面的图展示了重参数 (parameter)化技巧如何改变计算图以允许梯度流动。

> 该图展示了计算路径。在重参数化之前，$z$ 的采样节点阻断了梯度流向编码器参数 $\phi$。在重参数化之后，$z$ 从编码器输出和一个独立噪声源 $\epsilon$ 确定性地计算得到，使得梯度能够回流到 $\phi$。

### 此法为何奏效：路径导数估计器

重参数 (parameter)化技巧是文献中被称为路径导数估计器的一个示例。主要思想是将微分算子移到期望内部。我们希望计算：


$$
\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]
$$


其中 $f(z) = \log p_\theta(x|z)$。如果 $z = g_\phi(x, \epsilon)$ 且 $\epsilon \sim p(\epsilon)$，那么期望变为：


$$
\nabla_\phi \mathbb{E}_{p(\epsilon)}[f(g_\phi(x, \epsilon))]
$$


由于 $p(\epsilon)$ 不依赖于 $\phi$，并且假设 $f$ 和 $g_\phi$ 表现良好（可微），我们可以交换梯度和期望算子：


$$
\mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(x, \epsilon))]
$$


现在，梯度在期望内部。我们可以使用蒙特卡洛采样来近似此期望：从 $p(\epsilon)$ 中抽取 $L$ 个样本 $\epsilon^{(1)}, ..., \epsilon^{(L)}$，梯度估计如下：


$$
\frac{1}{L} \sum_{l=1}^{L} \nabla_\phi f(g_\phi(x, \epsilon^{(l)}))
$$


在实践中，对于VAE训练，我们通常在每个小批量的每个数据点 $x$ 使用单个 $\epsilon$ 样本（即 $L=1$）。项 $\nabla_\phi f(g_\phi(x, \epsilon^{(l)}))$ 可以使用链式法则计算：


$$
\nabla_\phi f(g_\phi(x, \epsilon^{(l)})) = \frac{\partial f}{\partial z} \Big|_{z=g_\phi(x, \epsilon^{(l)})} \frac{\partial g_\phi(x, \epsilon^{(l)})}{\partial \phi}
$$


这正是自动微分库（如TensorFlow或PyTorch中的库）在反向传播 (backpropagation)期间所做的。

### 优点与适用性

重参数 (parameter)化技巧的主要优点在于它允许使用标准反向传播 (backpropagation)算法对变分自编码器进行端到端训练。与处理随机节点的其他方法（例如分数函数估计器，也称为REINFORCE或对数导数技巧，这与本路径导数情境不同）相比，它通常会产生方差低得多的梯度估计器。方差较低的梯度通常会带来更稳定、更快的训练。

只要我们将随机变量 $z$ 表示为其参数和一个独立噪声源的确定性可微变换，重参数化技巧就适用。这对于许多连续分布很有效：

- **高斯分布**：$z = \mu + \sigma \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,1)$。
- **均匀分布**：如果 $z \sim U(a, b)$，则 $z = a + (b-a)\epsilon$，其中 $\epsilon \sim U(0,1)$。
- **指数分布**：如果 $z \sim \text{Exp}(\lambda)$，则 $z = -\frac{1}{\lambda} \log \epsilon$，其中 $\epsilon \sim U(0,1)$。
- **拉普拉斯分布**：$z = \mu - b \cdot \text{sgn}(\epsilon-0.5) \cdot \log(1 - 2|\epsilon-0.5|)$，其中 $\epsilon \sim U(0,1)$。
- 许多其他分布，包括伽马分布、贝塔分布、柯西分布，也可以进行重参数化。

然而，它并非普遍适用。对于离散潜在变量，即 $z$ 从离散集合中取值的情况，这种直接重参数化是不可能的，因为从连续的 $\epsilon$ 到离散的 $z$ 的映射会涉及不可微操作（如取整或argmax）。对于此类情况，会采用其他技术，例如Gumbel-Softmax技巧（一种连续松弛）或分数函数估计器，尽管它们伴随其自身的一些困难。

### 对ELBO优化的影响

通过应用重参数 (parameter)化技巧，ELBO中的重构项 $\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$ 变得相对于编码器参数 $\phi$ 可微。如果 $q_\phi(z|x)$ 和 $p(z)$ 被适当选择（例如，都是高斯分布），KL散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 通常是解析可微的。例如，如果 $p(z) = \mathcal{N}(0, I)$ 且 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$，则KL散度为：


$$
D_{KL}(q_\phi(z|x) || p(z)) = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_{\phi,j}(x)^2 + \sigma^2_{\phi,j}(x) - \log(\sigma^2_{\phi,j}(x)) - 1 \right)
$$


这个表达式显然相对于 $\mu_{\phi,j}(x)$ 和 $\sigma^2_{\phi,j}(x)$ 是可微的。
因此，整个ELBO可以相对于 $\phi$ 和 $\theta$ 使用梯度上升（或负ELBO的梯度下降 (gradient descent)）进行优化。

总而言之，重参数化技巧是训练变分自编码器的一种基本方法。它通过重新表述潜在变量 $z$ 的生成，巧妙地规避了通过采样过程进行微分的问题。这使得ELBO目标适用于标准基于梯度的优化，允许梯度从重构损失流回编码器网络的参数。这项创新是使变分自编码器实用且有效的重要一步。

## 参考资料

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  介绍了变分自编码器（VAEs）和重参数化技巧，使ELBO目标函数可以通过基于梯度的优化方法进行训练。
- [Stochastic Backpropagation and Variational Autoencoders](https://arxiv.org/abs/1401.4082) — Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra (2014)
  Journal: Proceedings of the 31st International Conference on Machine Learning (ICML), JMLR: W&CP; Volume: 32; DOI: [10.48550/arXiv.1401.4082](https://doi.org/10.48550/arXiv.1401.4082)
  同期发表的另一项工作，同样提出了用于训练VAEs及其他生成模型的重参数化方法。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 20
  一本内容丰富的教材，在深度学习背景下详细阐述了变分自编码器和重参数化技巧。
