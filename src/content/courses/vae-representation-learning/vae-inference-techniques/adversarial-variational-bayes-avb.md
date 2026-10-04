---
course: "vae-representation-learning"
chapter: "vae-inference-techniques"
lesson: "adversarial-variational-bayes-avb"
sourceId: 6350
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-4-vae-inference-techniques/adversarial-variational-bayes-avb"
title: "对抗变分贝叶斯 (AVB)"
description: "利用对抗训练实现VAE中更灵活的变分推断。"
order: 7
plots: []
sourceHash: "6c876b3b0a57ef953755ee552dd1d4ead8c13c0ce5eaba8bf7734896e50aae16"
sourceCorrections: []
---

近似后验分布 $q_\phi(z|x)$ 的表达能力是变分自编码器 (VAE) 性能的一个主要因素。标准VAE通常对 $q_\phi(z|x)$ 采用简单、显式的分布，例如对角高斯分布，主要原因在于这种选择使得证据下界 (ELBO) 中的Kullback-Leibler (KL) 散度项 $D_{KL}(q_\phi(z|x) || p(z))$ 可以进行解析计算。然而，这种简单性可能会严重限制 $q_\phi(z|x)$ 模拟真实且常复杂的后验分布 $p_\theta(z|x)$ 的能力。对抗变分贝叶斯 (AVB) 提供了一种复杂方法，克服此限制，允许使用高灵活度的隐式近似后验分布。

### 挑战：隐式后验分布与难处理的密度函数

为了显著提升近似后验分布的灵活性，我们可以*隐式地*定义它。我们不通过显式概率密度函数指定 $q_\phi(z|x)$，而是定义一个采样过程：


$$
z = g_\phi(x, \epsilon)
$$


其中 $g_\phi$ 是一个由$\phi$参数 (parameter)化的神经网络 (neural network)（编码器），$x$ 是输入数据，$\epsilon$ 是从一个简单分布（如标准正态分布 $p(\epsilon)$）中采样的噪声变量。这种构造使得 $q_\phi(z|x)$ 能够表示几乎任何复杂的分布。

使用这种隐式 $q_\phi(z|x)$ 的挑战在于，其概率密度函数 $q_\phi(z|x)$ (以及 $\log q_\phi(z|x)$) 通常未知且难以计算。这是一个问题，因为我们旨在最大化的ELBO为：


$$
\mathcal{L}_{ELBO}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))
$$


KL散度项展开为：


$$
D_{KL}(q_\phi(z|x) || p(z)) = \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x) - \log p(z)]
$$


在无法访问 $\log q_\phi(z|x)$ 的情况下，我们无法直接计算或优化此KL散度。

### 对抗变分贝叶斯：学习对数后验

对抗变分贝叶斯，由Mescheder、Nowozin和Geiger (2017) 提出，提供了一种巧妙的方法来处理这种难处理性。核心思想是训练一个辅助神经网络 (neural network)，通常称为*评论器*或*判别器* $T_\psi(x,z)$ (由 $\psi$ 参数 (parameter)化)，以近似难处理的对数密度 $\log q_\phi(z|x)$。

#### 评论器网络 $T_\psi(x,z)$

评论器网络 $T_\psi(x,z)$ 不是典型的GAN判别器，它不尝试区分“真实”样本和“伪造”样本。相反，它被训练用于估计由隐式编码器 $g_\phi(x, \epsilon)$ 生成的样本的对数密度。这是通过优化 $T_\psi(x,z)$ 相对于以下应最小化的目标函数来实现的：


$$
\mathcal{L}_T(\psi) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \mathbb{E}_{\epsilon \sim p(\epsilon)} [\exp(T_\psi(x, g_\phi(x,\epsilon))) - T_\psi(x, g_\phi(x,\epsilon))] \right]
$$


可以证明，当此目标函数 $\mathcal{L}_T(\psi)$ 最小化时，最优评论器 $T^*_\psi(x,z)$ 满足：


$$
T^*_\psi(x,z) \approx \log q_\phi(z|x) - 1
$$


常数 $-1$ 不是问题，因为它要么抵消，要么导致ELBO中的常数偏移，这不影响相对于 $\phi$ 或 $\theta$ 的梯度（除了ELBO的熵部分，在该部分中它得到适当考虑）。

#### VAE参数的AVB目标函数

一旦我们有了评论器 $T_\psi(x,z)$ 提供估计值 $T_\psi(x,z) + 1 \approx \log q_\phi(z|x)$，我们就可以将其代入ELBO公式。VAE的编码器参数 $\phi$ (定义 $g_\phi$) 和解码器参数 $\theta$ (定义 $p_\theta(x|z)$) 的目标是最大化：


$$
\mathcal{L}_{AVB}(\theta, \phi) = \mathbb{E}_{x \sim p_{data}(x)} \left[ \mathbb{E}_{\epsilon \sim p(\epsilon)} [\log p_\theta(x|g_\phi(x,\epsilon)) + \log p(g_\phi(x,\epsilon)) - (T_\psi(x,g_\phi(x,\epsilon)) + 1)] \right]
$$


这里，$z = g_\phi(x,\epsilon)$。$p(g_\phi(x,\epsilon))$ 项是在生成的潜在样本 $z$ 处评估的先验密度。

#### 训练动态：最小最大博弈

AVB的整体训练过程变成了一个最小最大（或鞍点）优化问题：

1. **更新评论器：** 相对于评论器的参数 $\psi$ 最小化 $\mathcal{L}_T(\psi)$。此步骤旨在使 $T_\psi(x,z)$ 更好地近似 $\log q_\phi(z|x) - 1$。
2. **更新VAE：** 相对于编码器参数 $\phi$ 和解码器参数 $\theta$ 最大化 $\mathcal{L}_{AVB}(\theta, \phi)$。此步骤更新VAE，使用当前评论器对 $\log q_\phi(z|x)$ 项的估计值。

这两个步骤在训练期间通常交替进行。

以下图表展示了AVB的架构和流程：

> 对抗变分贝叶斯中的数据流和目标函数。评论器 $T_\psi(x,z)$ 学习估计隐式后验 $q_\phi(z|x)$ 的对数密度，然后将其用于VAE的目标函数中。$C$ 表示常数项。

### AVB的“对抗性”特性

澄清“对抗性”在AVB背景下的含义很重要。它与生成对抗网络 (GAN) 中的对抗性不同，GAN中的判别器试图区分真实数据和生成数据。在AVB中：

- 编码器 $g_\phi$ (定义 $q_\phi(z|x)$) 生成潜在样本 $z$。
- 评论器 $T_\psi(x,z)$ 尝试准确估计这些样本的 $\log q_\phi(z|x)$。
- 随着编码器 $g_\phi$ 的更新，分布 $q_\phi(z|x)$ 也会改变，因此评论器 $T_\psi(x,z)$ 必须适应这些变化。

“对抗性”的方面源于这种交互：VAE参数 (parameter) $(\theta, \phi)$ 被更新以最大化依赖于评论器 $T_\psi$ 的目标函数，而 $T_\psi$ 同时被更新以更好地模拟由 $\phi$ 定义的密度。这种动态类似于一场双人博弈，构成了鞍点优化的依据。

### AVB的优点

使用AVB可以带来显著改进VAE的性能：

1. **高灵活度的后验分布：** 主要益处是能够使用由强大的神经网络 (neural network) $g_\phi(x, \epsilon)$ 定义的隐式 $q_\phi(z|x)$。这使得近似后验能够捕获复杂结构，例如多峰性或非高斯形状，可能更紧密地匹配真实后验 $p_\theta(z|x)$。
2. **直接ELBO优化 (近似)：** AVB仍然优化ELBO的一个近似。通过使用更具表达力的 $q_\phi(z|x)$，AVB可以获得比使用更简单、显式后验的VAE更紧密的ELBO值，前提是评论器对 $\log q_\phi(z|x)$ 的近似是准确的。
3. **改进的表示和样本：** 更准确的后验近似通常转化为更高质量的学习到的潜在表示，因此，也带来更好的生成性能（例如，更清晰、更多样化的生成样本 $x'$）。

### 挑战与考量

虽然功能强大，但AVB也带来了一些挑战：

1. **训练稳定性：** 最小最大优化问题有时难以稳定训练。通常需要仔细调整学习率和VAE组件及评论器网络架构的超参数 (parameter) (hyperparameter)。
2. **评论器的准确性：** AVB的有效性依赖于评论器 $T_\psi(x,z)$ 提供对 $\log q_\phi(z|x)$ 的良好估计。如果评论器训练不佳或能力不足，近似可能不准确，可能阻碍VAE训练。
3. **计算成本：** 训练一个额外的神经网络 (neural network)（评论器）并执行最小最大优化会增加每次训练迭代的计算开销，与标准VAE相比。
4. **超参数敏感性：** VAE和评论器之间的关系会引入需要仔细调整的新超参数。

尽管存在这些挑战，AVB代表着朝着VAE中更强大、更灵活的变分推断迈出的重要一步。通过超越简单的显式后验分布，AVB使得VAE能够模拟更复杂的数据分布并学习更丰富的潜在表示，扩展了这些生成模型的能力范围。它是一个典型范例，说明复杂的推断技术如何带来VAE的新功能。

## 参考资料

- [Adversarial Variational Bayes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF46qd8EOugc7YzliK8H5Rzofs4xiLi6k54Y-79IChilDzPlqKG_UWe1U9tUgnoiWWgHmvw6w4kLyQkg7bAS59ouhiywqt6QrVkGCAFbpSOWeZXtGggUuUSQGgdVeGre6LPzkajqI4HP94caqqCrkY1bTw-B69JUg==) — Lars Mescheder, Sebastian Nowozin, Andreas Geiger (2017)
  Journal: Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017; Pages: 1823-1833
  这篇基础论文介绍了对抗变分贝叶斯 (AVB)，详细阐述了其理论框架和隐式后验推断方法。
- [Auto-Encoding Variational Bayes](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEnxcu3eXzL6g3P5Y-EC9f-kMXvEqV-LbPtZ_l0e8ScEogp82UX5yw9YoeBq9SEGbOtsUZf8kiMnuU8Q7zA_MkYWpydmCQalYsfWVTLPwoImLMEt4LjWKw3Q_rj41DFB3seg3vOzyPlwLxr2PIQjiFfd_hssGLQitBqnPmsz-MYALMTuX20dQ==) — Diederik P. Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这篇开创性论文介绍了变分自编码器 (VAE) 框架、证据下界 (ELBO) 和重参数化技巧，为对抗变分贝叶斯 (AVB) 等生成模型奠定了基础。
- [Implicit Variational Inference](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGycAaegs6a-Ai04zWGpKtq1rHeeNYuCL0xYFNi2C39AkO2GBf6EC_LTkq4dBUqHnk3nvi-GAiWHL90q4eGVXee3wE7tEEwjPkCo2MtP4nxV-QCuUS0PIkLV7vc3Z48nZ3p8ObYZ7siztXENril) — Michalis Titsias, Miguel Lázaro-Gredilla (2018)
  Journal: Proceedings of the 35th International Conference on Machine Learning (ICML); Publisher: PMLR; Volume: 80
  该论文提出了一个使用隐式近似后验进行变分推断的通用框架，为处理难处理的后验密度问题提供了不同于AVB的视角。
