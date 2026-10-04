---
course: "synthetic-data-gans-diffusion"
chapter: "generative-modeling-foundations-revisited"
lesson: "diffusion-model-concepts-intro"
sourceId: 5640
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-1-generative-modeling-foundations-revisited/diffusion-model-concepts-intro"
title: "扩散模型原理介绍"
description: "扩散模型正向与逆向过程概述"
order: 5
plots: []
sourceHash: "e6c2a19d4b45a00127debdf15f5fe5547091d4bea73c9c9bd4a904e9ae78cbc2"
sourceCorrections: []
---

生成对抗网络（GAN）是一种主要的生成模型系列，其特点是采用对抗性训练机制；而扩散模型则提供了一种不同且日益强大的方法。这些模型近期才被提出，但已迅速取得了当前最好的效果，尤其在图像合成方面，常能生成高保真且非常多样化的样本。与GAN有时不稳定的最小-最大博弈不同，扩散模型一般提供更稳定的训练过程，但通常会以较慢的采样速度为代价。

扩散模型背后的核心思路涉及两个过程：一个固定的**正向过程**和一个学到的**逆向过程**。

### 正向过程：逐步添加噪声

设想有一个干净的数据样本，比如一张从真实数据分布$p_{data}(x)$中取出的图像$x_0$。正向过程通过在$T$个时间步中逐步添加少量高斯噪声来让这个样本逐渐失真。这形成了一个马尔可夫链，其中时间步$t$的状态（记作$x_t$）只依赖于前一个时间步$x_{t-1}$的状态：

$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$

这里，$\beta_t$表示一个小的正数常量，它决定了在时间步$t$添加的噪声的方差。序列$\beta_1, \beta_2, ..., \beta_T$被称为**方差调度**，一般是预先设定的（例如，线性或二次方增长）。

此过程的一个有用属性是，我们可以直接根据原始数据$x_0$来采样$x_t$，而无需迭代所有中间步骤。假如我们定义$\alpha_t = 1 - \beta_t$和$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$，则分布$q(x_t | x_0)$也是高斯分布：

$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$

随着$t$接近$T$，累积的噪声添加作用使得$\bar{\alpha}_t$趋近于零。结果，$x_T$几乎失去了关于原始$x_0$的所有信息，并且实际上与纯高斯噪声$x_T \approx \mathcal{N}(0, \mathbf{I})$变得无法区分。正向过程是固定的，它无需训练。

> 此图说明了固定的正向过程，其中数据$x_0$按照转移概率$q$在$T$个步骤中逐步添加噪声，直到它类似于纯噪声$x_T$。

### 逆向过程：学习去噪

核心生成任务在于**逆向过程**。我们希望从随机噪声$x_T \sim \mathcal{N}(0, \mathbf{I})$开始，反转加噪步骤，逐步对其进行去噪，最终得到一个看起来像是来自原始数据分布$p_{data}$的样本$x_0$。这包括学习$t = T, T-1, ..., 1$的转移概率$p_\theta(x_{t-1} | x_t)$。

假如正向过程中的噪声步长$\beta_t$足够小，那么真实的逆向转移$q(x_{t-1} | x_t, x_0)$也可以被证明近似为高斯分布。但是，计算这个真实的逆向转移需要知道原始数据$x_0$，而这正是我们想要生成的。

所以，我们使用由$\theta$参数 (parameter)化的神经网络 (neural network)来近似这些真实的逆向转移。这个网络以含噪样本$x_t$和当前时间步$t$作为输入，并学习预测分布$p_\theta(x_{t-1} | x_t)$的参数（一般是均值和方差）：

$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$

与其直接预测均值$\mu_\theta(x_t, t)$，一种常用且实用的参数化方式是训练网络（通常记作$\epsilon_\theta(x_t, t)$）来预测从$x_{t-1}$到$x_t$所添加的*噪声* $\epsilon$（或者更精确地说，是基于$x_0$对应于$x_t$的噪声分量）。有了预测的噪声$\epsilon_\theta(x_t, t)$，我们就可以计算出$p_\theta(x_{t-1} | x_t)$的参数。

### 训练与生成

逆向过程网络的参数 (parameter)$\theta$通过最大化训练数据的似然进行优化。这一般涉及优化对数似然的变分下界（ELBO）。对于扩散模型，这个目标函数往往可以被简化为更易于处理的形式。一个常见的简化目标是最小化正向过程中添加的实际噪声$\epsilon$（根据$x_0$可以方便地采样）与网络$\epsilon_\theta$预测的噪声之间的均方误差：

$L_{simple}(\theta) = \mathbb{E}_{t \sim [1, T], x_0 \sim p_{data}, \epsilon \sim \mathcal{N}(0, \mathbf{I})} \left[ || \epsilon - \epsilon_\theta(x_t, t) ||^2 \right]$

$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$是使用直接正向采样公式生成的。训练涉及重复采样数据点$x_0$、时间步$t$、采样噪声$\epsilon$、构建对应的$x_t$，并在这个损失上执行梯度下降 (gradient descent)步骤。

当模型$\epsilon_\theta(x_t, t)$训练完成后，生成（采样）过程如下：

1. 采样初始噪声$x_T \sim \mathcal{N}(0, \mathbf{I})$。
2. 从$t = T$开始，向后迭代至$t = 1$：
   - 使用网络$\epsilon_\theta(x_t, t)$预测噪声分量。
   - 使用此预测计算分布$p_\theta(x_{t-1} | x_t)$的参数（均值$\mu_\theta(x_t, t)$和方差$\Sigma_\theta(x_t, t)$）。
   - 采样$x_{t-1} \sim p_\theta(x_{t-1} | x_t)$。
3. 最终输出$x_0$是生成的样本。

> 此图比较了固定的正向加噪过程$q$与学习的逆向去噪过程$p_\theta$。逆向过程从噪声$x_T$开始，并运用一个训练过的模型$\epsilon_\theta$（通过训练目标$L_{simple}$优化），以迭代方式采样噪声较少的状态，最终生成样本$x_0$。

这种迭代式的加噪-去噪框架是扩散模型的基础。尽管直接明了，方差调度、网络结构（常是U-Net）和训练目标的不同选择会形成不同的模型变体，例如去噪扩散概率模型（DDPM）和基于分数模型，我们将在第四章详细说明这些内容。掌握这一核心机制对于实现和优化这些高级生成技术非常必要。

## 参考资料

- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](http://proceedings.mlr.press/v37/sohl-dickstein15.pdf) — Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Stefano Ermon (2015)
  Journal: Proceedings of the 32nd International Conference on Machine Learning (ICML); Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 37; Pages: 2256-2265; DOI: [10.5555/3045118.3045353](https://doi.org/10.5555/3045118.3045353)
  提出了扩散概率模型的理论框架，定义了正向和反向扩散过程。
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Jonathan Ho, Ajay Jain, Pieter Abbeel (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239)
  使扩散模型变得实用并取得了最先进的成果，引入了本节讨论的简化训练目标 ($L_{simple}$)。
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — Yang Song and Stefano Ermon (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600)
  介绍了基于分数的生成模型（噪声条件得分网络），这是一个与扩散模型并行且密切相关的框架，在文本中明确提及为一种变体。
