---
course: "vae-representation-learning"
chapter: "advanced-vae-topics-extensions"
lesson: "practice-hybrid-vae-gan"
sourceId: 6387
sourceUrl: "https://apxml.com/zh/courses/vae-representation-learning/chapter-7-advanced-vae-topics-extensions/practice-hybrid-vae-gan"
title: "动手实践：实施 VAE-GAN 混合架构"
description: "实施和试验 VAE-GAN 混合模型，以提升样本质量或进行特征表征学习。"
order: 8
plots: []
sourceHash: "ba6a349606f68858c1ef81ae5e1847819bdacf761c410e9241e6575ca9c2f77d"
sourceCorrections: []
---

实践实施混合 VAE-GAN 模型。结合变分自编码器（VAEs）与生成对抗网络 (GAN)（GANs），旨在整合两者的优势：VAEs 通常具有的稳定训练和有效的潜在表征，以及 GANs 典型的清晰、高保真样本。本次练习将带领您了解构建此类模型的主要架构组件、损失函数 (loss function)和训练步骤。

### VAE-GAN 的架构蓝图

典型的 VAE-GAN 架构包含三个主要的神经网络 (neural network)组件：

1. **编码器 ($q_{\phi}(z|x)$)**: 与标准 VAE 类似，编码器将输入数据点 $x$ 映射到潜在空间中的一个分布，该分布通常由均值 $\mu$ 和对数方差 $\log\sigma^2$ 参数 (parameter)化。
2. **解码器/生成器 ($p_{\theta}(\hat{x}|z)$)**: 该网络接收一个潜在向量 (vector) $z$（可从编码器的输出或先验分布 $p(z)$ 中采样），并生成一个数据样本 $\hat{x}$。在 VAE-GAN 的场景中，这个解码器也充当了 GAN 组件的生成器。
3. **判别器 ($D_{\psi}(x)$)**: 判别器被训练用于区分真实数据样本 $x$ 与解码器生成的样本 $\hat{x}$。许多 VAE-GAN 实施的一个重要方面是，判别器也可以在定义 VAE 的重构损失中起作用。

这些组件之间的配合构成了 VAE-GAN。编码器和解码器形成 VAE 结构，而解码器和判别器形成 GAN 结构。

> VAE-GAN 架构中的数据流和损失组件。`Dis_l(x)` 指的是判别器中间层的特征。

### 损失函数 (loss function)的构建

VAE-GAN 的训练目标通常结合了几个损失项：

1. **KL 散度损失 ($L_{KL}$)**: 这是标准的 VAE 项，它促使学习到的后验分布 $q_{\phi}(z|x)$ 接近先验分布 $p(z)$（通常是标准高斯分布 $N(0, I)$）。

   
   $$
   L_{KL} = D_{KL}(q_{\phi}(z|x) || p(z))
   $$
   
2. **重构损失 ($L_{recon}$)**: VAE-GAN 通常不在像素层面使用简单的 L1 或 L2 损失来衡量 $x$ 和 $\hat{x}$ 之间的差异，而是定义在特征空间中的重构损失。具体来说，我们可以使用判别器 $D_{\psi}$ 的中间层。令 $D_{\psi,l}(x)$ 表示输入 $x$ 在判别器第 $l$ 层的激活。重构损失的目标是匹配真实数据和重构数据的这些特征表征：

   
   $$
   L_{recon} = ||D_{\psi,l}(x) - D_{\psi,l}(\hat{x})||^2_2 \quad \text{或} \quad ||D_{\psi,l}(x) - D_{\psi,l}(\hat{x})||_1
   $$
   

   这种“感知损失”通常比像素层面的损失带来视觉上更清晰的重构。VAE 编码器和解码器被优化以最小化此损失。
3. **对抗损失 ($L_{adv}$)**: 这是标准的 GAN 损失。

   - **判别器 $D_{\psi}$** 被训练以最大化其区分真实样本和生成样本的能力：
     
     $$
     L_{D} = -\mathbb{E}_{x \sim p_{data}(x)}[\log D_{\psi}(x)] - \mathbb{E}_{z \sim p(z)}[\log(1 - D_{\psi}(p_{\theta}(\hat{x}|z)))]
     $$
     
     （或 LSGAN、WGAN 等的类似形式）。样本 $p_{\theta}(\hat{x}|z)$ 可以来源于 VAE 的解码器，其中 $z$ 来自重参数 (parameter)化后的编码器输出，或者 $z$ 从先验中采样得到。
   - **解码器/生成器 $p_{\theta}$** 被训练以最小化其被判别器识别为假的能力：
     
     $$
     L_{G} = -\mathbb{E}_{z \sim p(z)}[\log D_{\psi}(p_{\theta}(\hat{x}|z))] \quad \text{或} \quad -\mathbb{E}_{x \sim p_{data}(x)}[\log D_{\psi}(p_{\theta}(\hat{x}|q_{\phi}(z|x)))]
     $$
     

VAE（编码器和解码器）的总损失通常是加权和：


$$
L_{VAE} = \lambda_{KL} L_{KL} + \lambda_{recon} L_{recon} + \lambda_{adv\_G} L_{G}
$$


判别器使用 $L_D$ 单独训练。权重 (weight) $\lambda_{KL}$、$\lambda_{recon}$ 和 $\lambda_{adv\_G}$ 是超参数 (hyperparameter)，它们平衡了每个项的影响，并且通常需要细致调优。

### 实施指导

我们来概述一下实施 VAE-GAN 的步骤和考量，假定您正在使用 PyTorch 或 TensorFlow 等框架。

#### 1. 定义网络架构

- **编码器**: 典型的卷积神经网络 (neural network)（CNN），输出潜在分布的参数 (parameter)（均值和对数方差）。
- **解码器/生成器**: 一个转置卷积神经网络，接收潜在向量 (vector)并将其上采样到输入数据的维度。类似于 DCGAN 中使用的架构可以是一个良好的起始点。
- **判别器**: 一个 CNN 分类器，接收输入图像并输出一个标量概率（或分数），指示输入是真实还是伪造。请确保您可以轻松访问中间层的激活，以便计算重构损失。

#### 2. 优化器

您通常需要独立的优化器：

- 一个用于 VAE 组件（编码器和解码器/生成器）。
- 一个用于判别器。
  Adam 是两者的常见选择。

#### 3. 训练循环

训练循环包含对 VAE 组件和判别器的轮流更新。

**对于每一批真实数据 $x$：**

**A. 更新 VAE 组件（编码器和解码器/生成器）：**

1. **前向传播 (VAE)**:
   - 将 $x$ 通过编码器以获得 $\mu, \log\sigma^2$。
   - 使用重参数化技巧采样 $z \sim q_{\phi}(z|x)$：$z = \mu + \sigma \odot \epsilon$，其中 $\epsilon \sim N(0, I)$。
   - 将 $z$ 通过解码器/生成器以获得重构数据 $\hat{x} = p_{\theta}(\hat{x}|z)$。
2. **计算 VAE 损失**:
   - $L_{KL}$: 计算 $q_{\phi}(z|x)$ 和 $p(z)$ 之间的 KL 散度。
   - $L_{recon}$: 将 $x$ 和 $\hat{x}$ 都通过（当前、固定）判别器，以获得它们的中间层特征 $D_{\psi,l}(x)$ 和 $D_{\psi,l}(\hat{x})$。计算这些特征之间的 L1 或 L2 距离。
   - $L_{G}$: 将 $\hat{x}$（和/或从 $z \sim p(z)$ 生成的样本）通过判别器。计算生成器的对抗损失，目标是使 $D_{\psi}(\hat{x})$ 看起来真实。
3. **组合与反向传播 (backpropagation)**:
   - $L_{VAE} = \lambda_{KL} L_{KL} + \lambda_{recon} L_{recon} + \lambda_{adv\_G} L_{G}$。
   - 执行反向传播并更新编码器和解码器/生成器的权重 (weight)。

**B. 更新判别器：**

1. **前向传播 (判别器)**:
   - 对于真实数据：获得 $D_{\psi}(x)$。
   - 对于伪造数据:
   - 使用编码器输出的 $z$（如上所述）生成 $\hat{x}_{enc} = p_{\theta}(\hat{x}|z)$。将 $\hat{x}_{enc}$ 从 VAE 的计算图中分离。
   - 可选地，采样 $z_{prior} \sim p(z)$ 并生成 $\hat{x}_{prior} = p_{\theta}(\hat{x}|z_{prior})$。分离 $\hat{x}_{prior}$。
   - 获得 $D_{\psi}(\hat{x}_{enc})$ 和 $D_{\psi}(\hat{x}_{prior})$。
2. **计算判别器损失 ($L_D$)**:
   - 计算判别器的对抗损失，训练其将真实 $x$ 正确分类为真实，将伪造的 $\hat{x}$（如果使用，包括 $\hat{x}_{enc}$ 和 $\hat{x}_{prior}$）分类为伪造。
3. **反向传播**:
   - 执行反向传播并更新判别器的权重。

#### 实施考量:

- **平衡考量**: VAE 和 GAN 目标之间的配合可能很敏感。权重系数（$\lambda$）非常重要。如果 GAN 组件过于强大，它可能会压制 VAE 的重构或 KL 项，导致模式崩溃或低劣的潜在表征。如果 VAE 项过于强大，样本质量可能会降低。
- **重构的特征匹配**: 确保在计算 $L_{recon}$ 时，特征 $D_{\psi,l}(x)$ 和 $D_{\psi,l}(\hat{x})$ 是从该 VAE 更新步骤中 *相同* 的固定判别器获得的。判别器本身是单独更新的。
- **训练稳定性**: 所有网络中的批量归一化 (normalization)等方法、细致的学习率选择，以及为 VAE 和判别器使用不同的学习率都可能有所帮助。一些实施会比 VAE 组件更频繁地更新判别器，或者反之。
- **初始化**: 恰当的权重初始化（例如 Xavier/Glorot 或 He）是有益的。

### VAE-GAN 的评估

一旦您的 VAE-GAN 正在训练，请考量以下评估方面：

- **样本质量**: 通过将 $z \sim p(z)$ 输入解码器来生成样本，并进行视觉检查。它们是否清晰且多样？如果适用于您的数据集（例如图像），可以使用 Fréchet Inception Distance (FID) 等定量指标。如果可用，请将其与独立 VAE 或 GAN 的样本进行比较。
- **重构质量**: 模型重构输入数据的效果如何？视觉检查 $x$ 与 $\hat{x}$。即使像素层面的 MSE 不是最小，基于特征的重构损失也应该产生感知上良好的结果。
- **潜在空间插值**: 从潜在空间中采样两个点 $z_1, z_2$（例如，通过编码两幅不同图像或从先验中采样），并在它们之间进行线性插值。解码这些插值的潜在向量 (vector)。平滑的过渡表明一个结构良好的潜在空间，这是 VAE 经常追求的特性。
- **损失曲线**: 监测所有单独的损失组件 ($L_{KL}$、$L_{recon}$、$L_G$、$L_D$)。$L_{KL}$ 理想情况下应趋于稳定。$L_G$ 和 $L_D$ 可能会波动，表明对抗博弈正在进行。$L_{recon}$ 应该减小。

### 试验与后续步骤

构建 VAE-GAN 是一个进行试验的极好平台。以下是一些建议：

- **改变损失权重 (weight)**: 系统地调整 $\lambda_{KL}$、$\lambda_{recon}$ 和 $\lambda_{adv\_G}$，以观察它们对样本质量、重构忠实度以及潜在空间结构的影响。
- **重构使用不同判别器层**: 尝试使用判别器不同的中间层来计算 $L_{recon}$。较深的层可能捕捉更抽象的语义特征，而较浅的层则侧重于纹理和局部细节。
- **替代 GAN 形式**: 尝试采用不同的 GAN 损失函数 (loss function)（例如，使用带梯度惩罚的 Wasserstein GAN 损失，而不是标准二元交叉熵），以观察它是否能改善训练稳定性或样本质量。
- **架构变化**: 调整编码器、解码器和判别器的深度和宽度。如果您的数据存在长距离依赖，可以尝试注意力机制 (attention mechanism)。
- **数据集**: 在各种数据集（例如 MNIST、CIFAR-10、CelebA）上测试您的实施，以了解其性能特性如何变化。

本次实践练习应能为构建和理解 VAE-GAN 模型打下坚实根基。此过程通常涉及迭代优化和调优，但结合 VAE 和 GAN 两者优点的潜力，使其成为研究进阶生成模型中值得付出的努力。

## 参考资料

- [Autoencoding beyond Pixels using a Learned Similarity Metric](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGTIMdLe4KpqlHmmXF7HBhtB9rg47sMPu14S25NZS8_KdAhpf9frAlrr9dwtVguXgNNulQAtV8MW1WJjw9ciO5RF9s56EhFrW_WIjySg6YS57bXvj25ao59raeOVZ1Bh2-T2krG3wth2rs=) — Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, Ole Winther (2016)
  Journal: Proceedings of the 33rd International Conference on Machine Learning (ICML 2016); Publisher: PMLR; Volume: 48; Pages: 1558–1566; DOI: [10.48550/arXiv.1512.09300](https://doi.org/10.48550/arXiv.1512.09300)
  本文介绍了VAE-GAN架构，结合了变分自编码器和生成对抗网络，并利用判别器特征进行重建损失计算。
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Diederik P. Kingma, Max Welling (2014)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)
  这是介绍变分自编码器（VAE）的开创性论文，详细阐述了其架构、证据下界（ELBO）目标以及重参数化技巧。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems (NIPS); Volume: 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇开创性论文介绍了生成对抗网络（GAN），描述了基本的对抗训练过程。
- [Improved Techniques for Training GANs](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGsBMidG2Y1Ks1EPEcMI_-tVfeaMo4QlkTa9W0BL5R4OXs4BNpOC3voAD6wMWC0fa7a7UM74qXiwA5Z4vyGfa7n1ocukSxqG8IG-bphw2mxqMLeKrmiqIw1dg==) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems; Volume: 29; Pages: 2234–2242; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  本文介绍了多种提高GAN训练稳定性和性能的实用技术，对VAE-GAN中的GAN组件很有帮助。
