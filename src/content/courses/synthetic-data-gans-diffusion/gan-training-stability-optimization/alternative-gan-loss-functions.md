---
course: "synthetic-data-gans-diffusion"
chapter: "gan-training-stability-optimization"
lesson: "alternative-gan-loss-functions"
sourceId: 5674
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/alternative-gan-loss-functions"
title: "替代损失函数（WGAN, WGAN-GP, LSGAN）"
description: "实现Wasserstein GAN、梯度惩罚和最小二乘GAN目标函数。"
order: 3
plots: []
sourceHash: "32e730a0d0095a7cde94d13f3fb2793195d6b26fc90a9188509abddf45e9a793"
sourceCorrections: []
---

标准GAN的最小最大博弈，尽管优美，但训练过程常有不稳。生成器和判别器可能无法收敛，梯度可能消失，或者生成器仅能生成有限种类的输出（模式崩溃）。这些问题的一个主要原因，是原始GAN目标函数隐式最小化的詹森-香农（JS）散度的特性。当真实数据与生成数据的分布不重叠或重叠可忽略不计（这在训练初期或处理高维数据时常发生）时，JS散度会饱和，导致生成器的梯度接近于零。

为应对这些不足，研究人员提出了替代损失函数 (loss function)，这些函数能提供更稳定的梯度信号，并能更好反映真实与生成数据分布间的距离。下面我们来看三种主要的替代方案：Wasserstein GAN (WGAN)、带有梯度惩罚的WGAN (WGAN-GP) 和最小二乘GAN (LSGAN)。

### Wasserstein GAN (WGAN)

WGAN的核心思想是将JS散度替换为Wasserstein-1距离，也称为土堆距离（$W_1$）。直观地讲，如果将真实分布和生成分布想象成两堆土，$W_1$测量的是将一堆土变为另一堆土所需的最小“成本”（土的量乘以移动距离）。与JS散度不同，Wasserstein距离能提供有意义的梯度，即使分布没有显著重叠，使其成为更适合GAN训练的度量。

直接计算$W_1$是不可行的。然而，Kantorovich-Rubinstein对偶原理提供了一种计算它的方式：


$$
W_1(P_r, P_g) = \sup_{||f||_L \le 1} \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{\tilde{x} \sim P_g}[f(\tilde{x})]
$$


其中，$P_r$是真实数据分布，$P_g$是生成器的分布（$\tilde{x} = G(z)$），上确界取自所有1-Lipschitz函数$f$。如果对于所有$x_1, x_2$，函数$f$满足$|f(x_1) - f(x_2)| \le |x_1 - x_2|$，则称$f$为1-Lipschitz函数。

在WGAN框架中，判别器（现在常被称为“评论器”，用$D$或$f_w$表示）被训练来近似这个最优函数$f$。评论器输出一个标量分数（而非概率），反映输入的“真实性”。WGAN的目标函数变为：


$$
\min_G \max_{w \in \mathcal{W}} \mathbb{E}_{x \sim P_r}[D_w(x)] - \mathbb{E}_{z \sim p(z)}[D_w(G(z))]
$$


其中，$w$表示评论器的参数 (parameter)。约束$||f||_L \le 1$（Lipschitz约束）很重要。原始WGAN论文提出通过在每次梯度更新后，将评论器$w$的权重 (weight)裁剪到小的固定范围，例如$[-c, c]$，来强制执行此约束。

**评论器更新：** 最大化 $\mathbb{E}_{x \sim P_r}[D_w(x)] - \mathbb{E}_{z \sim p(z)}[D_w(G(z))]$。这会使真实样本的分数提高，假样本的分数降低。
**生成器更新：** 最小化 $-\mathbb{E}_{z \sim p(z)}[D_w(G(z))]$。这等同于最大化评论器对假样本的评分，鼓励生成器生成评论器评分更高（即认为更“真实”）的样本。

尽管带有权重裁剪的WGAN通常能带来更稳定的训练，并且与标准GAN相比有助于避免模式崩溃，但权重裁剪是强制执行Lipschitz约束的一种粗略方式。这可能导致：

1. **容量利用不足：** 将权重推向裁剪边界可能会限制评论器的建模能力。
2. **梯度爆炸或消失：** 如果裁剪参数$c$过大或过小，梯度仍然可能表现不佳。找到合适的$c$可能很困难。

### 带有梯度惩罚的WGAN (WGAN-GP)

WGAN-GP通过提出一种更直接的方式来强制执行Lipschitz约束，解决了权重 (weight)裁剪的问题：即惩罚评论器对其输入的梯度范数。一个可微函数是1-Lipschitz的，当且仅当其梯度在任何地方的范数都至多为1。WGAN-GP不是严格地强制执行此条件，而是向评论器的损失中添加一个惩罚项来促进此条件。

该惩罚项侧重于在真实分布和生成分布之间采样的点。对于一对真实样本$x$和生成样本$\tilde{x} = G(z)$，生成一个插值样本$\hat{x}$：


$$
\hat{x} = \epsilon x + (1 - \epsilon)\tilde{x}
$$


其中$\epsilon$是从$U[0, 1]$中均匀采样的。梯度惩罚项为：


$$
\lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D_w(\hat{x})||_2 - 1)^2]
$$


其中，$P_{\hat{x}}$是插值样本的分布，$||\cdot||_2$是L2范数（欧几里得范数），$\lambda$是惩罚系数（通常设为10）。该项惩罚在这些插值点上梯度范数偏离1的情况。

WGAN-GP评论器损失变为：


$$
L_{Critic} = \mathbb{E}_{\tilde{x} \sim P_g}[D_w(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D_w(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D_w(\hat{x})||_2 - 1)^2]
$$


评论器目标是*最小化*此损失（注意与WGAN最大化形式相比的符号变化，这在实际实现中很常见）。生成器损失与WGAN中相同，目标是最大化评论器对生成样本的评分：


$$
L_{Generator} = - \mathbb{E}_{\tilde{x} \sim P_g}[D_w(\tilde{x})]
$$


WGAN-GP通常能带来比原始WGAN和标准GAN更稳定的训练，常能生成更高质量的样本，无需仔细调整裁剪参数 (parameter)。它已成为GAN训练中广泛采用的基线。重要的实现细节包括从评论器中移除批量归一化 (normalization)层（因为惩罚项对批量统计数据敏感），以及如果需要归一化，则使用层归一化或其他替代方法。

### 最小二乘GAN (LSGAN)

LSGAN从不同角度处理训练不稳问题。它发现标准GAN判别器中使用的sigmoid交叉熵损失函数 (loss function)可能导致梯度消失。当判别器成功将生成样本分类为假（输出接近0的概率）时，流向生成器的梯度变得非常小，减缓了学习速度。

LSGAN用最小二乘（均方误差）目标替换了sigmoid交叉熵损失。LSGAN的目标函数为：

**判别器损失：**


$$
L_D^{\text{LSGAN}} = \frac{1}{2} \mathbb{E}_{x \sim P_r}[(D(x) - b)^2] + \frac{1}{2} \mathbb{E}_{z \sim p(z)}[(D(G(z)) - a)^2]
$$


**生成器损失：**


$$
L_G^{\text{LSGAN}} = \frac{1}{2} \mathbb{E}_{z \sim p(z)}[(D(G(z)) - c)^2]
$$


其中，$a$和$b$分别是假数据和真实数据的目标标签，$c$是生成器希望判别器对假数据输出的值。通常的选择是$a=0$，$b=1$，$c=1$。

主要思想是，最小二乘损失会惩罚那些即使被正确分类但距离决策边界（由$D(x)=c$定义）较远的样本。通过最小化$L_G$，生成器尝试通过生成使$D(\tilde{x})$接近真实数据目标标签（$b$，或在生成器目标函数中为$c$）的样本$\tilde{x}$来欺骗判别器。二次损失确保梯度不会像sigmoid交叉熵那样快速消失，从而带来更稳定的学习过程和潜在的更高质量结果。LSGAN通常比WGAN-GP更易于实现，因为它不需要梯度惩罚或权重 (weight)裁剪。

### 选择替代损失函数 (loss function)

- **WGAN-GP**通常被认为是稳定GAN训练的有力选择，特别是对于图像生成任务。它直接解决了JS散度的理论局限性，并提供了更平滑的梯度。主要缺点是梯度惩罚的计算成本。
- **LSGAN**提供了一个更简单的替代方案，它通过避免sigmoid交叉熵的饱和问题也提高了稳定性。它可能有效，且计算要求低于WGAN-GP。
- \*\*原始WGAN（带有权重 (weight)裁剪）\*\*由于裁剪的实际问题，现在使用较少，但理解其原理对于WGAN-GP有重要的参考作用。

试验通常是必要的，但WGAN-GP和LSGAN提供了强大的工具来克服GAN训练中臭名昭著的不稳问题，让您能训练更复杂的模型并获得更好的结果。本章后面的实践环节将指导您实现WGAN-GP。

## 参考资料

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  提出梯度惩罚方法以强制执行 Lipschitz 约束，解决了原始 WGAN 中权重裁剪带来的问题。
- [Least Squares Generative Adversarial Networks](https://arxiv.org/abs/1611.04076) — Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, Stephen Paul Smolley (2017)
  Journal: 2017 IEEE International Conference on Computer Vision (ICCV); DOI: [10.48550/arXiv.1611.04076](https://doi.org/10.48550/arXiv.1611.04076)
  提出将最小二乘损失函数用于判别器，解决了梯度消失问题并增强了训练稳定性。
