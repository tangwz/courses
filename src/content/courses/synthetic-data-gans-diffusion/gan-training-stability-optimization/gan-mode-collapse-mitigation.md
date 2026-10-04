---
course: "synthetic-data-gans-diffusion"
chapter: "gan-training-stability-optimization"
lesson: "gan-mode-collapse-mitigation"
sourceId: 5671
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/gan-mode-collapse-mitigation"
title: "模式坍塌：原因与缓解策略"
description: "了解模式坍塌和批次鉴别等技术。"
order: 2
plots: ["plots/5671-0.json"]
sourceHash: "b18f97439623b6d310e743d230aa74699d46b2140306e620519fdb1b0e1a0d61"
sourceCorrections: []
---

训练生成对抗网络 (GAN)时，最常遇到也最令人沮丧的问题之一是**模式坍塌**。它出现于，当生成器$G$学会只生成能够欺骗判别器$D$的可能输出中非常有限的一部分时。生成器没有捕获真实数据分布$p_{data}$的完整多样性和复杂性，而是收敛到只生成数据的一个或几个“模式”，完全忽略了其他模式。设想在一个手写数字（0-9）数据集上训练GAN；严重的模式坍塌可能导致生成器*只*生成令人信服的数字“1”的图像，无论输入噪声向量 (vector)$z$是什么。

## 了解模式坍塌的原因

模式坍塌与GAN目标函数描述的最小-最大博弈的动态变化紧密相关：


$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$


考虑训练过程：

1. 判别器$D$试图最大化其区分真实样本（$x$）和生成样本（$G(z)$）的能力。
2. 生成器$G$试图最小化判别器的成功，其本质是尝试生成$D$会分类为真实（即使$D(G(z))$接近1）的$G(z)$。

当判别器相对于生成器变得过于熟练，或者当生成器找到欺骗当前判别器的“捷径”时，模式坍塌常会发生。如果$G$发现一个或一小组输出，当前$D$始终将其错误分类为真实，则优化过程会提供强大的梯度，促使$G$ *只*生成这些特定输出。如果生成器已经可以通过坚持这几个模式来有效最小化损失，那么它就没有太多动力去学习数据分布的其他部分。

这可能导致一个循环：$D$最终学会检测$G$生成的有限模式，更新其参数 (parameter)，然后$G$又找到另一组有限的输出来欺骗更新后的$D$。生成器可能在几个模式之间跳跃，而从未学习到完整的数据分布。当判别器变得过于准确时，标准GAN损失的梯度可能会消失或表现不佳，这会进一步阻碍生成器学习完整分布的能力。

这是目标多模态 (multimodal)分布与遭受模式坍塌的生成器进行比较的视觉呈现：



![模式坍塌可视化](plots/5671-0.json)



> 蓝色圆圈代表目标数据分布中不同的模式。红色叉号表示生成器的输出，它们紧密聚集在目标模式之一周围，这表明发生了模式坍塌。

## 缓解策略

已经开发出几种技术来对抗模式坍塌，并鼓励生成器产生更多样化的样本集。

### 小批量判别

与判别器独立评估每个样本不同，小批量判别允许判别器查看同一小批量中样本*之间*的关系。其核心思想是为小批量中的每个样本计算一个特征向量 (vector)，然后衡量这些特征向量在整个批次中的相似性。

如果生成器正在生成非常相似的样本（即坍塌），判别器可以检测到批次内这种多样性不足的情况，并惩罚生成器。这鼓励$G$生成彼此不同的样本批次。

实现上通常涉及在判别器中添加一个特定层。该层计算小批量中样本之间距离的汇总统计量，并在最终分类层之前将此信息连接到中间特征。令$f(x_i) \in \mathbb{R}^A$为判别器$D$中样本$x_i$的中间层输出。小批量判别从$f(x_i)$计算一个矩阵$M \in \mathbb{R}^{B \times C}$，计算批次中所有样本$i, j$的$M_i$和$M_j$行之间的L1距离，应用负指数$c_{b}(x_i, x_j) = \exp(-||M_{i,b} - M_{j,b}||_{L1})$，将每个样本$i$的这些距离在其他样本$j$上求和：$o(x_i)_b = \sum_{j=1}^n c_b(x_i, x_j)$，并将结果向量$o(x_i) = [o(x_i)_1, o(x_i)_2, ..., o(x_i)_B]$连接到下一层的输入特征$f(x_i)$。这为判别器提供了关于批次多样性的明确信息。

### 展开式GAN

这种技术修改了生成器的更新步骤。生成器不再仅仅根据$D$的*当前*状态来优化$G$，而是通过模拟判别器的多次更新来“向前看”。$G$优化其参数 (parameter)，不仅是为了欺骗当前的$D$，也是为了欺骗未来的$D$版本。

通过展开优化，$G$被阻止进行那些可能有效欺骗当前$D$但$D$经过几次更新后很容易被检测到的更新。这种预见性有助于防止生成器坍塌到只在短期内有效的模式。尽管这种方法效果好，但展开操作会大幅增加计算成本，因为它在一个生成器更新步骤中需要多次判别器更新。

### 替代损失函数 (loss function)

正如下一节讨论的，某些损失函数本质上比原始的最小最大损失更稳定，并且更不容易发生模式坍塌。

- **Wasserstein GAN (WGAN)** 和 **WGAN-GP**：这些使用Wasserstein距离，即使判别器表现良好时也能提供更平滑、更有意义的梯度。这种持续反馈帮助生成器更有效地学习分布，而不会轻易陷入困境。
- **最小二乘GAN (LSGAN)**：这用最小二乘目标取代了sigmoid交叉熵损失。它惩罚那些判别器识别为远离决策边界的样本，提供了更稳定的梯度，并阻止$G$在成功欺骗$D$后将样本推得太远进入“假”区域。

使用这些替代损失函数（将在“替代损失函数（WGAN, WGAN-GP, LSGAN）”一节中详细介绍）通常是提高训练稳定性和缓解模式坍塌最有效的方法之一。

### 特征匹配

特征匹配修改了生成器的目标函数，而不是最大化判别器的输出。目标是匹配判别器中间层提取的真实样本与生成样本的特征统计量。生成器的目标是最小化差异，这通常通过平均特征激活的平方差来衡量：


$$
||\mathbb{E}_{x \sim p_{data}} f(x) - \mathbb{E}_{z \sim p_z} f(G(z))||_2^2
$$


其中$f(x)$表示判别器中间层的激活。这可以防止生成器在当前判别器上过度训练，并鼓励它生成在各种特征上与真实数据统计量相似的样本。

### 其他技术

- **添加噪声：** 在判别器输入或权重 (weight)中注入噪声有时可以防止它变得过于自信和尖锐，使得生成器更难针对其特定弱点进行攻击。
- **经验回放：** 存储过去的生成样本并在训练判别器时偶尔重复使用它们，可以防止生成器“遗忘”之前学到的模式（灾难性遗忘）。
- **多GAN：** 可以使用涉及多个生成器专注于数据分布不同部分的集成方法，尽管这会大幅增加复杂性。

诊断和处理模式坍塌通常涉及实验。在整个训练过程中监测生成样本的多样性，并尝试不同的稳定化技术，是构建成功GAN模型的重要步骤。这里讨论的方法，特别是替代损失函数 (loss function)和如小批量判别等技术，为应对GAN训练中这个常见挑战提供了有效工具。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了生成对抗网络框架及其最小-最大目标函数，为模式崩溃提供了基础背景。
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems 29; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  提出了小批量判别和特征匹配技术，这两种是缓解模式崩溃并提高GAN训练稳定性的策略。
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems; Volume: 30; DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  介绍了WGAN-GP，通过梯度惩罚改进了Wasserstein GAN，从而实现更稳定的训练和更好的样本多样性。
