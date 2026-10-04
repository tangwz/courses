---
course: "generative-adversarial-networks-gans"
chapter: "gans-beyond-image-generation"
lesson: "gans-discrete-data-text"
sourceId: 2749
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gans-discrete-data-text"
title: "离散数据带来的难题：文本生成"
description: "由于不可微性，将标准GAN应用于文本等离散序列存在困难。"
order: 1
plots: []
sourceHash: "c3041702b753337f8498e30b6f198dc58d006316bf21399edbcf882426343c2d"
sourceCorrections: []
---

生成对抗网络 (GAN)在生成高质量图像方面取得了显著的成就，这主要是因为图像存在于连续空间中，其中像素值的渐进变化具有意义。然而，将GAN应用于生成离散序列（例如自然语言文本）时，带来了一系列独特的重大难题，这些难题与图像合成中遇到的问题有根本不同。

文本由离散单元组成：字符、单词或从有限词汇表 (vocabulary)中选取的标记 (token)。文本生成GAN通常包含一个生成器，该生成器在序列的每一步为下一个标记输出词汇表上的概率分布。为形成实际的文本序列，需要一个采样步骤，通常涉及根据这些概率选择一个标记（例如，使用`argmax`或多项式采样）。

主要难题正是出现在这个采样步骤。考虑标准的GAN训练过程：判别器评估生成的序列并产生一个损失信号。这个损失需要经过判别器，然后*经过生成过程*，反向传播 (backpropagation)回生成器的参数 (parameter)以便更新它们。然而，从离散分布中采样的行为其本身是不可微的。

设 $G$ 为生成器，$D$ 为判别器。生成器 $G(z)$ 产生一个在词汇表 $V$ 上的概率分布序列 $P_1, P_2, ..., P_T$，其中 $z$ 是一个潜在向量 (vector)。为得到具体的序列 $S = (w_1, w_2, ..., w_T)$，我们在每一步 $t$ 采样 $w_t \sim P_t$。判别器接着评估 $D(S)$。问题在于，我们无法计算判别器损失相对于生成器参数 $\nabla_{\theta_G} D(S)$ 的梯度，因为采样步骤 $w_t \sim P_t$ 中断了梯度流动。没有直接的方法来计算生成器参数 $\theta_G$ 的微小变化（会影响概率 $P_t$）将如何改变与*特定采样序列* $S$ 相关的损失。


$$
\text{生成器输出: } P = G(z;\theta_G) \rightarrow \text{采样: } S \sim P \rightarrow \text{判别器评估: } L = D(S)
$$


梯度 $\frac{\partial L}{\partial \theta_G}$ 存在问题，因为采样步骤 $S \sim P$ 是不可微的。

这种缺少从判别器评估直接返回生成器的梯度路径的情况，阻碍了使用标准梯度下降 (gradient descent)算法（如Adam或SGD）来有效训练生成器。生成器基本上没有收到任何有用的信号，说明如何调整其内部权重 (weight)以生成更能欺骗判别器的序列。它可能学会生成有效的概率分布，但它难以学习连贯文本所特有的复杂、长程依赖关系和结构，因为学习信号要么不存在，要么极其稀疏且方差很高。

相反，图像生成器通常直接输出像素值（或连续映射到像素的值）。生成器输出的微小变化对应于图像中微小、连续的变化，使得梯度可以从判别器的评估平滑地流回。

这个根本的不可微障碍是原始GAN框架直接应用于文本生成往往不成功并导致不稳定或糟糕结果的主要原因。克服这一点需要不再局限于标准的反向传播，这促使了替代训练策略的出现，我们接下来将考察这些策略，例如强化学习 (reinforcement learning)的设定以及为离散采样过程提供连续近似的技术。

## 参考资料

- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://ojs.aaai.org/index.php/AAAI/article/view/10804) — Lantao Yu, Weinan Zhang, Jun Wang, Yong Yu (2017)
  Journal: Proceedings of the AAAI Conference on Artificial Intelligence; Publisher: Association for the Advancement of Artificial Intelligence; Volume: 31; Pages: 2883-2889; DOI: [10.1609/aaai.v31i1.10804](https://doi.org/10.1609/aaai.v31i1.10804)
  这篇开创性论文提出了最早的有效方法之一，通过将生成器的任务表述为强化学习问题，训练生成对抗网络处理文本等离散序列数据，直接解决了不可微分性问题。
- [GANS for Sequence Generation: A Review of Promises and Pitfalls](https://ieeexplore.ieee.org/document/9211327) — Mohammad Reza Zafarani, Amirhossein Khosravi (2020)
  Journal: IEEE Access; Publisher: IEEE; Volume: 8; Pages: 181816-181829; DOI: [10.1109/ACCESS.2020.3028302](https://doi.org/10.1109/ACCESS.2020.3028302)
  这篇综述论文讨论了生成对抗网络在序列生成中的应用，详细阐述了序列离散性带来的根本性挑战，并回顾了各种提出的解决方案及其局限性。
