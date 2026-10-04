---
course: "cnns-for-computer-vision"
chapter: "gans-image-synthesis"
lesson: "gan-training-challenges"
sourceId: 2617
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-7-gans-image-synthesis/gan-training-challenges"
title: "训练生成对抗网络的挑战"
description: "论述训练生成对抗网络时遇到的常见困难，例如模式崩溃、不收敛和不稳定性。"
order: 2
plots: ["plots/2617-0.json"]
sourceHash: "9ed91fd1de14b7edc75761133c9f4a507b867a630f709ee0b64cf8265292466c"
sourceCorrections: []
---

训练生成对抗网络 (GAN)常被形容为一场细致的平衡。生成器（$G$）和判别器（$D$）被锁定在一场竞争性博弈中，正式来说是一个零和博弈，即一个网络的收益是另一个网络的损失。目的是找到纳什均衡，这是一种任一玩家通过单方面改变策略都无法改善自身结果的状态。然而，在神经网络 (neural network)参数 (parameter)的高维非凸空间中找到这种均衡是公认的难题。在此审视GAN训练中遇到的主要困难。

## 不收敛

最主要的困难是训练过程可能就是无法收敛。$G$和$D$的更新是基于从各自损失函数 (loss function)中推导出的梯度。在标准极小极大博弈公式中：


$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{数据}}(\mathbf{x})} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{\mathbf{z}}(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))]
$$


$G$和$D$的梯度下降 (gradient descent)更新并非总是导向预期的均衡状态。请考虑以下情况：

- **判别器能力远超生成器：** 如果$D$在早期变得过于擅长，它能轻松区分真实样本和伪造样本。$G$的损失，即$\log(1 - D(G(\mathbf{z})))$，可能会饱和。如果$D(G(\mathbf{z}))$接近0（意味着$D$确信样本是伪造的），那么$G$的梯度会变得非常小（消失），提供很少关于如何改进的信息。$G$基本上停止学习。
- **生成器能力远超判别器：** 相反，如果$G$迅速学会生成能欺骗$D$的样本，$D$可能难以提供有用的梯度。它的准确率可能徘徊在50%左右（随机猜测），并且其损失可能不会显著下降，阻碍了其引导$G$生成在整个数据分布中更真实样本的能力。

这种缺乏稳定的收敛性意味着$G$和$D$的损失曲线在训练期间经常大幅波动，其本身不一定表示样本质量正在提高。

## 模式崩溃

模式崩溃可能是GAN训练中最广为人知的失败模式。当生成器$G$只学习生成真实数据分布中可能输出的一小部分时，就会发生这种情况。$G$没有捕捉训练数据的完整多样性，而是找到一个或几个“模式”（输出类型），这些模式在欺骗当前判别器$D$方面特别有效，并专门生成这些。

想象一下在手写数字MNIST数据集上训练一个GAN。理想情况下，$G$应该学习生成0到9所有数字的真实图像。在模式崩溃的情形下，$G$最终可能只生成看起来像数字'1'的图像，或者可能是'1'和'7'，完全忽略其他数字。即使生成的'1'高度真实并能欺骗$D$，生成器也未能学习到真实的潜在数据分布。

**为什么会发生？** $G$的目标是最小化其损失，这通常表现为最大化$D$将其输出分类为真实的概率。如果$G$发现一个$D$持续误分类的输出，它有很强的动机持续生成该输出的变体。研究输出空间的其他部分可能风险更高，并在初期导致更高的损失。这导致$G$“坍缩”到少数几个安全的模式上。

模式崩溃可以是部分（缺少某些模式）或完全的（只生成一种类型的输出）。它表明$G$没有学习到真实数据中的复杂性和多样性。



![2D模式崩溃示意图](plots/2617-0.json)



> 模式崩溃的示例。真实数据有两个明显的模式（蓝色和绿色簇），但生成器（红色叉）只学习生成对应于第一个模式的样本。

## 训练不稳定

GAN训练可能高度不稳定。$G$和$D$的参数 (parameter)可能会剧烈波动而非平稳收敛。这种不稳定性通常表现为：

- **损失波动：** $G$和$D$的损失值可能剧烈波动而不安稳。
- **对超参数 (hyperparameter)的敏感性：** GAN通常对学习率、优化器参数（如Adam中的动量）和批量大小的选择非常敏感。微小变化有时可能导致截然不同的结果，包括彻底的训练失败。
- **模型架构敏感性：** 为$G$和$D$选择的特定架构能显著影响稳定性。某些架构选择（例如，缺乏归一化 (normalization)、不合适的激活函数 (activation function)）会加剧不稳定性。

主要问题仍然是平衡训练动态的困难。如果$G$相对于$D$更新过快，它可能会迅速抓住$D$的弱点，可能导致模式崩溃。如果$D$更新过快，它可能会抑制$G$的学习信号。这需要仔细调整，并且通常涉及启发式方法或架构限制（如DCGAN中引入的，稍后讨论）来稳定过程。

## 梯度问题

GAN的对抗性质在反向传播 (backpropagation)过程中可能导致特定的梯度问题：

- **梯度消失：** 如不收敛部分所述，如果判别器变得过于准确，生成器的损失函数 (loss function)可能会饱和，导致梯度消失。原始的极小极大损失，特别是$\log(1 - D(G(\mathbf{z})))$，在$D(G(\mathbf{z}))$接近0时，已知容易受到此影响。已提出替代损失函数，例如非饱和启发式损失（其中$G$最大化$\log D(G(\mathbf{z}))$而非最小化$\log(1 - D(G(\mathbf{z})))$）或Wasserstein损失（WGAN），专门通过提供更多有益的梯度来缓解此问题。
- **梯度爆炸：** 虽然在GAN中不如梯度消失常见，但梯度有时会变得过大，导致大的参数 (parameter)更新并引发发散。梯度裁剪（限制梯度的最大幅度）等技术，您可能在第二章中看到过，是预防这种情况的标准做法。

## 评估困难

与典型的监督学习 (supervised learning)任务中损失下降通常表示模型正在改进不同，GAN训练中$G$和$D$的损失曲线通常是图像质量或多样性的不良指标。$D$的损失可能下降是因为它变得更好了，或者是因为$G$崩溃并生成了易于检测的伪造品。$G$的损失可能下降是因为它成功欺骗了弱的$D$，不一定是因为它正在生成真正真实的图像。

这种缺乏可靠、可解释的损失指标使得难以：

- 知道何时停止训练。
- 仅基于损失值有效比较不同模型或超参数 (parameter) (hyperparameter)。
- 调试训练问题。

因此，评估GAN通常依赖于对生成样本的目视检查以及旨在评估质量和多样性的定量指标，例如Fréchet Inception 距离（FID）和Inception 分数（IS）。这些指标将在本章后面介绍，它们提供更有意义的评估，但通常是离线计算，并且在训练循环本身期间不提供实时反馈。

解决这些挑战一直是GAN研究的一个主要关注点，导致损失函数 (loss function)、正则化 (regularization)技术、架构设计和训练过程的众多改进，一些内容我们将在后续章节中查看。理解这些潜在困难是成功训练您自己生成模型的第一步。

## 参考资料

- [Generative Adversarial Networks](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates; Volume: 27; Pages: 2672-2680
  介绍了生成对抗网络框架和原始的 minimax 目标，强调了早期的训练难题。
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) — Alec Radford, Luke Metz, Soumith Chintala (2016)
  Journal: International Conference on Learning Representations; Publisher: Proceedings of Machine Learning Research (PMLR); Volume: 49; DOI: [10.48550/arXiv.1511.06434](https://doi.org/10.48550/arXiv.1511.06434)
  提出了用于稳定深度卷积 GAN 的架构指导原则，显著提高了训练稳定性和生成样本的质量。
- [Wasserstein Generative Adversarial Networks](https://arxiv.org/pdf/1701.07875.pdf) — Martin Arjovsky, Soumith Chintala, Léon Bottou (2017)
  Journal: International Conference on Machine Learning; Pages: 214-223
  提出了 Wasserstein GAN，它使用 Earth-Mover 距离提供更稳定的梯度并解决 GAN 训练中的梯度消失问题。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://proceedings.neurips.cc/paper/2017/file/8a1d694707635956a81389c0d36c97d9-Paper.pdf) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 30; Pages: 6626-6637; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  引入了 Fréchet Inception 距离 (FID) 作为评估 GAN 生成图像质量和多样性的定量指标。
