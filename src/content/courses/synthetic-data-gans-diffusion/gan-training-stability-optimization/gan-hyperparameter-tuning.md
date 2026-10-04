---
course: "synthetic-data-gans-diffusion"
chapter: "gan-training-stability-optimization"
lesson: "gan-hyperparameter-tuning"
sourceId: 5682
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-3-gan-training-stability-optimization/gan-hyperparameter-tuning"
title: "GAN的超参数调整策略"
description: "调整学习率、批大小和网络结构的最佳实践。"
order: 6
plots: []
sourceHash: "c749eb82d17f170bd891dd3c6f42c42ff1a48fc970960022a906f92c454e7917"
sourceCorrections: []
---

生成对抗网络 (GAN) (GANs) 的超参数 (parameter) (hyperparameter)调整常常感觉更像是艺术而非科学，特别是与标准监督学习 (supervised learning)模型相比。生成器 ($G$) 和判别器 ($D$) 之间最小-最大博弈中所需的精细平衡，使得训练过程对这些设置非常敏感。选择不当很容易导致不稳定问题，例如振荡、发散或模式崩溃，这些挑战在GAN训练中经常出现，即使使用了高级损失函数 (loss function)或正则化 (regularization)也是如此。这里将介绍有效应对这一复杂调整过程的策略。

## 识别GAN中的重要超参数 (parameter) (hyperparameter)

尽管许多超参数会影响训练，但有些超参数始终对GAN的稳定性和性能有明显影响：

- **学习率：** 为生成器 ($\alpha_G$) 和判别器 ($\alpha_D$) 设置单独的学习率是常规做法。它们的绝对值和相对比率很重要。如双时间尺度更新规则 (TTUR) 中所讨论的，使用不同的学习率（例如 $\alpha_D > \alpha_G$）有时可以稳定训练。
- **优化器选择：** Adam被广泛使用，但其动量参数 ($\beta_1$, $\beta_2$) 能明显影响稳定性。像 $\beta_1 = 0.5$（而非默认的0.9）这样的值常被推荐用于GANs，以减少动量引起的振荡。RMSprop甚至SGD有时也会被使用，取决于具体的GAN变体和数据集。
- **批大小：** 影响梯度估计方差和训练速度。更大的批大小有时可以通过提供更好的梯度估计来稳定训练，但它们需要更多内存，并可能导致更尖锐的最小值，可能损害泛化能力。批大小也与某些归一化 (normalization)技术（如批归一化）和损失函数 (loss function)有关系。
- **网络结构：** 深度、宽度、层类型（卷积、注意力）、归一化层（批归一化、实例归一化、层归一化）以及激活函数 (activation function)（ReLU、LeakyReLU）都起作用。结构选择常与其他超参数有密切联系。例如，谱归一化的有效性可能取决于网络深度。
- **正则化 (regularization)强度：** 如果使用梯度惩罚（如WGAN-GP中）等技术或其他形式的正则化（例如权重 (weight)衰减、一致性正则化），相应的权重系数（$\lambda_{GP}$等）是需要调整的重要超参数。
- **损失函数参数：** 一些替代损失函数有自己的参数（例如某些hinge损失中的margin值）。
- **潜在向量 (vector)维度：** 输入噪声向量 $z$ 的大小会影响生成器的表示能力。

## GAN超参数 (parameter) (hyperparameter)调整方法

标准超参数优化技术可以应用，但需要仔细考虑GAN特有的难点。

### 手动调整和启发式方法

鉴于其敏感性和彼此关联，从已有的启发式方法开始是常见做法：

- **从简单开始：** 从相关工作或基线实现中已知的良好结构和超参数 (parameter) (hyperparameter)集开始。
- **首先调整学习率：** 学习率通常是最敏感的参数。尝试通常在 $10^{-5}$ 到 $10^{-3}$ 范围内的值。考虑TTUR方法。
- **调整优化器参数：** 如果使用Adam，尝试 $\beta_1 = 0.5$ 和 $\beta_2 = 0.999$。
- **监控训练动态：** 密切关注生成器和判别器的损失曲线。它们不一定非要收敛到零，但理想情况下应该稳定。寻找发散（损失爆炸）或模式崩溃的迹象（生成器损失迅速下降而判别器损失增加，表明生成器找到了一个判别器不容易检测到的单一输出）。同时定期监控梯度范数和输出样本质量。
- **一次只改变一项：** 一次只改变一个超参数（或一个相关的对，如学习率），以了解其具体效果。

### 自动化搜索方法

尽管手动调整提供了直觉，但自动化方法可以更系统地研究超参数 (parameter) (hyperparameter)空间。

- **随机搜索：** 对于高维空间 (high-dimensional space)通常比网格搜索更有效。从预定义范围中随机抽取超参数配置。它计算密集但可并行化。
- **贝叶斯优化：** 构建目标函数（例如FID分数、IS，或基于损失稳定性的代理）的概率模型（例如高斯过程），并使用它来选择接下来要评估的有前景的超参数配置。这比随机搜索的样本效率明显更高，考虑到训练GANs的高成本，这很有帮助。Optuna、Hyperopt或Ray Tune等工具可以促进这一点。

> GAN超参数、稳定技术和常见训练问题之间的相互依赖关系。调整涉及到理解并应对这些关系。

### 调整期间的评估

超参数 (parameter) (hyperparameter)调整需要一个客观的衡量标准。尽管判别器或生成器损失可以提供训练期间稳定性的线索，但它们在表示最终样本质量或多样性方面表现不佳。

- 使用第五章中讨论的指标，例如Frechet Inception距离 (FID) 或Inception分数 (IS)，定期计算或在固定数量的训练迭代后计算。
- 对生成样本的视觉检查仍然必不可少，用于识别像伪影或缺乏精细细节等细节问题，这些是量化 (quantization)指标可能遗漏的。
- 对于自动化搜索，选择一个平衡质量和多样性的指标（如FID），并且可以足够高效地计算以允许多次试验。

## 实用建议

- **参考前期工作：** 从论文中报告的适用于类似任务或结构的超参数 (parameter) (hyperparameter)开始。除非必要，否则不要从零开始。
- **隔离调整：** 如果可能，将结构选择与优化器/正则化 (regularization)参数分开调整。
- **资源分配：** GAN调整是资源密集型的。使用较小的数据集或较低分辨率的图像进行初步的广泛搜索，然后，在完整数据/分辨率上微调 (fine-tuning)参数。
- **耐心和监控：** GAN训练可能需要时间才能稳定或显示问题。使用TensorBoard或Weights & Biases等工具密切监控运行情况，以便跟踪损失、梯度范数和样本质量随时间的变化。

为GAN找到合适的超参数通常涉及迭代细化。通过理解每个参数的作用，采用系统搜索方法，并仔细监控训练动态和评估指标，您可以大大提高实现稳定训练和生成高质量合成数据的机会。

## 参考资料

- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper/2017/hash/892c91e0a653ba199e53026a79357404-Abstract.html) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30; DOI: [10.55989/NIPS-2017-1049](https://doi.org/10.55989/NIPS-2017-1049)
  介绍了带有梯度惩罚的Wasserstein GAN (WGAN-GP)，这是一种稳定GAN训练并缓解模式崩溃的有效方法。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://papers.nips.cc/paper/2017/hash/694216fcb37498c88fcd35a64b1d7d07-Abstract.html) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 6626-6637; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  提出了双时间尺度更新规则（TTUR），表明为生成器和判别器使用不同的学习率可以稳定GAN训练。
- [Hyperparameter Optimization: A Review of Algorithms and Applications](https://www.mdpi.com/2076-3417/11/14/6312) — Lars Hillebrand, Marius Brehler, Philipp R. Schauer, Steffen W. W. Meyer, Christian Bockermann (2021)
  Journal: Applied Sciences; Volume: 11; Pages: 6312; DOI: [10.3390/app11146312](https://doi.org/10.3390/app11146312)
  全面回顾了各种超参数优化算法，包括随机搜索和贝叶斯优化，对GAN的系统化调优具有参考价值。
