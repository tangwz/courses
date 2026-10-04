---
course: "generative-adversarial-networks-gans"
chapter: "gan-training-stabilization"
lesson: "mode-collapse-causes-consequences"
sourceId: 2687
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-3-gan-training-stabilization/mode-collapse-causes-consequences"
title: "模式坍塌：成因与后果"
description: "细致介绍模式坍塌问题，即生成器只生成有限种类的输出。"
order: 2
plots: []
sourceHash: "092ecd2fae24846874f0d119b8e611fdccb2dab48e331826a92943c3a6c6ec97"
sourceCorrections: []
---

**模式坍塌**是GAN训练中一个最常遇到且令人沮丧的问题。简而言之，模式坍塌指的是生成器学会只生成极其有限种类的输出，常常集中于真实数据分布中某个单一或一小部分模式。生成器没有学到训练数据集的丰富差异，而是将其输出分布实质上“坍塌”到少数几个样本上，这些样本对当前的判别器特别有效。

设想在包含各种手写数字（0到9）图像的数据集上训练GAN。一个理想的生成器会学着生成所有十种数字的合理样本。然而，一个发生模式坍塌的生成器可能只生成看起来像数字“1”的图像，或者可能只生成“1”和“7”，完全忽略了实际数据分布$p_{data}$中存在的其他数字（模式）。

> 真实数据分布（左侧，蓝色）显示出多种模式（簇）。发生模式坍塌的生成器（右侧，红色）将其输出集中在这些模式中的某一种上，未能涵盖整体的多变性。

### 模式坍塌的成因

了解模式坍塌发生的原因，需要审视极小极大博弈的动态和原始GAN目标函数的性质：

1. **生成器的驱动力：** 生成器的首要目标是生成判别器判断为真实的样本。如果生成器发现某种特定类型的样本能够持续有效地欺骗当前判别器，它就会有很强的驱动力去持续生成该样本的变体。它可能会觉得，与其在整个数据空间中寻找并同时学到多种模式，不如专注于在一种模式内实现完美生成更容易。
2. **判别器的作用：** 判别器过于强大或学习速度过快会加剧这个问题。如果判别器变得非常擅长区分真实样本和生成器当前输出的样本，它可能会给生成器提供陡峭但信息量不足的梯度。生成器可能会学到，其当前输出的*微小*变体很容易被识别为虚假，这会将其推回它所发现的单一成功模式，而不是引导它去数据分布中未被触及的区域。
3. **目标函数：** 原始GAN目标函数，它隐式地最小化$p_{data}$和$p_g$之间的詹森-香农（JS）散度，对此有很大影响。JS散度有其局限性，特别是在分布$p_{data}$和$p_g$几乎没有重叠，或存在于高维像素空间中的低维流形上时（这是一种常见情况）。在这种情况下，JS散度会饱和，导致生成器的梯度消失。这意味着生成器几乎接收不到关于如何调整其参数 (parameter)以更好地符合真实数据分布的信号，使其难以摆脱坍塌状态。优化过程实质上使生成器陷入一个不佳的局部最小点。
4. **优化不稳定性：** 极小极大优化本身与标准的监督学习 (supervised learning)最小化相比，具有内在的不稳定性。生成器和判别器不断相互适应。这种动态可能导致振荡或循环：生成器找到一种模式，判别器学会识别它，生成器又跳到另一种容易找到的模式，如此循环，却从未收敛到一个能覆盖完整分布的状态。

### 模式坍塌的后果

模式坍塌的首要后果是生成样本的**多样性**严重不足。虽然生成的单个样本可能看起来很真实（局部合理），但整体的样本集合未能体现训练数据中固有的变化性。

- **用途受限：** 对于需要多样化输出的应用（例如，生成不同艺术风格、模拟各种情境），一个发生模式坍塌的GAN实际上毫无用处。
- **评估偏差：** 如果评估仅依赖于少数样本的视觉质量，模式坍塌可能不会被察觉。这些样本可能看起来不错，但它们未能反映生成器对真实数据分布进行建模的能力。
- **学习失败：** 从根本上讲，模式坍塌表明生成器未能学到目标数据的深层结构和变化性。

模式坍塌明确表明，标准GAN训练设置可能很脆弱。这指出了对更精巧的损失函数 (loss function)和稳定化技术的需求，这些技术能提供更有意义的梯度，促使生成器多方位生成，并阻止生成器局限于狭窄、不具代表性的输出分布。后续章节中讨论的方法，例如Wasserstein距离和梯度惩罚，直接解决了这些不足。

## 参考资料

- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍原始GAN框架，定义了最小最大目标及其与Jensen-Shannon散度的联系，这些是理解模式崩溃的基础。
- [Wasserstein GAN](http://proceedings.mlr.press/v70/arjovsky17a.html) — Martin Arjovsky, Soumith Chintala, and Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning; Publisher: PMLR; Volume: 70; Pages: 214-223
  提出使用Wasserstein-1距离作为GAN损失函数，以提供更稳定的梯度，并解决模式崩溃和梯度消失问题。
- [Improved Training of Wasserstein GANs](https://proceedings.neurips.cc/paper_files/paper/2017/file/892c3b1c6dccd52936e275f2ff8417d7-Paper.pdf) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville (2017)
  Journal: Advances in Neural Information Processing Systems 30; Pages: 5767-5777
  引入梯度惩罚来强制WGAN中的Lipschitz约束，显著提高了训练稳定性和样本质量。
- [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498) — Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 29; Pages: 2234-2242; DOI: [10.48550/arXiv.1606.03498](https://doi.org/10.48550/arXiv.1606.03498)
  提出包括minibatch discrimination在内的多项技术，以缓解模式崩溃并稳定GAN训练。
