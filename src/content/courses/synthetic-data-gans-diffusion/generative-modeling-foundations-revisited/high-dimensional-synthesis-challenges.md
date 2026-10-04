---
course: "synthetic-data-gans-diffusion"
chapter: "generative-modeling-foundations-revisited"
lesson: "high-dimensional-synthesis-challenges"
sourceId: 5634
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-gans-diffusion/chapter-1-generative-modeling-foundations-revisited/high-dimensional-synthesis-challenges"
title: "高维数据生成中的挑战"
description: "讨论维度灾难和模式崩溃等常见难题。"
order: 3
plots: []
sourceHash: "dd6d714c7ffe5b4541410f5e47f27d574500870db682ece701995c8a0f21d80b"
sourceCorrections: []
---

生成简单表格数据或低分辨率图像会遇到一些障碍，但在处理高维数据时，例如百万像素图像、长音频波形或复杂视频序列，这些困难会大大增加。随着维度（$d$）的增加，空间的性质会发生显著改变，这给旨在学习潜在数据分布 $p_{data}(x)$ 的生成模型带来了若干根本性挑战。

### 维度灾难

其中一个最重要的障碍是“维度灾难”。在高维空间 (high-dimensional space)中，体积呈指数增长，随着维度的增加。因此，任何有限数据集都会变得极其稀疏。想象将点均匀地散布在一个单位立方体中。随着维度的增加，大多数点会靠近边界，并且相邻点之间的距离会变大。

这种稀疏性使得生成模型极其难以准确估计真实数据分布 $p_{data}(x)$。模型可能会遇到空间中大量的区域没有训练样本，使泛化变得困难。学习有意义的结构需要巨量数据或模型架构中内置的非常强的先验知识。例如，生成一张逼真的1024x1024像素图像，意味着在一个超过300万维（考虑RGB通道）的空间中建模分布。在如此广大的空间中捕获像素之间复杂的依赖关系是一项艰巨的任务。

### 模式崩溃

一种常见的失败模式，在生成对抗网络 (GAN) (GANs) 中尤为突出，但也与其他生成方法相关，是“模式崩溃”。这发生在生成器 $G$ 只学习生成真实数据分布中有限的输出类型子集时。生成器 $G$ 没有捕获 $p_{data}(x)$ 的全部多样性，而是将许多不同的输入潜在向量 (vector) $z$ 映射到相同或非常相似的输出。

例如，如果在包含不同种族、姿势和表情的人脸数据集上训练GAN，模式崩溃可能导致生成器仅生成单一民族的人脸或仅生成正面肖像，而与输入噪声 $z$ 无关。判别器 $D$ 可能擅长识别这些少数模式为真实的，但生成器未能生成数据分布的其他部分，导致样本多样性差。这从根本上损害了生成真正代表原始数据集的合成数据的目标。

> 模式崩溃发生在生成器只生成有限种类输出时，未能捕获目标数据分布的多样性，即使采样了不同的潜在向量。

减轻模式崩溃通常需要修改GAN目标函数或架构，这些技术我们将在第3章进行查看。

### 训练不稳定性

生成模型，特别是GANs，以不稳定的训练动态著称。GAN目标函数所描述的最小-最大博弈难以优化。常见问题包括：

- **损失波动：** 生成器和判别器损失可能会剧烈波动而无法收敛。
- **梯度消失：** 判别器过快变得过于强大，向生成器提供无用（接近零）的梯度。
- **梯度爆炸：** 不稳定的更新导致梯度过大，引起数值溢出。

实现生成器和判别器共同进步的稳定平衡，需要仔细调整超参数 (parameter) (hyperparameter)、网络架构和优化策略。扩散模型通常表现出更稳定的训练，但仍可能对噪声调度和网络参数化等选择敏感。第3章和第4章将分别具体介绍稳定GAN和扩散模型训练的技术。

### 计算成本

生成高维数据，特别是高分辨率图像或视频，需要大量的计算资源。StyleGAN或大型扩散模型等先进模型通常涉及拥有数亿甚至数十亿参数 (parameter)的网络。训练这些模型需要：

- **强大硬件：** 通常需要多台高端GPU或TPU。
- **大内存：** 存储高分辨率数据的模型参数、激活和梯度需要大量的GPU内存。
- **延长训练时间：** 在分布式计算集群上，训练可能需要数天、数周甚至数月。

从训练好的模型中采样也可能计算密集，特别是对于传统上需要许多顺序去噪步骤的扩散模型（尽管存在像DDIM这样的更快采样方法，如第4章所述）。

### 评估难度

评估生成的高维样本的质量本身就存在难度。我们如何量化 (quantization)测量生成的图像是否“真实”或者生成的分布 $p_{gen}(x)$ 是否与真实分布 $p_{data}(x)$ 匹配？尽管视觉检查有参考价值，但它是主观的，并且不易扩展。

像Inception Score (IS) 或Fréchet Inception Distance (FID) 这样的标准度量对于图像来说已变得流行，但它们有自己的局限性。它们依赖于预训练 (pre-training)分类器（如InceptionNet）的嵌入 (embedding)，这可能无法捕获与特定任务相关的所有图像质量或多样性方面。此外，为音频或结构化数据等其他数据类型定义评估度量仍是一个活跃的研究方向。我们将在第5章专门介绍各种评估度量及其解释。

### 维护全局结构和长距离依赖

高维数据通常具有复杂的内部结构和长距离依赖。例如：

- **图像：** 逼真的物体具有特定的形状、纹理和空间关系。生成连贯图像需要协调大距离上的像素值。
- **音频：** 音乐具有和声结构、节奏和旋律；语音具有语法规则和随时间推移一致的说话者特征。
- **视频：** 场景必须在物体、光照和运动方面保持时间上的一致性。

生成模型，特别是那些基于感受野有限的卷积层的模型，可能难以捕获这些全局属性。尽管带有注意力机制 (attention mechanism)的GANs或在序列上操作的扩散模型等架构已取得进展，确保长距离连贯性仍然是一个持续的挑战，特别是随着数据维度的增加。

克服这些挑战对于推动合成数据生成的发展来说非常重要。后续章节中介绍的复杂架构和技术，例如StyleGAN基于风格的控制、ProGAN的渐进式增长、CycleGAN的域适应以及扩散模型的迭代细化过程，都部分旨在解决建模高维数据分布中固有的这些根本性困难。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: 22-24
  本书讨论了维度灾难现象及其对统计建模和机器学习的影响。
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems 27; Volume: 27; Pages: 2672-2680; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  这篇开创性论文介绍了生成对抗网络，为讨论的许多挑战（包括训练动态和模式崩溃）奠定了基础。
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Martin Arjovsky, Soumith Chintala, and Léon Bottou (2017)
  Journal: Proceedings of the 34th International Conference on Machine Learning (ICML); Volume: 70; Pages: 214-223; DOI: [10.48550/arXiv.1701.07875](https://doi.org/10.48550/arXiv.1701.07875)
  提出了基于 Wasserstein 距离的新 GAN 目标，以稳定训练并缓解模式崩溃，直接应对了一个核心挑战。
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500) — Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter (2017)
  Journal: Advances in Neural Information Processing Systems 30; Volume: 30; Pages: 6667-6678; DOI: [10.48550/arXiv.1706.08500](https://doi.org/10.48550/arXiv.1706.08500)
  介绍了 Fréchet Inception Distance (FID) 作为评估生成图像质量和多样性的稳健指标，解决了评估方面的困难。
