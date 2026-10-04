---
course: "generative-adversarial-networks-gans"
chapter: "gans-beyond-image-generation"
lesson: "gans-audio-synthesis"
sourceId: 2757
sourceUrl: "https://apxml.com/zh/courses/generative-adversarial-networks-gans/chapter-6-gans-beyond-image-generation/gans-audio-synthesis"
title: "基于GAN的音频合成 (WaveGAN, SpecGAN)"
description: "使用GAN架构生成原始音频波形或频谱图。"
order: 4
plots: []
sourceHash: "476bb9a8ca0a155b2ec6e2d42924dcde0395ade6347cb51edb0ac723205e76e8"
sourceCorrections: []
---

将生成对抗原理应用于音频合成，与图像生成相比，带来了一些特有的难题。音频信号本质上是一维时间序列，其特点是时间分辨率极高（例如，CD质量为每秒16,000个样本），并具有显著的长程依赖关系，这些关系构成音乐结构、语音模式和环境音。对生成模型来说，准确捕捉这些特性是一项具有挑战性的任务。对于使用GAN生成音频，目前出现了两种主要策略：直接生成原始音频波形，以及生成频谱图等中间表示。

### 对抗式音频生成中的挑战

在研究具体模型之前，了解其固有的挑战会很有帮助：

1. **高维度与时间分辨率：** 几秒钟的原始音频包含成千上万个样本。与用于图像的标准2D卷积网络相比，有效处理如此长的序列需要架构调整。
2. **长程依赖：** 音乐结构、旋律、和声，甚至连贯的语音都依赖于时间上相距较远的各点之间的关系。具有小感受野的标准卷积层难以有效捕捉这些长期关联。
3. **相位敏感性：** 与图像中相位信息通常在感知上不太重要不同，音频中频率分量之间的相位关系对于保真度非常重要。重建或生成连贯的相位信息并非易事。
4. **感知度量：** 评估生成音频的质量很复杂。来自图像生成的标准度量（如FID或IS，经过调整后）可能无法完全反映感知上的音频质量。主观听力测试通常仍是黄金标准，但难以规模化。

### WaveGAN：直接波形合成

WaveGAN通过直接合成原始音频波形来处理音频生成。该模型由Donahue等人于2018年提出，它将成功的深度卷积GAN (DCGAN) 架构调整用于一维音频数据。

**架构：**
WaveGAN没有使用图像GAN中常见的2D卷积和转置卷积，而是采用了它们的一维对应版本。

- **生成器：** 以潜在向量 (vector) $z$ 作为输入，使用一系列1D转置卷积层（有时称为分数步长卷积）逐步对表示进行上采样，直到达到所需的音频长度。
- **判别器：** 以真实或生成音频波形片段作为输入，使用一系列1D卷积层对其进行下采样，最终输出一个值，表示输入被认为是真实的还是伪造的。

为了处理长序列和依赖关系，WaveGAN架构通常使用：

- **更大的滤波器尺寸：** 与图像模型中典型的3x3或5x5核相比，1D卷积核通常更宽（例如25个样本）。这使得每一层在时域上具有更大的感受野。
- **显著的步幅：** 判别器中的步幅（以及生成器中相应的上采样因子）通常更大（例如4），以快速降低或增加时间分辨率。

> 图像GAN中使用的2D卷积与WaveGAN中使用的1D卷积的比较。1D滤波器仅沿时间维度滑动。

转置卷积中的一个特有挑战是“棋盘伪影”的潜在出现。WaveGAN引入了**相位混洗**，这是一项类似于图像模型中像素混洗的操作，但调整用于一维音频。它随机移动生成器每层激活的相位，目的是减少这些伪影并提升音频质量。

**训练：**
由于GAN训练中经常遇到的不稳定性，WaveGAN通常采用稳定化技术，例如带有梯度惩罚的Wasserstein GAN损失 (WGAN-GP)。

**优点：**

- 直接对原始波形进行建模，隐含地捕捉相位信息。
- 避免了单独的相位重建步骤。

**缺点：**

- 由于音频的高采样率，计算成本较高。
- 难以有效建模非常长程的依赖关系，有时会导致全局结构连贯性不足。

### SpecGAN：频谱图合成

另一种方法是通过在时频域操作来避免原始波形生成的复杂性。这涉及生成频谱图，它们是音频的2D表示。

**频谱图表示：**
频谱图通常使用短时傅里叶变换 (STFT) 计算。STFT将音频信号分解成短的、重叠的窗口，并计算每个窗口的傅里叶变换。这产生了一个2D表示，其中一个轴表示时间（对应于窗口），另一个轴表示频率，强度或颜色表示每个时间点上每个频率分量的幅度（或功率）。

**架构与训练：**
由于频谱图本质上是2D数据数组（类似于图像），专门用于图像合成的标准2D GAN架构可以直接应用。这包括DCGAN、WGAN-GP、StyleGAN、BigGAN等。

1. **预处理：** 真实音频数据使用STFT转换为频谱图。
2. **GAN训练：** GAN（生成器和判别器）使用这些2D频谱图作为目标数据分布进行训练。生成器学习从随机噪声向量 (vector) $z$ 生成看起来真实的频谱图。
3. **后处理（波形合成）：** 为了获得可听见的输出，生成的频谱图必须转换回原始音频波形。这一步很重要。由于STFT通常会丢弃相位信息（或者GAN只生成幅度频谱图），因此需要进行相位重建。常见方法包括：
   - **Griffin-Lim算法：** 一种迭代算法，尝试估计与生成的幅度频谱图一致的相位。它通常会产生可听见的伪影。
   - **声码器神经网络 (neural network)：** 更新的方法使用单独训练的神经网络（声码器），例如WaveNet、WaveGlow或MelGAN，这些网络专门设计用于根据频谱图（或Mel频谱图等相关表示）合成高保真波形。这些方法通常比Griffin-Lim产生更好的效果，但会在流程中增加另一个复杂的模型。

> 使用基于频谱图的GAN (SpecGAN) 生成音频的流程。

**优点：**

- 使用图像生成中强大且经过充分研究的2D卷积架构。
- 由于频谱图的维度较低，其计算要求比直接处理原始音频更低。
- 通常能更好地捕捉时频表示中存在的中长期结构。

**缺点：**

- 相位信息在STFT期间丢失，必须进行估计或生成，如果处理不当，这会限制音频保真度。
- 最终音频质量高度依赖于频谱图逆变换方法的有效性（Griffin-Lim或声码器）。
- 潜在的不匹配：GAN针对频谱图相似性进行优化，这可能与波形合成后的感知音频质量不完全对应。

### 比较与应用场景

WaveGAN和SpecGAN之间的选择通常取决于具体的应用和可用的资源。

- 当极其高的保真度和精确的相位建模很重要，且计算成本不是主要考虑因素时，**WaveGAN**可能更受青睐。它更简单，因为它避免了单独的声码器步骤。
- **SpecGAN**通常更实用，因为它计算需求较低，并且能够使用成熟的2D GAN架构。结合强大的神经网络 (neural network)声码器，它能在许多音频合成任务中（如音乐生成或文本到语音）取得领先成果。

近期研究持续探究混合方法以及对这两种方法论的改进。像GANSynth这样的技术学习幅度和相位的独立表示，而其他方法则将自注意力 (self-attention)机制 (attention mechanism)融入WaveGAN和SpecGAN的变体中，以更好地捕捉长程依赖关系。此外，对抗训练越来越多地被用于声码器内部，创建出基于GAN的声码器，目的在于从频谱图更快地合成更高保真度的波形。这些进展表明了将GAN有效应用于复杂的音频合成方面中的活跃发展。

## 参考资料

- [Adversarial Audio Synthesis](https://arxiv.org/abs/1802.04208) — Chris Donahue, Julian McAuley, Miller Puckette (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1802.04208](https://doi.org/10.48550/arXiv.1802.04208)
  介绍了WaveGAN，这是一种使用一维卷积GAN进行直接原始音频波形合成的开创性模型，通过相位洗牌解决生成挑战。
- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) — Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 30; Pages: 5767-5777; DOI: [10.48550/arXiv.1704.00028](https://doi.org/10.48550/arXiv.1704.00028)
  提出了带有梯度惩罚的Wasserstein GAN，这是一种用于稳定GAN训练的强大且被广泛采用的技术，对WaveGAN等模型尤为重要。
- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711) — Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, Aaron Courville (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 32; DOI: [10.48550/arXiv.1910.06711](https://doi.org/10.48550/arXiv.1910.06711)
  提出了一种快速且高保真的基于GAN的声码器，用于将梅尔频谱图转换为原始音频，对基于频谱图的音频合成方法质量至关重要。
