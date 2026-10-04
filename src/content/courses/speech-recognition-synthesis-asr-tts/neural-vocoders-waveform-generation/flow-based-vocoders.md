---
course: "speech-recognition-synthesis-asr-tts"
chapter: "neural-vocoders-waveform-generation"
lesson: "flow-based-vocoders"
sourceId: 3197
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/flow-based-vocoders"
title: "基于流的声码器 (WaveGlow, FloWaveNet)"
description: "使用归一化流实现并行波形生成。"
order: 3
plots: []
sourceHash: "72eb7b691f1e5fac3103f7908ed8c56c492845c345a24602d831542a83441afc"
sourceCorrections: []
---

自回归 (autoregressive)模型，如WaveNet和WaveRNN，虽然能达到高的音频保真度，但有一个明显的缺点：推理 (inference)速度慢，因为它们是按序列、逐样本生成的过程。即使生成几秒钟的音频也可能耗时很长，这使其难以用于实时应用。基于流的模型提供了一种不同的方法，可以实现并行波形生成，大幅缩短合成时间，同时保持高质量。

### 用于波形生成的归一化 (normalization)流

基于流的模型属于生成模型系列，称为归一化流。其基本设想是学习一个可逆映射 $f$，它存在于一个简单的基本分布 $p_Z(z)$（通常是标准高斯分布）与复杂的目标分布 $p_X(x)$（真实音频波形的分布）之间。如果我们能从 $p_Z(z)$ 中采样 $z$ 并计算 $x = f(z)$，就能生成逼真的音频。反之，给定一个音频样本 $x$，我们可以计算 $z = f^{-1}(x)$，并评估其在基本分布下的似然度。

变换 $f$ 被构建为一系列更简单的可逆函数：$f = f_L \circ \dots \circ f_2 \circ f_1$。为了使模型能够通过最大似然进行训练，必须满足两个条件：

1. 变换 $f$ 必须易于可逆；计算 $f^{-1}$ 应高效。
2. 雅可比矩阵 $\frac{\partial f}{\partial z}$ 的行列式必须高效计算。

变量变换公式使我们能够计算数据点 $x$ 的精确对数似然：


$$
\log p_X(x) = \log p_Z(z) + \log \left| \det \left( \frac{\partial f^{-1}(x)}{\partial x} \right) \right|
$$


等效地，使用反函数定理：


$$
\log p_X(x) = \log p_Z(f^{-1}(x)) - \log \left| \det \left( \frac{\partial f(z)}{\partial z} \right) \right|_{z=f^{-1}(x)}
$$


训练涉及在数据集上最大化此对数似然。

### WaveGlow

WaveGlow是基于流的神经声码器的一个显著实例，其灵感来源于最初为图像生成开发的Glow模型。它在梅尔频谱图的条件下，直接将高斯分布映射到语音波形。

**结构：**
WaveGlow采用单一的网络结构，可执行正向 ($x \to z$) 和反向 ($z \to x$) 两种变换。其核心组成部分是：

1. **仿射耦合层：** 这是流的构建模块。它们将输入数据 $z$（或中间激活）分成两半，$z_a$ 和 $z_b$。其中一半（$z_a$）通过一个变换网络（通常是像WaveNet的非因果扩张卷积那样的卷积网络）来计算缩放参数 (parameter) $s$ 和偏置 (bias)参数 $t$。这些参数随后使用仿射变换来变换另一半（$z_b$）：$z'_b = s \odot z_b + t$。第一半保持不变：$z'_a = z_a$。此操作易于可逆：$z_b = (z'_b - t) / s$。重要的是，此变换的雅可比行列式仅仅是缩放因子 $s$ 的乘积，使其计算高效。WaveGlow对音频略作修改，根据梅尔频谱特征应用此变换。
2. **可逆1x1卷积：** 在耦合层之间，可逆1x1卷积用于置换特征图的通道。这使得来自不同通道的信息能够在随后的耦合层之间混合，从而提升模型的表示能力。1x1卷积的雅可比行列式也易于计算。

> 概述了WaveGlow在推理 (inference)时的变换过程。高斯噪声通过仿射耦合层和可逆1x1卷积的多个步骤，以梅尔频谱图为条件，生成输出音频波形。每个步骤都是可逆的，从而可以通过最大似然进行训练。

**训练与推理：**
WaveGlow通过最大化真实音频波形在其对应梅尔频谱图条件下的对数似然进行训练。这涉及将音频 $x$ 通过正向变换 $f^{-1}$ 得到 $z = f^{-1}(x|\text{mel})$，然后最大化 $\log p_Z(z)$ 加上所有层的对数行列式之和。

推理是高度并行且高效的。您从标准高斯分布中采样一个随机向量 (vector) $z$，提供条件梅尔频谱图，并对网络进行单次正向传播，得到 $x = f(z|\text{mel})$。由于操作（卷积、仿射变换）可以在GPU上有效并行化，合成速度明显快于自回归 (autoregressive)模型。

### FloWaveNet

FloWaveNet是另一种基于流的声码器结构，与WaveGlow有相似之处，但其耦合层使用不同的结构，在仿射耦合块中融入了WaveNet扩张因果卷积的构想。与WaveGlow一样，它旨在通过学习从噪声到音频的可逆变换，在声学特征的条件下实现快速、并行的波形合成。

### 优点与考量

**优点：**

- **快速推理 (inference)：** 主要优势。并行生成使合成速度比自回归 (autoregressive)模型快几个数量级。
- **高保真度：** 能够生成高质量、听感自然的语音，通常可与自回归模型媲美。
- **精确似然训练：** 训练基于最大化数据的精确对数似然，提供了一个稳定且理论有据的目标函数，这与GANs不同，GANs可能更难训练。

**考量：**

- **模型大小：** 基于流的模型可能相当庞大，因为它们内部堆叠了耦合层和变换网络。
- **训练成本：** 训练这些模型需要大量的计算资源和大型数据集，与其他大型深度学习 (deep learning)模型类似。
- **稳定性：** 尽管能够实现高保真度，但有时可能被认为略逊于最佳自回归模型，可能会生成偶尔的伪影，尽管这一差距已大幅缩小。

基于流的声码器，如WaveGlow，代表着神经语音合成的重大进展，在音频质量和推理速度之间取得了极佳的平衡。它们在现代TTS系统中得到广泛应用，在这些系统中低延迟并非首要限制（与流式ASR不同），但快于实时合成的能力非常受欢迎。它们为自回归模型缓慢的生成速度以及基于GAN的声码器可能不太稳定的训练提供了一个有说服力的替代方案。

## 参考资料

- [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002) — Ryan Prenger, Rafael Valle, Bryan Catanzaro (2019)
  Journal: ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); DOI: [10.48550/arXiv.1811.00002](https://doi.org/10.48550/arXiv.1811.00002)
  介绍了WaveGlow架构及其在语音合成中的应用，实现了并行生成。
- [Glow: Generative Flow with Invertible 1x1 Convolutions](https://arxiv.org/abs/1807.03039) — Diederik P. Kingma, Prafulla Dhariwal (2018)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2018); Publisher: Neural Information Processing Systems Foundation; DOI: [10.48550/arXiv.1807.03039](https://doi.org/10.48550/arXiv.1807.03039)
  介绍了Glow模型，WaveGlow在此基础上进行改进，其特点是使用了仿射耦合层和可逆1x1卷积来实现生成流。
- [FloWaveNet: A Generative Flow for Raw Audio](https://doi.org/10.21437/Interspeech.2019-2169) — Taejun Kim, Hyoung-Seok Kwon, Jae-Sung Bae, Kyeong-Jin Mun, Kook-Cho Roh (2019)
  Journal: Proc. Interspeech 2019; Publisher: International Speech Communication Association (ISCA); Pages: 3495-3499; DOI: [10.21437/Interspeech.2019-2169](https://doi.org/10.21437/Interspeech.2019-2169)
  描述了FloWaveNet，本节中提到的另一种基于流的声码器架构，专为原始音频生成设计。
- [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) — Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio (2017)
  Journal: International Conference on Learning Representations (ICLR 2017); DOI: [10.48550/arXiv.1605.08803](https://doi.org/10.48550/arXiv.1605.08803)
  介绍了Real NVP，一个使用仿射耦合层的归一化流模型，仿射耦合层是本节中讨论的基于流模型的核心组件。
