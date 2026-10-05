# 基于GAN的声码器（MelGAN, HiFi-GAN）

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-5-neural-vocoders-waveform-generation/gan-based-vocoders)

[返回章节目录](README.md) · [返回课程目录](../README.md)

自回归 (autoregressive)模型如WaveNet逐点生成高保真音频，流模型如WaveGlow通过可逆变换实现并行合成，而生成对抗网络（GAN）为波形生成提供了另一种强大方法。GAN采用生成器和判别器网络之间的对抗训练过程，在生成逼真（听觉上）数据方面非常有效。将这种方法应用于声码器，已使得模型在计算效率和音频质量之间取得了出色的平衡。

### GAN用于声码的工作方式

在声码背景下，GAN框架按以下方式运行：

1. **生成器（G）：** 接收中间声学表示（通常是梅尔频谱图）作为输入，并尝试生成对应的原始音频波形。生成器的结构通常设计用于高效上采样，常使用转置卷积。
2. **判别器（D）：** 接收原始音频波形（真实或由G生成）作为输入，并尝试将其分类为真实或伪造。判别器学习识别合成音频中存在的不明显特征和潜在伪影。

训练包含一场对抗博弈：生成器通过生成日益逼真的波形来试图欺骗判别器，而判别器则提高其区分真实音频和生成音频的能力。这种动态促使生成器生成与真实录音在感知上无法区分的波形。

从数学角度来看，这常被表述为具有价值函数 $V(G, D)$ 的最小-最大博弈。例如，使用标准GAN损失函数 (loss function)：


$$
\min_{G} \max_{D} V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{x})} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_{z}(\mathbf{z})} [\log(1 - D(G(\mathbf{z})))]
$$


这里，$\mathbf{x}$ 表示真实音频数据，$\mathbf{z}$ 表示生成器 $G$ 的输入条件（梅尔频谱图）。在实际中，常使用诸如最小二乘GAN (LSGAN) 损失的变体，以实现更稳定的训练。

### MelGAN: 高效对抗性波形生成

MelGAN是首批成功的基于GAN的声码器之一，展现出比自回归 (autoregressive)模型显著更快的推理 (inference)速度，同时保持了良好的音频质量。

#### 结构

- **生成器：** MelGAN的生成器通常是全卷积结构，使用转置卷积在时间维度上对输入梅尔频谱图进行上采样，直到匹配目标音频波形分辨率。残差块常被置于上采样层内。其非自回归特性使得推理过程的计算效率高。
- **判别器：** MelGAN的一个重要改进是多尺度判别器的应用。MelGAN没有使用单一判别器来分析全分辨率音频，而是采用多个判别器，处理输入波形的不同下采样版本（例如，原始、2倍下采样、4倍下采样）。每个判别器都有类似的卷积结构，但处理的是不同分辨率的波形。这促使生成器生成在不同时间尺度上都逼真的音频，捕获细致细节和整体结构。

> MelGAN的简化结构，展示了生成器从梅尔频谱图生成波形的过程，以及多尺度判别器在不同分辨率下评估其逼真程度。

#### 损失函数 (loss function)

MelGAN的训练目标结合了对抗损失（鼓励生成逼真的输出）和特征匹配损失。特征匹配损失通过惩罚判别器中真实样本和生成样本的中间特征图激活之间的差异来帮助稳定训练。这为生成器提供了感知上的指导，而不仅仅是欺骗判别器的最终分类层。


$$
L_G = \sum_{k=1}^K \mathbb{E}_{(\mathbf{x}, \mathbf{z})} [ || D_k(G(\mathbf{z})) - D_k(\mathbf{x}) ||_1 ] \quad \text{(特征匹配损失)}
$$


$$
L_{Adv}(G, D) = \sum_{k=1}^K \mathbb{E}_{\mathbf{z}} [ (D_k(G(\mathbf{z})) - 1)^2 ] \quad \text{(生成器对抗损失 - LSGAN变体)}
$$


$$
L_{Adv}(D, G) = \sum_{k=1}^K \mathbb{E}_{\mathbf{x}} [ (D_k(\mathbf{x}) - 1)^2 ] + \mathbb{E}_{\mathbf{z}} [ (D_k(G(\mathbf{z})))^2 ] \quad \text{(判别器对抗损失 - LSGAN变体)}
$$


总生成器损失是 $L_G$ 和 $L_{Adv}(G, D)$ 的加权和。

MelGAN提供了比自回归模型显著的速度提升，使其适用于近实时合成，尽管与最佳自回归或流模型相比，输出质量有时可能包含轻微伪影。

### HiFi-GAN: 高保真生成对抗网络 (GAN)

HiFi-GAN在MelGAN的成功基础上发展，目标是实现更高音频保真度和改进的感知质量，同时保持计算效率。它对生成器和判别器结构都进行了改进。

#### 结构改进

- **生成器（多感受野融合 - MRF）：** HiFi-GAN的生成器采用一个名为多感受野融合（MRF）的模块。在生成器主结构内部（该结构仍使用转置卷积进行上采样），MRF模块使用多个并行残差块处理中间特征。每个残差块使用具有不同扩张率和核大小的空洞卷积。这些并行块的输出随后被融合（通常是求和）。这种设计使得生成器能够同时捕获不同时间分辨率的音频模式，提高其建模复杂波形结构和长程依赖关系的能力，而不会明显增加计算成本。
- **判别器（多周期判别器 - MPD + 多尺度判别器 - MSD）：** HiFi-GAN结合使用了两种类型的判别器：

  - **多周期判别器（MPD）：** 这是一个重要改进。MPD包含多个子判别器，每个子判别器在不同的周期间隔检查输入波形。例如，一个子判别器可能查看样本 (0, p, 2p, ...)，另一个查看 (0, q, 2q, ...)，其中 p 和 q 是不同的周期长度（例如，2, 3, 5, 7）。为此，输入波形根据周期跳过样本进行重塑，然后送入2D卷积结构。这种结构在检测GAN有时产生的、标准卷积判别器经常遗漏的微小周期性伪影方面特别有效。
  - **多尺度判别器（MSD）：** 类似于MelGAN的判别器，HiFi-GAN也包含一个MSD，通过平均池化操作在不同音频分辨率上运行，以确保在各种时间尺度上的逼真度。

> HiFi-GAN结构概览，突出了生成器中的多感受野融合 (MRF) 以及多周期 (MPD) 和多尺度 (MSD) 判别器的结合。

#### 表现

生成器中的MRF与MPD/MSD判别器的结合，助力HiFi-GAN在基于GAN的声码器中取得先进音频质量，通常可与自回归 (autoregressive)模型媲美，但推理 (inference)速度明显优越。它有效减少了早期GAN声码器中有时听到的伪影，生成清晰自然的语音。

### 训练考量

训练GAN很困难：

- **稳定性：** GAN训练对超参数 (parameter) (hyperparameter)、初始化和优化器选择非常敏感。常采用判别器中的谱范数归一化 (normalization)或使用梯度惩罚等技术。
- **模式崩溃：** 生成器可能只学习生成有限种类的声音，未能涵盖训练数据的完整多样性。MelGAN和HiFi-GAN中的多组件判别器有助于缓解这一问题。
- **评估：** 评估GAN很困难。尽管存在客观指标，但通过听力测试（如平均主观评价得分 - MOS）进行的感知评估对于衡量生成音频的真实质量和自然度仍然必不可少。

### 总结

基于GAN的声码器，如MelGAN和HiFi-GAN，代表着神经网络 (neural network)波形合成方面的重要进步。通过将对抗训练与精心设计的生成器和判别器结构相结合（运用多尺度分析、感受野融合和周期模式检测），它们实现了高保真音频生成，同时具有出色的计算效率。它们的非自回归 (autoregressive)特性使得它们特别适合需要低延迟语音合成的应用，构成了许多现代TTS流程的根本。尽管训练需要细心，但所得模型提供了令人满意的速度和质量结合。

## 参考资料

- [Generative Adversarial Networks](https://doi.org/10.48550/arXiv.1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: Advances in Neural Information Processing Systems; Volume: 27; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了生成对抗网络（GANs）的创始概念及其对抗训练框架。
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646) — Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)
  Journal: Advances in Neural Information Processing Systems; Volume: 33; DOI: [10.48550/arXiv.2010.05646](https://doi.org/10.48550/arXiv.2010.05646)
  详细介绍了HiFi-GAN，通过结合多感受野融合生成器和多周期与多尺度判别器，提升了GAN声码器的音频质量和生成速度。

---

[上一节](03-%E5%9F%BA%E4%BA%8E%E6%B5%81%E7%9A%84%E5%A3%B0%E7%A0%81%E5%99%A8%20%28WaveGlow%2C%20FloWaveNet%29.md) · [下一节](05-%E7%94%A8%E4%BA%8E%E5%A3%B0%E7%A0%81%E5%99%A8%E7%9A%84%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B.md)
