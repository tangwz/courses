---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-text-to-speech-synthesis"
lesson: "gan-tts"
sourceId: 3176
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis/gan-tts"
title: "生成对抗网络（GANs）在文本到语音中的应用"
description: "将生成对抗网络应用于声学建模或直接生成波形。"
order: 4
plots: []
sourceHash: "30bf8d9d4b71f8efa3472c9b2352c18bd3f81480d03c910bdebeb4a9741a9477"
sourceCorrections: []
---

生成对抗网络 (GAN)（GANs）代表了一类功能强大的生成模型，在文本到语音合成方面，它提供了一种不同于Tacotron等序列到序列模型和FastSpeech等并行方法的独特途径。GANs最初是为图像生成而开发的，后来也成功应用于文本到语音技术中，在高效生成高质量输出方面尤其具有独特优势。

### GAN的思路：对抗性博弈

其核心是，GAN涉及两个神经网络 (neural network)之间的双人博弈：

1. **生成器（G）：** 这个网络试图生成合成数据（在我们的例子中是语音表示，如梅尔频谱图，甚至是原始音频），使其看起来与真实数据无异。它接收一些输入，通常是文本嵌入 (embedding)以及可能的噪声或韵律信息，并输出合成的语音表示。
2. **判别器（D）：** 这个网络充当评判者的角色。它同时接收真实的语音表示（来自训练数据集）和虚假的语音表示（来自生成器），并试图正确地将它们分类——将真实的标记 (token)为“真实”，将虚假的标记为“虚假”。

训练过程使这两个网络相互对抗：

- 生成器学习生成越来越真实的数据，以“欺骗”判别器。
- 判别器学习更好地区分真实数据和虚假数据。

这个对抗过程持续进行，直到理想情况是生成器生成的数据如此真实，以至于判别器只能随机猜测（达到约50%的准确率）。这种极小极大博弈的目标函数可以表示为：


$$
\min_{G} \max_{D} V(D, G) = \mathbb{E}_{x \sim p_{\text{数据}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))]
$$


其中 $x$ 是真实数据，$z$ 是输入噪声/条件信息，$G(z)$ 是生成数据，$D(x)$ 是判别器认为 $x$ 是真实数据的概率，而 $p_{\text{数据}}$ 和 $p_z$ 分别是真实数据和输入噪声的分布。

### 将GANs应用于文本到语音

在文本到语音方面，GANs主要有两种用法：

1. **声学特征生成：** 与FastSpeech等模型类似，GANs可以训练用于以非自回归 (autoregressive)方式直接从文本表示中生成声学特征（如梅尔频谱图）。

   - **生成器**接收处理后的文本输入（例如，与时长预测对齐 (alignment)的音素嵌入 (embedding)），并合成一个梅尔频谱图。
   - **判别器**训练用于区分数据集中真实的梅尔频谱图和由G生成的合成频谱图。

   这种方法采用对抗损失来隐式捕获自然语音频谱的复杂分布，通常带来感知上令人信服的结果。训练稳定性可以通过加入额外的损失项来增强，例如特征匹配损失（比较判别器中真实和虚假输入的中间激活）或重建损失（例如生成频谱图与真实频谱图之间的L1或L2距离）。

   > GAN应用于文本到语音中梅尔频谱图生成的基本结构。生成器从文本相关输入中生成频谱图，而判别器试图将它们与真实频谱图区分开。损失信号指导两个网络的更新。
2. **波形生成（声码器）：** GANs也可以训练用于从声学特征（如梅尔频谱图）中直接合成原始音频波形。MelGAN和HiFi-GAN等模型属于这一类别。尽管这在技术上是声码器阶段的一部分（我们将在下一章详细介绍），但值得在此提及的是，对抗训练原理对于生成高保真音频非常有效。生成器充当声码器，判别器对波形片段进行操作以评估其真实性。

### GANs在文本到语音中的优势

- **并行生成：** 与其他非自回归 (autoregressive)模型一样，基于GAN的声学特征生成器可以并行生成整个输出序列，与自回归模型相比，推理 (inference)速度大大加快。
- **高感知质量：** 对抗训练通常擅长捕获语音的感知特性，可能带来更自然的合成语音，尤其是在用于声码时。
- **隐式建模：** GANs通过对抗过程隐式学习数据分布，避免了显式密度建模的需要，这对于语音等复杂数据可能具有挑战性。

### 挑战与考量

- **训练稳定性：** 训练GANs出了名的困难。实现收敛需要仔细的超参数 (parameter) (hyperparameter)调整、损失函数 (loss function)的选择以及网络架构设计。诸如模式崩溃（指生成器只产生有限种类的输出）等问题可能发生。
- **评估：** 评估GANs可能很棘手。频谱图上的L1/L2损失等标准指标可能与感知质量关联不佳。主观听觉测试（如平均意见得分，MOS）通常是必要的但成本高昂。判别器损失本身并非总是生成质量的可靠指标。
- **架构设计：** 专门为语音任务设计有效的生成器和判别器架构是一个活跃的研究方向。在声码GANs（如HiFi-GAN）中，常见有多尺度判别器，它们在不同输入分辨率上运行，以捕获精细的波形细节和更宏观的结构。

GANs为构建文本到语音系统组件提供了一个有吸引力的替代方案。尽管它们带来了独特的训练挑战，但它们高效生成输出并可能达到高感知质量的能力，使其成为高级文本到语音工具包中的一项重要技术，尤其适用于非自回归 (autoregressive)声学建模和高保真神经声码器。

## 参考资料

- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) — Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (2014)
  Journal: arXiv preprint arXiv:1406.2661; DOI: [10.48550/arXiv.1406.2661](https://doi.org/10.48550/arXiv.1406.2661)
  介绍了生成对抗网络的基本概念和框架，包括其最小-最大目标函数。
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://neurips.cc/virtual/2020/public/2dbe1267b1897486e495207b2756df6a.html) — Jungil Kong, Jaehyeon Kim, Jaekyoung Bae (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33
  提出了一种高效高保真的基于GAN的声码器，成为从声学特征生成原始音频的标准方法。
- [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://proceedings.neurips.cc/paper_files/paper/2019/file/9f381f7055743818dd10b42f638f28f0-Paper.pdf) — Kundan Kumar, Rithesh Kumar, Alain de Boissiere, Lucas Gestin, Flavien Marion, Rony Rodriguez, Jean-Baptiste Roux, Sampath Singam, Aaron van den Oord, S. Zeghidour, Rafael Soricut (2019)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 32; Pages: 16909–16919; DOI: [10.55916/HV8C-T992](https://doi.org/10.55916/HV8C-T992)
  介绍了一种早期且有影响力的基于GAN的模型，用于根据梅尔谱图生成高质量语音波形。
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://proceedings.mlr.press/v139/kim21h/kim21h.pdf) — Jaehyeon Kim, Jungil Kong, Inwan Yoon (2021)
  Journal: International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 139; DOI: [10.48550/PMLR.v139.kim21h](https://doi.org/10.48550/PMLR.v139.kim21h)
  提出了一种端到端文本到语音模型，利用条件变分自编码器和对抗训练进行声学特征和波形生成，实现了高质量和高效率。
