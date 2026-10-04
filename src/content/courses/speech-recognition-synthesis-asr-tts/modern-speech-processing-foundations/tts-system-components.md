---
course: "speech-recognition-synthesis-asr-tts"
chapter: "modern-speech-processing-foundations"
lesson: "tts-system-components"
sourceId: 3113
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-1-modern-speech-processing-foundations/tts-system-components"
title: "文本转语音（TTS）系统的组成部分"
description: "详述各阶段：文本规范化、语言学分析、时长建模、声学特征生成、声码。"
order: 5
plots: []
sourceHash: "fbf628eef0259c4fcb9e43e0735512381bf29e7dd06e5784bd183630dde89874"
sourceCorrections: []
---

自动语音识别（ASR）侧重于将语音转录成文字，而文本转语音（TTS）合成则承担了互补的任务：从输入文本生成可听的语音。虽然现代端到端神经网络 (neural network)系统通常直接学习映射关系，但理解传统的逻辑组成部分有助于更好地分析和设计复杂的TTS流程。这些组成部分顺序运作，将文本转换为听起来自然的波形。

让我们剖析参数 (parameter)化或拼接式TTS系统中涉及的典型阶段，同时考虑到神经网络方法可能会整合或重新构想这些步骤。

### 文本处理（前端）

从文本到语音的过程始于理解和规范化输入文本。此前端处理确保文本清晰无歧义，并为后续阶段适当地构建其结构。

#### 1. 文本规范化

原始文本通常包含非标准词语，如数字、缩写、符号和标点符号，这些需要转换为其完全拼写出来、可读的形式。这个过程，称为文本规范化（TN）或非标准词（NSW）处理，对于可懂度很重要。

示例：

- `$12.50` -> “十二美元五十美分”
- `Dr. Smith` -> “史密斯医生”
- `St. Louis` -> “圣路易斯”
- `10 Downing St.` -> “唐宁街十号”
- `1998` -> “一千九百九十八”

文本规范化可能很复杂，涉及正则表达式、有限状态转换器（FSTs），或越来越多地使用机器学习 (machine learning)模型。它高度依赖语言，需要谨慎处理歧义（例如，“St.”可以表示“Saint”或“Street”）。文本规范化中的错误会直接影响合成语音。

#### 2. 语言学分析

文本规范化后，会进行语言学分析，以提取与发音和韵律（语调、节奏、重音）相关联的特征。

- **字素到音素（G2P）转换：** 这将书面文本（字素）转换为语音表示（音素）。例如，词语“synthesis”可能会被转换为音素序列，比如`/s ɪ n θ ə s ɪ s/`（使用ARPABET记号）。这可以通过以下方式实现：
  - **发音词典（词库）：** 将词语映射到其语音转录的大型查找表。
  - **基于规则的系统：** 针对发音模式的手工规则。
  - **机器学习模型：** 在词典数据上训练的统计或神经网络 (neural network)模型，用于预测已知词和未知词（词汇表 (vocabulary)外，OOV）的发音。
- **词性（POS）标注：** 识别每个词语的语法作用（名词、动词、形容词等）有助于解决发音歧义（例如，“reCORD”（动词）与“REcord”（名词））并为韵律预测提供信息。
- **韵律预测：** 虽然基本系统可能使用启发式方法，但进阶系统明确预测韵律特征。这可以包含：
  - **语段停顿预测：** 确定停顿应该发生的位置。
  - **重音/强调预测：** 识别应该强调的词语或音节。
  - **语调轮廓生成：** 预测基频（$F_0$）轮廓，这在很大程度上决定了语音感知到的旋律。这通常涉及分析句子结构（例如，疑问句与陈述句）。

前端的输出通常是富含语言学和韵律信息的音素序列。

### 声学特征生成（后端/合成器）

后端从前端接收处理过的语言学特征，并生成中间声学表示。

#### 3. 时长建模

每个音素（或其他语言单元）没有固定的时长。其长度根据语音上下文 (context)、词语或短语内的位置、语速和强调而显著变化。时长模型预测输入序列中每个单元的时长（通常以毫秒或帧为单位）。准确的时长建模对于实现自然的语音节奏很重要。传统系统常使用HMMs或决策树，而现代神经网络 (neural network)系统通常在其架构中包含时长预测器（例如在FastSpeech中所示或集成到注意力机制 (attention mechanism)中）。

#### 4. 声学特征预测

这是核心合成步骤，系统将语言学特征（音素、时长、韵律目标）映射到声学特征序列。这些特征旨在捕捉目标语音的频谱包络和激励特性，但以一种压缩的中间格式呈现。

- **特征类型：** 常见的选择是梅尔频谱图，它表示音频的功率谱，映射到人类频率感知的非线性梅尔尺度上。其他特征，如倒谱系数或从HMM状态派生的语言-声学特征也曾被使用。
- **建模方法：**
  - **经典参数 (parameter)化：** 基于HMM的系统根据在上下文相关状态上训练的统计模型生成参数（如频谱特征和$F_0$）。
  - **拼接式：** 从大型数据库中选择并拼接预先录制的语音单元（双音素、音素）。尽管对于特定音色可以达到高质量，但它存在可听见的断裂，且缺乏灵活性。
  - **神经网络序列到序列：** 像Tacotron或Transformer TTS这样的模型，使用带注意力的编码器-解码器架构，直接从输入语言学特征（通常是字符或音素）学习到声学特征（通常是梅尔频谱图）的复杂非线性映射。这些模型隐式处理时长和上下文建模。非自回归 (autoregressive)模型如FastSpeech使用显式时长预测器结合前馈变换器以实现更快的生成。

这一阶段的输出是逐帧的声学特征向量 (vector)序列。

### 波形生成（声码器）

合成器生成的声学特征尚不是音频波形。最后一步使用声码器（语音编码器/解码器）将这些特征转换为可听见的声压波。

#### 5. 声码

- **传统声码器：** 信号处理技术如源-滤波器模型（例如STRAIGHT）或基于短时傅里叶变换（STFT）幅度的重叠相加方法（例如Griffin-Lim算法）很常见。这些通常产生可懂但有些人工化或“嗡嗡声”的语音，尤其是在较低特征分辨率下。
- **神经网络 (neural network)声码器：** 现代TTS很大程度上依赖于神经网络声码器，它们是深度生成模型，经训练可根据声学特征合成高保真波形。示例有：
  - **自回归 (autoregressive)模型：** WaveNet、WaveRNN逐个音频样本生成，达到高质量但推理 (inference)速度通常较慢。
  - **流式模型：** WaveGlow、FloWaveNet使用归一化 (normalization)流进行并行生成。
  - **基于GAN的模型：** MelGAN、HiFi-GAN、Parallel WaveGAN使用生成对抗网络 (GAN)实现高效、高质量的波形合成。
  - **扩散模型：** 近期方法将扩散概率模型应用于波形生成。

这些进阶声码器（在第5章中会更详细地讨论）是过去十年TTS自然度大幅提升的重要原因。

### 系统概述

这些组成部分之间的紧密关联可以可视化为一个流程：

> 文本转语音合成的典型流程，展示了从输入文本到前端处理、后端合成（时长和声学特征预测），以及最终通过声码器生成波形的过程。现代端到端系统可能会整合这些阶段中的多个。

虽然这些组成部分顺序呈现，但它们紧密关联。早期阶段的错误或局限（例如不正确的音素转换或不自然的持续时间预测）显著影响最终输出质量。当我们在第4章讨论进阶TTS模型时，我们会看到端到端神经网络 (neural network)如何旨在学习从文本到声学特征的完整映射，有时甚至直接到波形，这可能会简化流程，但通常需要更大的数据集和细致的训练策略。理解这些基本构造仍然对于诊断问题、定制系统以及认识生成类人语音所涉及的复杂性来说，是必不可少的。

## 参考资料

- [Speech and Language Processing (3rd Edition)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这是广泛使用的教材的正在进行的草稿。它涵盖了语音和语言处理的各个方面，包括文本到语音合成基础知识、文本规范化、语言分析以及TTS系统演变的综合章节。
- [The HMM-based Speech Synthesis System (HTS)](https://www.sp.nitech.ac.jp/~tokuda/publication/interspeech02.pdf) — Keiichi Tokuda, Takashi Zen, Yoshihiko Nankaku (2002)
  Journal: 7th International Conference on Spoken Language Processing, ICSLP2002 - INTERSPEECH 2002; Publisher: ISCA
  本文介绍了基于HMM的语音合成基础框架，该框架多年来一直是主要的参数化TTS方法。它详细描述了频谱特征和基频（F0）的统计建模。
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884) — Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu (2018)
  Journal: ICASSP; Pages: 4779-4783; DOI: [10.48550/arXiv.1712.05884](https://doi.org/10.48550/arXiv.1712.05884)
  本文介绍了Tacotron 2，这是一个重要的端到端神经文本到语音系统，通过预测梅尔频谱图生成自然语音，然后由WaveNet声码器将其转换为音频。它展示了迈向更简单、更高质量的神经TTS的重大一步。
