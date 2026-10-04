---
course: "speech-recognition-synthesis-asr-tts"
chapter: "modern-speech-processing-foundations"
lesson: "asr-system-components"
sourceId: 3110
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-1-modern-speech-processing-foundations/asr-system-components"
title: "ASR 系统组成部分"
description: "分析典型部分：声学模型、发音词典、语言模型。"
order: 4
plots: []
sourceHash: "ccce9b1b09c32ee1febfecf54410e7ecaec3d033f1d1d8d8e7b01b936c34f82f"
sourceCorrections: []
---

自动语音识别 (ASR) 系统的架构为理解和开发现代语音处理方案提供了基础框架。尽管现代端到端系统有时会模糊界限，但将其拆分为不同部分对于分析和设计仍有价值。传统上，ASR 系统结合多种知识来源，将声学信号转换为词语序列。

> 传统 ASR 流程中各部分的示意图。解码器结合来自声学模型、发音词典和语言模型的信息，以找到最有可能的文本转录。

### 声学模型 (AM)

声学模型负责连接声学方面（音频特征）和语言学方面（基本声音单元）之间的距离。

- **输入：** 从输入音频信号中提取的声学特征序列。如“高级音频特征提取”部分所述，这些可以是梅尔频率倒谱系数 (MFCCs)、滤波器组能量，甚至是神经网络 (neural network)初始层学习到的特征。
- **输出：** 基本语言单元的概率分布。历史上，这些单元通常是隐马尔可夫模型 (HMM) 系统中使用的上下文 (context)相关音素或“音素状态”。在现代系统，尤其是端到端架构中，输出单元可能是音素、字素（字符）、子词 (subword)单元（如字节对编码单元），甚至直接是词语。
- **功能：** 声学模型学习声学信号片段与这些语言单元之间的统计关系。例如，给定一小段音频特征，声学模型可能会输出概率，表明该片段对应音素 /s/、/t/、/a/ 等的可能性有多大。深度神经网络，包括卷积神经网络 (CNN) (CNNs)、循环神经网络 (RNN)（如 LSTM 或 GRU）和 Transformer，已成为构建高性能声学模型的标准方法，大部分取代了旧的高斯混合模型 (GMM)-HMM 方法。这些网络擅长建模语音中存在的复杂时间依赖性和声学变异。

声学模型的主要任务可以看作是估计 $P(A|U)$，即在给定语言单元序列 $U$ 的情况下，观测到声学特征序列 $A$ 的概率。

### 发音词典 (Dictionary)

发音词典提供词语与其构成声音单元（通常是音素）之间的映射。

- **功能：** 它定义了系统中每个词语的标准发音。例如，“cat”这个词语可能被映射到音素序列 /k/ /æ/ /t/。它允许系统根据声学模型在解码过程中提出的音素序列来假设词语序列。
- **挑战：** 创建和维护词典可能需要大量工作。它需要处理发音变体（例如，“tomato”的不同发音）、同音词（“to”、“too”、“two”）以及词汇外 (OOV) 词语。对于 OOV 词语，通常采用字素到音素 (G2P) 转换模型等技术来即时预测发音。
- **现代背景：** 在纯粹基于字素的端到端系统（直接输出字符）中，显式词典可能被绕过，尽管模型必须在内部学习发音规则。子词 (subword)单元也减少了对固定词典的依赖，能更好地处理 OOV 词语和形态丰富的语言。然而，显式词典在许多混合系统中仍然普遍存在，并且在可用时可以提高准确性。

### 语言模型 (LM)

语言模型捕获语言的统计模式，本质上是估计给定词语序列出现的可能性。

- **功能：** 它提供目标语言中有效且可能出现的词语序列的先验知识。例如，它会给“recognize speech”分配比“wreck a nice beach”更高的概率，即使它们在声学上听起来相似。这有助于解码器解决由嘈杂音频、不清晰发音或声学模型局限性引起的歧义。
- **类型：** 传统语言模型常使用 n-gram，它基于前面的 $n-1$ 个词语来估计一个词语的概率。虽然计算效率高，但在捕获长程依赖方面存在局限性。现代 ASR 系统越来越多地采用神经网络 (neural network)语言模型（RNN-LM、Transformer-LM），它们可以建模更丰富的上下文 (context)信息，并且通常提供更优异的性能，尽管计算成本更高。这些高级语言模型将在第 3 章中进一步研究。
- **结合：** 假设词语序列 $W$ 的语言模型得分 $P(W)$ 在解码过程中与声学模型得分 $P(A|W)$（通过声学模型和词典得出）结合。

### 解码器 (搜索算法)

解码器是协调声学模型、词典和语言模型之间相互作用的引擎，它在给定输入声学特征 $A$ 的情况下，找到最可能的词语序列 $W^*$。目标是解决语音识别的基本方程，该方程通常使用贝叶斯定理公式化：


$$
W^* = \underset{W}{\operatorname{argmax}} P(W|A) = \underset{W}{\operatorname{argmax}} \frac{P(A|W) P(W)}{P(A)}
$$


由于给定输入音频的 $P(A)$ 是常数，搜索简化为：


$$
W^* = \underset{W}{\operatorname{argmax}} P(A|W) P(W)
$$


实践中，通常引入语言模型权重 (weight) $\lambda$ 和插入惩罚来平衡各项贡献：


$$
W^* = \underset{W}{\operatorname{argmax}} P(A|W) P(W)^\lambda
$$


找到最优的 $W^*$ 涉及在可能的词语序列的宽广空间中搜索。高效的搜索算法，如维特比解码（用于基于 HMM 的系统）和各种形式的束搜索被采用。在许多混合系统中，声学模型、词典和语言模型被编译成一个统一的搜索图，通常使用加权有限状态转换器 (WFSTs) 表示，这实现了高效解码。

### 向端到端系统演变

需要注意的是，高级端到端模型（如第 2 章讨论的 CTC、RNN-T、基于注意力的模型）旨在通过单一、统一的神经网络 (neural network)，将输入声学特征直接映射到输出转录（字符、词语或子词 (subword)）。这些模型在其网络参数 (parameter)中隐式学习声学建模、发音甚至语言建模的方面。尽管这简化了流程，但结合外部语言模型（通过第 3 章讨论的浅层或深层融合等技术）通常仍有益于实现最先进的性能，特别是在需要广泛领域知识的任务上。因此，即使在使用这些新架构时，了解传统部分的作用仍然相关。

## 参考资料

- [Speech and Language Processing: An Introduction to Computational Linguistics and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  这本综合性教材涵盖了语音和语言处理的原理与实践，详细介绍了ASR系统组件、特征提取、声学和语言建模以及解码算法。它也包含了现代神经网络技术。
- [Deep Neural Networks for Acoustic Modelling in Speech Recognition: The Shared Views of Four Research Groups](https://ieeexplore.ieee.org/document/6296526) — Geoffrey Hinton, Li Deng, Dong Yu, George Dahl, Adam Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara Sainath, and Brian Kingsbury (2012)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 29; Pages: 82-97; DOI: [10.1109/MSP.2012.2205597](https://doi.org/10.1109/MSP.2012.2205597)
  本文介绍了用于声学建模的深度神经网络，显著提升了语音识别性能，超越了高斯混合模型，并影响了现代ASR系统的设计。
- [Speech Recognition with Deep Recurrent Neural Networks](https://ieeexplore.ieee.org/document/6638947) — Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton (2013)
  Journal: IEEE Conference Publication; Publisher: IEEE; Pages: 6645-6649; DOI: [10.1109/ICASSP.2013.6638947](https://doi.org/10.1109/ICASSP.2013.6638947)
  介绍了用于端到端语音识别的循环神经网络换能器（RNN-T），展示了一种无需显式对齐即可将声学特征直接映射到字符序列的架构。
