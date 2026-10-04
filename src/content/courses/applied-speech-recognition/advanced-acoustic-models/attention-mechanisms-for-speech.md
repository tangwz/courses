---
course: "applied-speech-recognition"
chapter: "advanced-acoustic-models"
lesson: "attention-mechanisms-for-speech"
sourceId: 7135
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/attention-mechanisms-for-speech"
title: "用于语音识别的注意力机制"
description: "了解注意力机制如何使模型在转录过程中集中于音频序列的相关部分。"
order: 1
plots: ["plots/7135-0.json"]
sourceHash: "a26b08036fc38c15eaf0ada5848cdca96e946378625a58a88cc1652b4c3aca2c"
sourceCorrections: []
---

连接主义时间分类（CTC）提供了一种有效方法来训练声学模型，而无需预先对齐 (alignment)的数据集。然而，CTC基于一个强假设：模型在每个时间步的输出条件独立于所有其他输出。这意味着在时间 $t$ 预测“c”的概率不取决于模型在时间 $t-1$ 预测了“a”。这一局限阻止了模型学习输出序列中字符间的语言关联性。

为解决此问题，我们可以使用**注意力机制 (attention mechanism)**。设想一个人如何转录音频。他们不会一次性处理整个声波，然后写出完整句子。相反，他们会先听，写下几个词，甚至可能会重听一小段以捕捉一个难词。注意力机制使模型能够模仿这种行为，在生成输出转录的每个部分时，选择性地关注输入音频序列的不同部分。它使模型能够权衡每个输出字符或词不同音频帧的重要性，从而有效形成动态且上下文 (context)相关的对齐。

### 基于注意力模型的结构

注意力机制 (attention mechanism)通常充当编码器和解码器之间的桥梁。

- **编码器**：这通常是一个循环神经网络 (neural network) (RNN)（如双向LSTM）或Transformer网络，用于处理整个输入音频特征序列（例如，对数梅尔谱图）。它生成一系列隐藏状态向量 (vector)，$h_1, h_2, ..., h_T$，其中每个$h_j$表示时间步$j$附近音频的信息。这些向量共同构成整个语音的高级表征。
- **解码器**：这是另一个网络（通常是LSTM），一次生成一个令牌（例如，一个字符）的输出转录。
- **注意力机制**：这是连接两者的组件。在解码过程的每一步，注意力机制都会计算一组“注意力权重 (weight)”，并使用它们来计算一个上下文 (context)向量。这个向量指导解码器对下一个输出令牌的决策。

### 注意力计算

为每个输出步骤计算上下文 (context)向量 (vector)的过程可以分为三部分。假设解码器即将生成转录的第$i$个字符。它使用其先前的隐藏状态，$s_{i-1}$，来查询编码器的输出向量，$h_1, ..., h_T$。

#### 1. 计算注意力分数

首先，模型需要一种方式来评估每个输入帧$h_j$与当前正在生成的输出（由解码器状态$s_{i-1}$表示）的对齐 (alignment)程度。这通过一个评分函数完成。一种常见的方法，称为加性注意力，使用一个小型前馈神经网络 (neural network)：


$$
e_{ij} = \text{评分}(s_{i-1}, h_j)
$$


分数$e_{ij}$量化 (quantization)了第$j$个音频帧与第$i$个输出字符的相关性。分数越高，表示该帧对当前解码步骤越重要。

#### 2. 使用Softmax得出注意力权重 (weight)

原始分数$e_{ij}$本身用处不大，因为它们的尺度可能变化很大。为了将它们归一化 (normalization)为更易解释的格式，我们将其通过`softmax`函数。这会将分数转换为概率分布，称为注意力权重，表示为$\alpha_{ij}$。


$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}
$$


每个权重$\alpha_{ij}$是一个介于0和1之间的值，并且给定解码步骤$i$的所有权重之和为1（$\sum_{j=1}^{T} \alpha_{ij} = 1$）。可以将这些权重看作解码器在生成当前输出字符时应对每个特定音频帧投入的“注意力”量。

#### 3. 计算上下文向量

最后，上下文向量，$c_i$，被计算为所有编码器隐藏状态的加权和。此和中使用的权重是我们刚计算的注意力权重$\alpha_{ij}$。


$$
c_i = \sum_{j=1}^{T} \alpha_{ij} h_j
$$


这个上下文向量是输入音频的总结，专为生成第$i$个输出字符而定制。它包含了当前预测所需的最相关声学信息。解码器随后使用此上下文向量及其自身的隐藏状态，预测序列中的下一个字符。整个过程重复进行，直到解码器生成一个特殊的序列结束标记 (token)。

### 注意力权重 (weight)可视化

注意力机制 (attention mechanism)的一大优点是其可解释性。通过在热力图中绘制注意力权重$\alpha_{ij}$，我们可以清楚地看到模型在生成每个输出字符时具体关注了输入音频的哪些部分。



![词语“HELLO”的注意力对齐](plots/7135-0.json)



> 输入音频帧与词语“HELLO”的输出字符之间的对齐 (alignment)。蓝色越深表示注意力权重越高。请注意强大的对角线模式，这表明当模型从“H”生成到“O”时，其关注点在音频帧中逐步移动。

这种清晰的单调对齐在语音识别中很常见，并为调试提供了有益的工具。如果注意力模式显得混乱或不合理，这通常表明模型训练存在问题。

通过引入注意力机制，我们的ASR模型不再受CTC严格假设的限制。它们可以学习音频与文本之间柔性、数据驱动的对齐，从而显著提高转录准确性，特别是对于更长、更复杂的语音。这种机制是我们将要讨论的现代序列到序列模型的一个基本组成部分。

## 参考资料

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://doi.org/10.48550/arXiv.1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (2014)
  Journal: ICLR 2015; DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  为序列到序列模型引入了注意力机制，特别是本节讨论的加性注意力。
- [Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition](https://doi.org/10.1109/ICASSP.2016.7472098) — William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals (2016)
  Journal: 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); Publisher: IEEE; DOI: [10.1109/ICASSP.2016.7472098](https://doi.org/10.1109/ICASSP.2016.7472098)
  将基于注意力的编码器-解码器模型应用于端到端语音识别，展示了强大的性能。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 30; DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍了Transformer架构，该架构完全依赖自注意力机制，并作为现代ASR模型中常见的编码器。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  Publisher: Pearson
  一本标准教材，提供了对语音识别的广泛理解，包括关于注意力机制和序列到序列模型的详细章节。
