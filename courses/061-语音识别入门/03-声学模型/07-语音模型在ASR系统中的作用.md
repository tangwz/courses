# 语音模型在ASR系统中的作用

来源：[原文](https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/role-of-acoustic-model-in-asr)

[返回章节目录](README.md) · [返回课程目录](../README.md)

语音识别系统依赖于将声波转换为特征向量 (vector)序列。高斯混合模型 (GMM) 和隐马尔可夫模型 (HMM) 通常用于表示语音的独特特点。语音模型整合这些概念，在完整的语音识别系统中扮演着主要角色。

语音模型是经过处理的音频信号与语言层面之间的桥梁。它的任务不是理解单词或句子，而是听取一小段音频，并判断它最像哪个基本语音单元，即音素。

可以将语音模型看作一位高度专业的语音学家。如果您给它一个25毫秒音频片段的特征向量，它无法告诉您说话者说的是“cat”还是“car”。然而，它可以告诉您该声音是/k/、/æ/或/t/的概率。它为每个音频帧执行此计算，从而生成一个连续的音素概率流。

### 输入、处理与输出

为了理解语音模型的功能，明确它的输入和输出内容非常有帮助。

- **输入：** 模型接收在信号处理阶段生成的特征向量 (vector)序列（例如，MFCC）。每个向量都是原始音频中一个非常短的片段的数值概括。
- **处理：** 在模型内部，无论是传统的GMM-HMM系统还是现代神经网络 (neural network)，都会发生统计比较。模型经过数千小时的带标注语音训练，因此它已经习得了每种语言中每个音素的典型特征向量模式。它将传入的特征向量与这些习得的模式进行比较。
- **输出：** 模型输出一组概率。对于每个音频帧，它会生成该帧对应于每个可能音素的可能性。这就是我们之前提到的概率，通常写为$P(\text{音频特征} | \text{音素})$。这个概率流随后被传递给流水线中的下一个主要组成部分：解码器。

下图展示了语音模型在整个ASR系统中的位置。它接收特征提取的输出，并为解码器提供重要输入。

> ASR流水线，展示了语音模型的核心地位。它将特征向量转换为音素概率，解码器会结合语言模型的输入来使用这些概率。

### 提供证据，而非答案

初学者常有的一个误区是认为语音模型的功能超出了实际。语音模型只是证据来源之一，其输出本身就带有模糊性。

请考虑一个典型例子，以下两个短语听起来非常相似：

- “recognize speech”
- “wreck a nice beach”

这两个短语的音素序列几乎相同。语音模型如果仅分析声音，很可能会给这两个音素序列都分配较高的概率分数。它不具备语法、语境或哪个短语在对话中更可能被说出的判断能力。它只会报告：“根据音频信号，这些是合理的声音序列。”

这正是ASR系统不仅需要语音模型的原因。它产生的模糊性必须由另一个组件来消除。

### 与语言模型配合使用

语音模型的输出提供了转录所需信息的前半部分。后半部分来自**语言模型**，我们将在下一章中介绍。

语音模型回答的是“声音与音频特征的匹配程度如何？”，而语言模型回答的是“这个词序列在这种语言中出现的可能性有多大？”

最后一个组成部分，即**解码器**，负责结合这两种信息来源。它寻找一个词序列，该序列既具有较高的语音得分（声音与音频匹配良好），又具有较高的语言模型得分（这些词构成一个合理的句子）。通过权衡两个模型提供的信息，解码器能够正确地选择“recognize speech”而非“wreck a nice beach”，因为前者是更常见且语法上更合理的短语。

总而言之，语音模型是负责让ASR系统立足于声音物理特性的组成部分。它将音频文件中的抽象数值特征转换为有意义的音素概率，为解码器提供开始搜索正确词语所需的重要依据。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  全面介绍ASR原理的教材，内容涵盖特征提取、声学模型（HMM/GMM和深度学习）、语言模型和解码策略。
- [Deep Neural Networks for Acoustic Modeling in Speech Recognition](https://doi.org/10.1109/MSP.2012.2205597) — Geoffrey Hinton, Li Deng, Dong Yu, George Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N. Sainath, and Brian Kingsbury (2012)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 29; Pages: 82-97; DOI: [10.1109/MSP.2012.2205597](https://doi.org/10.1109/MSP.2012.2205597)
  一篇重要论文，详细阐述了深度神经网络在声学模型中的应用和效率，是ASR领域的一大进步。
- [An Introduction to Hidden Markov Models](https://doi.org/10.1109/MASSP.1986.1165342) — Lawrence R. Rabiner (1986)
  Journal: IEEE ASSP Magazine; Publisher: IEEE; Volume: 3; Pages: 4-16; DOI: [10.1109/MASSP.1986.1165342](https://doi.org/10.1109/MASSP.1986.1165342)
  基础性论文，清晰阐述了隐马尔可夫模型，该模型是传统语音识别系统中声学建模的基础。

---

[上一节](06-%E5%9F%BA%E4%BA%8E%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%A3%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B%E4%BB%8B%E7%BB%8D.md) · [下一节](../04-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/01-%E4%BB%80%E4%B9%88%E6%98%AF%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%EF%BC%9F.md)
