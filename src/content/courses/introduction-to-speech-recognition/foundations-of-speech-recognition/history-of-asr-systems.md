---
course: "introduction-to-speech-recognition"
chapter: "foundations-of-speech-recognition"
lesson: "history-of-asr-systems"
sourceId: 7081
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-1-foundations-of-speech-recognition/history-of-asr-systems"
title: "自动语音识别系统简史"
description: "回顾语音识别技术的发展历程，从早期系统到现代深度学习方法。"
order: 2
plots: []
sourceHash: "70218afe179f1950286b45bb85ba2bde3099065bd49e6f4054e6ed4a98db0357"
sourceCorrections: []
---

能够与机器对话并让其理解你的话语，这或许让人感觉是最近才出现的能力，但对这一目标的追求早在半个多世纪前就开始了。自动语音识别（ASR）的发展历程表明，它从简单的数字识别器稳步演变为驱动当今语音助手的复杂系统。理解这一演变过程有助于弄清ASR系统为何以目前的方式构建。

### 早期阶段：数字与词语的识别

20世纪50年代和60年代，语音识别的首次尝试虽有雄心，但受到严格限制。1952年，贝尔实验室开发了“Audrey”系统，这台机器能够识别从零到九的口语数字。然而，它有一个显著的局限性：只能识别其创造者的声音。十年后，即1962年，IBM演示了其“Shoebox”机器，它能理解16个英语单词和相同的数字集。

这些早期系统是基于声学模式匹配。它们分析语音信号不同频带中存在的能量，并尝试将其与预先录制的模板进行匹配。这种方法适用于：

- **孤立词语：** 你必须一次说一个词，词语之间有清晰的停顿。
- **小词汇量：** 系统只能识别少量词语。
- **特定说话人操作：** 它们必须针对特定说话人的声音进行训练。

尽管存在这些局限，这些早期项目证明了机器识别语音的可行性。

### 统计方法转向：隐马尔可夫模型

20世纪70年代是一个重要转折点。研究人员不再试图匹配整个声音模式，而是开始应用统计方法。这项工作得到了美国政府机构DARPA的大量资助，促成了隐马尔可夫模型（HMM）的采用。

隐马尔可夫模型是一种统计模型，将语音视为一系列声音。它不匹配整个单词，而是计算特定音频特征序列对应于音素（声音的基本单位）序列的概率。这是一种更灵活、更有效的方法，能够处理人类语音的可变性。隐马尔可夫模型能够模拟声音如何从一个过渡到下一个，这是识别连续、流畅语音的重要一步。

### 主流方法：GMM-HMM 系统

近二十年来，从20世纪90年代到21世纪00年代末，自动语音识别的标准方法是将隐马尔可夫模型与另一种统计工具——高斯混合模型（GMM）结合使用。

- **隐马尔可夫模型（HMM）** 处理语音的*序列*，模拟音素如何相互连接形成单词。
- **高斯混合模型（GMM）** 处理每个音素的*声音*，模拟每个不同声音的音频特征（例如MFCC，我们稍后会介绍）的分布。

这种GMM-HMM组合功能强大，足以构建出首批商业成功的自动语音识别产品，例如Dragon NaturallySpeaking。这些系统能够处理大词汇量，并且是与说话人无关的，这意味着它们无需特定训练即可为大多数用户服务。

> 语音识别技术发展的重要时期时间线。

### 深度学习 (deep learning)变革

2010年左右，该方面又发生了一次重大变化，深度学习获得了广泛应用。研究人员发现，深度神经网络 (neural network)（DNN）在学习音频特征与语音之间的复杂关联方面表现出色。

最初，深度神经网络被用于替代传统GMM-HMM系统中的GMM部分。仅此一项改变就使词错误率（WER）（衡量自动语音识别准确性的标准指标）显著降低。

最近，研究转向了**端到端**模型。这些是单一的、大型神经网络，它们能直接从音频特征学习如何将语音转录成文本，无需单独的声学、发音和语言建模组件。这种方法简化了自动语音识别流程，并将性能提升到新的水平。手机上的语音助手、智能音箱及其他设备，都由这些现代的、基于深度学习的系统提供支持。这段历史为我们接下来考察的组件奠定了基础，很多组件都源于这些早期系统。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (current draft)
  一本内容全面的教材，涵盖了语音识别的历史背景、HMM和GMM等基础算法以及现代深度学习方法。
- [Deep Neural Networks for Acoustic Modeling in Speech Recognition](https://ieeexplore.ieee.org/document/6296526) — Geoffrey Hinton, Li Deng, Dong Yu, George Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, and Andrew Senior (2012)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 29; Pages: 82-97; DOI: [10.1109/MSP.2012.2205597](https://doi.org/10.1109/MSP.2012.2205597)
  这篇重要论文描述了将深度神经网络整合到传统基于HMM的声学模型中实现的早期显著进步。
