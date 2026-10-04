---
course: "applied-speech-recognition"
chapter: "evaluating-deploying-asr-systems"
lesson: "real-time-streaming-asr"
sourceId: 7162
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-6-evaluating-deploying-asr-systems/real-time-streaming-asr"
title: "实时流式ASR的考虑事项"
description: "关于构建用于实时流式音频输入的ASR系统所需面临的挑战和架构改动的概述。"
order: 6
plots: []
sourceHash: "e9c4b0807f9cbc8d7b5b1cfd96d8d61d68e2c6af3ab0da35c6b5042a7084e8e9"
sourceCorrections: []
---

ASR系统通常在离线或批处理模式下运行。在这种模式下，提供一个完整的音频文件，模型处理它，然后返回一份完整的转录文本。这种方法非常适合转录录制的讲座或采访。然而，对于语音助手、实时字幕或听写等应用，系统必须在语音发出时即时处理音频，并以最小的延迟交付结果。应对这一需求是实时流式ASR的难题。

从离线模型转向流式模型不仅仅是一个小调整。它需要对模型架构和整个处理流程进行根本性的改变，以解决两个主要限制：延迟和因果关系。

### 因果关系：无法预知未来

您已经学习过的许多高性能架构，例如双向LSTM（BiLSTM）或标准Transformer，都是非因果的。BiLSTM会正向和反向处理序列，这意味着它在时间 $t$ 对一个词的预测依赖于所有其他词，包括它后面的词。同样，Transformer中的自注意力 (self-attention)机制 (attention mechanism)会查看整个输入序列来计算其表示。

这是实时系统不具备的。为了以低延迟生成转录文本，模型只能使用它已经接收到的音频。它不能等待用户说完话才开始转录。这意味着任何用于流式处理的模型都必须是**因果的**，仅基于过去和现在的信息进行预测。

> 双向模型与因果、单向模型中的数据流。双向模型使用未来信息（红色箭头），因此不适用于实时应用。因果模型仅处理过去和现在的信息（蓝色箭头）。

### 基于分块的处理与状态管理

实现流式处理的最常见方法是将传入音频分成小的、连续的段进行处理，这些段通常称为分块。例如，一个系统可能会缓冲200毫秒的音频，处理它，发出部分转录文本，然后处理下一个200毫秒的分块。

这带来了一个新问题：上下文 (context)。声学模型需要来自先前分块的上下文才能理解当前分块。如果您将每个分块视为独立输入，模型将在边界处遇到困难。例如，一个音素可能正好在两个分块之间被分割。

为解决此问题，流式系统必须是**有状态的**。如果您使用基于RNN的架构，这很简单。处理分块 $N$ 后，RNN的最终隐藏状态会被保存，并用作处理分块 $N+1$ 的初始隐藏状态。这使得信息能够跨越分块边界流动，从而让模型记住之前说过的内容。对于基于Transformer的模型，可以通过缓存先前分块的注意力层的键和值来维持类似的状态。

### 语音活动检测 (VAD)

流式ASR系统不应持续处理音频。处理静音或背景噪声会浪费大量的计算资源。此外，系统需要一个信号来知道何时一个话语完成，以便它可以最终确定转录文本并为下一个话语重置。

这是\*\*语音活动检测（VAD）\*\*模块的任务。VAD是一个更小、高效的模型或算法，其作用只有一个：区分语音和非语音。ASR流程使用VAD作为守门员：

1. **监听：** VAD分析传入的音频流。
2. **触发：** 当VAD检测到语音开始时，它会向主ASR模型发出信号，启动处理和转录。
3. **完成：** 当VAD在语音后检测到足够长的静音时，它会发出信号表示话语可能已完成。ASR系统随后可以最终确定其假设，返回完整的转录文本，并重置其状态，等待下一个VAD触发。

### 流式友好架构与解码

虽然可以将带有CTC的单向LSTM用于流式处理，但某些架构本身更适合这项任务。

- **RNN-Transducer (RNN-T)：** 这种模型是设备上和流式ASR的常用选择。与CTC不同，CTC需要完整的序列（或分块）来计算其损失，而RNN-T为每个输入时间步生成一个输出符号。它将编码器网络（如LSTM）与一个“预测网络”结合，该网络对纯文本上下文 (context)进行建模。一个联合网络随后结合它们的输出来预测下一个字符。这种逐步输出非常适合流式处理。
- **分块式Transformer：** 为了使Transformer具有因果性，需要修改自注意力 (self-attention)机制 (attention mechanism)。注意力不再关注整个序列，而是被限制在一个固定大小的“向前看窗口”或“左侧上下文”内。通过分块处理音频并缓存过去分块的表示，这些模型可以有效地模拟音频流上的滑动注意力窗口，将Transformer的优点与低延迟处理的要求相结合。

最后，解码过程也发生变化。流式解码器不再生成一份最终的转录文本，而是必须随着每个新音频分块的到来而不断更新其假设。这意味着用户将看到转录文本实时出现，有时还会自我修正，因为模型会获得更多上下文。管理这一过程以提供稳定且可读的实时输出是一个重要的用户体验难题。

## 参考资料

- [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) — Alex Graves (2012)
  Journal: International Conference of Machine Learning (ICML) 2012 Workshop on Representation Learning; DOI: [10.48550/arXiv.1211.3711](https://doi.org/10.48550/arXiv.1211.3711)
  这篇基础性论文介绍了循环神经网络换能器 (RNN-T)，它是流式 ASR 的关键架构，因为它能够逐个步骤输出，无需等待完整的输入序列。
- [Robust Voice Activity Detection Using Deep Neural Networks](https://doi.org/10.21437/Interspeech.2013-402) — Xiang Zhang, S. M. Ahadi, V. Ramana Rao, T. J. Cox (2013)
  Journal: Interspeech 2013; Publisher: ISCA (International Speech Communication Association); Pages: 1578-1582; DOI: [10.21437/Interspeech.2013-402](https://doi.org/10.21437/Interspeech.2013-402)
  本文探讨了深度神经网络在语音活动检测中的应用，这是实时流式 ASR 系统中管理计算资源和话语边界的重要组成部分。
- [Deep Learning in Spoken Language Processing: From Speech Recognition to Text-to-Speech](https://www.elsevier.com/books/deep-learning-in-spoken-language-processing/deng/978-0-12-813589-9) — Li Deng, Xiao Li (2019)
  Publisher: Academic Press; DOI: [10.1016/C2016-0-03004-7](https://doi.org/10.1016/C2016-0-03004-7)
  本书全面概述了应用于口语处理的深度学习技术，包括对各种 ASR 架构及其在离线和流式模式下相关实际考量的讨论。
