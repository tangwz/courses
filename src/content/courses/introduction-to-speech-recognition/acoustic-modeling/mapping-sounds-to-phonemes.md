---
course: "introduction-to-speech-recognition"
chapter: "acoustic-modeling"
lesson: "mapping-sounds-to-phonemes"
sourceId: 7101
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/mapping-sounds-to-phonemes"
title: "将声音映射到音素"
description: "理解声学模型的主要任务，即根据一段音频计算音素的概率。"
order: 2
plots: []
sourceHash: "ca9225c9a3fb38bb4c37ad21625096b91ebbc5e6a575ecb11d0f93ab798fb8d9"
sourceCorrections: []
---

本质上，声学模型充当着一座桥梁。音频波形会被转换为一系列特征向量 (vector)，例如MFCC。每个向量都提供了声音特征在极短时间（通常为10到25毫秒）内的数字快照。主要目的是将这些数字转换成语言的基本声音单位，即音素。

声学模型的主要任务是判断每个特征向量最可能的音素。然而，这并非简单的点对点查找。一个人在单词“top”中发出/t/音的方式，在声学上可能与“water”或“stop”中的/t/不同。口音、语速甚至说话者的情绪都可能改变声音。因此，声学模型需要考虑概率。

模型不会做出明确的决定，而是为语言中每个可能的音素计算一个似然分数。对于单帧音频特征，它会询问：

- 这些特征对应音素/k/的概率是多少？
- 它们对应/æ/的概率是多少？
- 它们对应/t/的概率是多少？
- 以此类推，对所有音素进行计算。

这个过程为每个时间步生成一个概率分布。结果不是单一的音素，而是一系列可能性，每个都带有一个分数。这种关系通常以数学形式表示为似然 $P(\text{特征} | \text{音素})$。这个公式的含义是“在特定音素被说出时，观察到一组特定音频特征的概率。”模型会为每个音素计算这个值，从而评估每个音素与观察到的音频数据吻合的程度。

下图描绘了单个时间帧的这种映射关系。MFCC特征向量被输入到声学模型中，模型再输出每个候选音素的概率集合。

> 声学模型根据特定时间帧的音频特征计算每个音素的概率。

这种概率输出具有重要意义。如果两个音素听起来非常相似，例如/p/和/b/，模型可能会为两者都分配较高的概率。例如，对于特定声音，它可能会给出 $P(\text{特征} | \text{/p/}) = 0.45$ 和 $P(\text{特征} | \text{/b/}) = 0.40$。声学模型本身不必做出最终选择。它只提供这些分数。

通过处理音频片段中整个特征向量序列，声学模型会生成一个相应的概率分布序列。这种丰富的、时间对齐 (alignment)的语音信息成为ASR流程后续阶段的主要输入。系统随后会使用语言模型和解码器来权衡这些可能性，考虑周围的上下文 (context)，并最终决定说话者说的是“big”还是“pig”。

在接下来的章节中，我们将了解如何构建这种模型，从传统的统计方法开始，逐步转向现代神经网络 (neural network)方法。

## 参考资料

- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://doi.org/10.1109/5.18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  一篇基础性论文，介绍了隐马尔可夫模型（HMMs）作为语音识别中声学建模的主要统计框架。
- [Automatic Speech Recognition: A Deep Learning Approach](https://link.springer.com/book/10.1007/978-1-4471-5779-3) — Dong Yu, Li Deng (2014)
  Publisher: Springer; DOI: [10.1007/978-1-4471-5779-3](https://doi.org/10.1007/978-1-4471-5779-3)
  一本详细介绍深度学习技术在自动语音识别中应用的著作，提供了从传统声学模型演进的背景。
