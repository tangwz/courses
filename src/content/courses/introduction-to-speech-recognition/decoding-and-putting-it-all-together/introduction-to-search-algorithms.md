---
course: "introduction-to-speech-recognition"
chapter: "decoding-and-putting-it-all-together"
lesson: "introduction-to-search-algorithms"
sourceId: 7129
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together/introduction-to-search-algorithms"
title: "搜索算法概览"
description: "对语音识别中搜索问题的整体介绍，并说明为何需要高效算法。"
order: 3
plots: []
sourceHash: "0d2b52d3bd89708d87a16c9533a7b350d556f9fa57c9ac0b8ca2d3195519b44f"
sourceCorrections: []
---

解码器的任务是找到与输入音频 ($O$) 最匹配的单个词序列 ($W$)。它通过结合声学模型得分 $P(O|W)$ 和语言模型得分 $P(W)$ 来完成此任务。难题在于可能的词序列数量庞大到难以想象。一段十秒长的音频片段可能对应数百万甚至数十亿个潜在句子。

检查每一种可能性（这种方法被称为“暴力搜索”）在计算上是不可行的。如果你的词汇量是 20,000 个词，那么可能的五词句子的数量将是 $20,000^5$，这个数字非常大，以至于最快的计算机也无法在合理的时间内检查所有这些组合。解码器需要一种效率高得多的策略。

### 将搜索问题视为寻路任务

处理这个问题的更好方法是将其重新定义为寻路问题。设想一个大型图，其中每个点代表一个可能跟在前一个词后面的词。解码器的任务是找到通过这个图从起点到终点“成本最低”或“最有可能”的路径。

图中的每条路径都代表一个独特的句子，或者一个“假设”。路径上的每一步都有相关的成本，这个成本根据声学和语言模型概率计算得出。声学模型得分低（意味着声音匹配不佳）或语言模型得分低（意味着词序列不太可能）的路径将很快变得非常“昂贵”。

> 一个简化的搜索图。解码器检查不同的路径（假设），例如“recognize speech”和“wreck a nice beach”。红色路径表示声学模型和语言模型组合得分最高的路径。

### 剪除不太可能的假设

搜索算法不盲目地考察所有路径，而是使用一种智能技术，称为**剪枝**。当解码器逐词构建句子假设时，它会记录这些假设的得分。如果某条路径的可能程度显著低于目前已知的最佳路径，算法就会放弃它。

这是一种“束搜索”的形式，解码器在每一步只保留少数（或“束”）最有希望的假设，并丢弃其余的。通过及早剪除大部分不佳路径，解码器可以将计算资源集中在那些实际有可能正确的假设上。

这个过程使得现代语音识别成为可能。如果没有高效的搜索算法来处理庞大的可能性空间，即使是最好的声学和语言模型也将毫无用处。在下一节中，你将学习维特比算法，这是一种经典且有效的算法，是许多解码器的主要方法。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  涵盖语音识别基础知识的经典教材，包括声学模型、语言模型、解码问题、维特比算法和束搜索。
- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://ieeexplore.ieee.org/document/18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  一篇经典教程，解释了隐马尔可夫模型和维特比算法，对语音识别中的概率序列建模和解码至关重要。
