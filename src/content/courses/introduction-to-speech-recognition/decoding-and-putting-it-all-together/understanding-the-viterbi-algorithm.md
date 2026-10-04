---
course: "introduction-to-speech-recognition"
chapter: "decoding-and-putting-it-all-together"
lesson: "understanding-the-viterbi-algorithm"
sourceId: 7131
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together/understanding-the-viterbi-algorithm"
title: "理解维特比算法"
description: "对维特比算法的原理说明，它是一种用于找到最可能路径的动态规划方法。"
order: 4
plots: []
sourceHash: "365af08dfaf9f8b4167529a51e9219f264537dac5c214e5c987a8a384913df86"
sourceCorrections: []
---

寻找最佳词语序列是一个重大的计算难题。对于给定的音频输入，可能的句子数量庞大到惊人。解码器计算每个可能句子的概率的暴力尝试法根本不可行。对于一个包含50,000词的词汇表 (vocabulary)，即使一个简短的10词句子，也有$50,000^{10}$种潜在组合。我们急需一种效率更高的办法，在这个庞大的搜索空间中找到出路。

这正是维特比算法发挥作用之处。它是一种高效算法，属于被称为*动态规划*的一类方法。动态规划的核心思想是将一个复杂问题分解成一系列简单的子问题，每个子问题只解决一次并存储其解决方案。维特比算法运用此原则来找到模型中最可能的状态序列，在我们的情境中，这对应于最可能的词语序列。

### 穿越时间的路径

想象解码过程是在一个网格中寻找最佳路径，这个网格通常被称为**篱笆图**或**格子图**。网格中的每一列代表一个时间点（对应于一个音频帧或片段），每一列中的每个节点代表在该时间点可能说出的词语（或音素）。

该算法每次一个时间步地从左到右在篱笆图中移动。在每一步，它计算到达*每个*可能词语的最可能路径。维特比算法的重要发现是：为了找到到达当前时间步某个词语的最佳路径，你只需要知道到达*所有*前一时间步词语的最佳路径。你不需要记住每条可能路径的完整历史。

通过扩展前一步的最佳路径并加入新的成本（基于声学和语言模型概率），算法找到新的最佳路径集合。然后它丢弃所有其他不太可能的路径。这种持续修剪不可能路径的做法，正是该算法如此高效的原因。

### 示例说明

假设在听到一些音频后，解码器试图在两个部分句子之间做出选择：“recognize speech”和“wreck a nice beach”。声学模型可能会发现这两个短语听起来相似。然而，语言模型知道“recognize speech”比“wreck a nice speech”是更常见的短语。

维特比算法将按以下步骤进行：

1. **时间步 1：** 它考虑音频的第一部分。声学模型对“recognize”和“wreck a nice”都给出高分。算法将两者都保留为可能性。
2. **时间步 2：** 它分析下一段音频，听起来像“speech”。解码器现在评估四条潜在路径：
   - `recognize` -> `speech`
   - `recognize` -> `beach`
   - `wreck a nice` -> `speech`
   - `wreck a nice` -> `beach`
3. **计算与修剪：** 对于这四条路径中的每一条，解码器通过结合声学分数（音频与词语的匹配程度）和语言模型分数（词语序列的可能性）来计算一个总分数。
   - `P(音频 | "recognize speech") * P("recognize speech")` 很可能得分很高。
   - `P(音频 | "wreck a nice beach") * P("wreck a nice beach")` 也将有相当高的分数。
   - 然而，像`recognize` -> `beach`和`wreck a nice` -> `speech`这样的路径，其语言模型概率会非常低。算法会为它们分配低分，并有效地将它们从考虑中移除。
4. **回溯：** 到达音频末尾后，算法确定总体概率最高的路径的最后一个词。从那里，它通过保存的指针向后追溯，在每个时间步重建唯一的最佳词语序列。

下图显示了此过程。在每个时间步，算法扩展前一步幸存下来的路径。不太可能的路径（虚线）被丢弃，只保留到达每个词语的最可能路径。通过从末尾回溯找到最终的最佳路径。

> 一个显示维特比搜索的简化格子图。算法根据结合声学和语言模型的分数评估转换。不太可能的路径被修剪（灰色虚线），而最可能路径（橙色）则通过从得分最高的最终状态回溯构建。

### 其工作原理

维特比算法保证能找到穿过篱笆图的唯一最佳路径。它这样做而无需评估每一个可能性。通过在每个时间步做出局部最优决策（即只保留到达每个词语的最佳路径），它达到了全局最优解。这种将一个难以处理的问题转化为可管理的问题的做法，是现代ASR系统如何快速准确地生成转录的根本方法。

## 参考资料

- [Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm](https://ieeexplore.ieee.org/document/1054010) — Andrew J. Viterbi (1967)
  Journal: IEEE Transactions on Information Theory; Publisher: IEEE; Volume: 13; Pages: 260-269; DOI: [10.1109/TIT.1967.1054010](https://doi.org/10.1109/TIT.1967.1054010)
  首次提出Viterbi算法的原始论文。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2023)
  Publisher: Pearson
  一本全面的教材，涵盖了隐马尔可夫模型和语音识别背景下的Viterbi算法。
- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://ieeexplore.ieee.org/document/18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  一篇被广泛引用的教程，深入解释了隐马尔可夫模型及其在语音识别中的应用，包括Viterbi算法。
