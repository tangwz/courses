---
course: "introduction-to-speech-recognition"
chapter: "acoustic-modeling"
lesson: "hmm-for-sequential-data"
sourceId: 7105
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/hmm-for-sequential-data"
title: "隐马尔可夫模型 (HMM) 在序列数据中的应用"
description: "了解隐马尔可夫模型（HMM）如何用于处理语音的序列特性以及对声音间转移的建模。"
order: 4
plots: []
sourceHash: "5bdd48bec9a0ca94f2d60b899dc43943a6d0a9f078b69e1d34bf81ec26f3be2e"
sourceCorrections: []
---

语音本身就是序列化的。单词“cat”中的发音以特定顺序出现，正是这个顺序赋予了单词其含义。一个只识别单个声音特征的模型，例如高斯混合模型，会忽略这种基本结构。这就像拥有一袋字母，你可以看到有哪些字母，但无法构成单词或句子。为了正确理解语音，我们需要一个能理解序列的系统。这就是隐马尔可夫模型（HMM）的作用。

HMM 是一种统计工具，设计用于对序列数据进行建模，其中生成数据的底层过程无法直接看到。想象你身处一个没有窗户的房间，想猜测外面的天气。天气本身，无论是“晴朗”还是“下雨”，都是一个*隐藏状态*。你无法直接看到它。然而，你可以观察从外面进来的同事是否带着湿伞。这就是你的*观测值*。通过随着时间推移跟踪这些观测值，你可以推断出最可能发生的天气状况序列。

### HMM 在语音中的应用

该模型直接适用于语音识别：

- **隐藏状态：** 语言的音素就是隐藏状态。我们从不在原始音频波形中直接“看到”一个完美的 /k/ 或 /æ/。
- **观测值：** 我们从短音频帧中提取的特征向量 (vector)，例如 MFCCs，就是我们的观测值。它们是我们所拥有的声学证据。

HMM 提供了一个数学框架，用于在给定观测到的音频特征序列的情况下，找到最可能的隐藏音素序列。

### 隐马尔可夫模型的组成部分

为了构建这个框架，HMM 依赖于三个主要组成部分：

1. **状态集 ($S$):** 对于语音，我们不只是为每个音素使用一个状态。单个音素的发音从其开始到结束都会变化。因此，我们通常用一个由少量状态组成的链来建模每个音素，通常是三个：一个起始状态、一个中间状态和一个结束状态。这使得模型能够更好地捕捉语音声音的动态特性。
2. **转移概率 ($A$):** 这些定义了从一个状态转移到另一个状态的概率。例如，从 /k/ 的“起始”状态转移到其“中间”状态的概率很高。也存在从 /k/ 的最终状态转移到下一个音素（如 /æ/）的第一个状态的概率。这些概率强制执行声音的合法顺序。状态也可以转移回自身，这使得模型能够考虑音素被说得更慢或更快的情况。
3. **发射概率 ($B$):** 这是在特定状态下，产生特定观测值（一个音频特征向量 (vector)）的概率。它回答了这个问题：如果我们处于代表音素 /æ/ 中间部分的状态，那么观测到我们刚刚计算出的特定 MFCC 向量的概率 $P(\mathrm{音频特征} | \mathrm{状态 æ-中})$ 是多少？

下图展示了一个简单的 HMM，用于单词“cat”，该单词由音素 /k/、/æ/ 和 /t/ 组成。每个音素都用三个状态建模，箭头代表可能的转移。

> 一个表示单词“cat”（/k/ /æ/ /t/）的 HMM。该模型通过每个音素的子状态进行转移。自循环使得模型能够在每个状态中停留不同的时间，从而解释语速差异。

通过结合转移概率和发射概率，HMM 可以计算任何给定音素序列与输入音频的总得分。ASR 系统的目标是找到通过所有可能音素状态的单一路径，该路径生成观测到的音频特征的概率最高。这是一个复杂的搜索问题，但可以通过一种称为维特比算法的特殊过程高效解决，该算法我们将在后续章节中讨论。

HMM 为声学模型提供了序列“支架”。它巧妙地处理语音声音的“何时”问题，对其顺序和持续时间进行建模。然而，HMM 框架本身并未规定*如何*计算发射概率，即 $P(\mathrm{音频特征} | \mathrm{状态})$ 部分。它需要一种方法来评估给定音频帧与音素状态的匹配程度。正如你将在下一节中看到的那样，这正是 GMM 再次出现的地方，它们与 HMM 形成强大的合作关系。

## 参考资料

- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://ieeexplore.ieee.org/document/18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  介绍HMM理论及其在语音识别中具体应用的奠基性论文。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: Chapter 13; DOI: [10.1117/1.2819119](https://doi.org/10.1117/1.2819119)
  在更广泛的机器学习背景下，提供了HMM的统计和概率处理方法。
