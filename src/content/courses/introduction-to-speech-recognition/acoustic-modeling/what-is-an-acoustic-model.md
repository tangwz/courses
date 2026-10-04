---
course: "introduction-to-speech-recognition"
chapter: "acoustic-modeling"
lesson: "what-is-an-acoustic-model"
sourceId: 7100
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-3-acoustic-modeling/what-is-an-acoustic-model"
title: "什么是声学模型？"
description: "明确说明声学模型的作用：表示音频信号与语言音素之间的关系。"
order: 1
plots: []
sourceHash: "906772512b5141499d4b02acc834bc446bbb49829aa63d4de79008b2ee4c5f38"
sourceCorrections: []
---

语音识别系统将原始音频转换为特征向量 (vector)序列。在这些向量中，梅尔频率倒谱系数（MFCCs）等数值虽然代表了声音的抽象属性，但本身缺乏固有的语言意义。这带来了一个基本难题。为了克服这一点，需要一个组件将这些数值特征转换为语言的基本单位。这个组件就是声学模型。

它的核心是一个统计模型，充当声音和音素之间的翻译器。对于由单个特征向量表示的每个短音频片段，声学模型会计算该声音对应于给定语言中每个可能音素的概率。例如，当它分析一个特征向量时，可能会确定该声音有 70% 的可能性是 /t/，10% 的可能性是 /d/，而其他所有音素的可能性都非常低。

可以将其视为一个专门的模式识别器。长期以来，它已在数千小时的语音数据上进行训练，这些数据的音频都与其正确的语音转录精确对齐 (alignment)。通过这种训练，它学习了每个音素的独特特征。它学习了 /s/ 音在特征向量形式下“看起来”是怎样的，以及 /ʃ/（“sh”音）又“看起来”是怎样的。

### 声学模型的作用

声学模型的主要职责是回答一个具体问题：“给出这个特定音频特征片段，它对应特定音素的可能性有多大？”这个过程对音频输入中的每个时间帧重复执行，从而生成一系列音素概率。

> 声学模型获取一帧音频特征，并计算每个可能音素的可能性。

这个输出不是一个明确的答案。它是一组概率。模型不会说“这是一个 /t/ 音。”相反，它为每种可能性提供一个统计得分。这是一个重要区别，因为语音本质上是可变的。一个人对 /t/ 的发音会根据其口音、语速或前后音素而变化。通过提供概率，声学模型为ASR系统提供了考虑多种语音解释的灵活性。

### 概率基础

声学模型学习到的关系正式表达为条件概率。它计算可能性，通常写作：


$$
P(\text{音频特征} | \text{音素})
$$


你可以将其读作“观察到这组特定音频特征的概率，*假设*某个音素被说出。”

例如，模型计算：

- $P(\text{特征} | \text{/k/})$: 如果是 /k/ 音，这些特征的可能性有多大？
- $P(\text{特征} | \text{/æ/})$: 如果是 /æ/ 音（如“cat”中的发音），这些特征的可能性有多大？
- $P(\text{特征} | \text{/t/})$: 如果是 /t/ 音，这些特征的可能性有多大？

声学模型对语言中的每个音素执行此计算。产生最高概率的音素被认为是该小段音频可能性最大的候选。这些可能性得分随后会传递到ASR流程的下一阶段——解码器，解码器将结合来自语言模型的信息来构建最终的文本转录。

在接下来的章节中，我们将了解用于构建这些模型的技术，首先是高斯混合模型（GMMs）和隐马尔可夫模型（HMMs）的经典组合。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本教材概述了语音识别，描述了声学模型作为ASR核心部分及其统计特性。
- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://doi.org/10.1109/5.18626) — Lawrence R. Rabiner (1989)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 77; Pages: 257-286; DOI: [10.1109/5.18626](https://doi.org/10.1109/5.18626)
  一篇极具影响力的论文，解释了隐马尔可夫模型，这是早期语音识别系统中声学建模的基本统计模型。
- [An Overview of Automatic Speech Recognition](https://link.springer.com/book/10.1007/978-1-4471-5779-3) — Dong Yu, Li Deng (2014)
  Publisher: Springer; Pages: 1-28; DOI: [10.1007/978-1-4471-5779-3](https://doi.org/10.1007/978-1-4471-5779-3)
  本章概述了自动语音识别，其中包括对ASR系统中声学模型功能和设置的清晰描述。
