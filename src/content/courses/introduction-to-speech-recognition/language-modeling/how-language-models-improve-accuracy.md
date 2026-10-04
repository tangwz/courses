---
course: "introduction-to-speech-recognition"
chapter: "language-modeling"
lesson: "how-language-models-improve-accuracy"
sourceId: 7122
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-4-language-modeling/how-language-models-improve-accuracy"
title: "语言模型如何提高准确性"
description: "了解声学模型和语言模型得分的结合如何带来更准确、更连贯的转录。"
order: 6
plots: []
sourceHash: "a17495cc98dbae408cc3acf0421b39821db0bdf7d7998f739d6fdba8f4d9644a"
sourceCorrections: []
---

声学模型为转录提供原始材料，但它在运行时不具备语法或语义感知。它努力将声音映射到音素，但这可能导致如果两个短语听起来相似，一个无意义的短语会被认为与一个有意义的短语具有相同的可能性。这时，语言模型就变得不可或缺。它扮演着语言判断者的角色，评估哪个词语序列最合理。

### 结合两种证据

为生成准确的转录，语音识别系统必须平衡两种不同的证据：

1. **声学证据：** 潜在的词语序列与输入音频信号的匹配程度如何？该分数来自**声学模型（AM）**。
2. **语言学证据：** 该词语序列在给定语言中出现的可能性有多大？此概率来自**语言模型（LM）**。

在语音识别中，解码器在生成文本转录时，并不仅仅选择声学匹配度最佳的选项。相反，它结合了声学模型和语言模型的信息，寻找具有最高组合得分的词语序列。这可以表示为寻找词语序列 $W$，它能在给定音频 $A$ 的情况下使该序列的概率最大。这种关系通常简化为寻找声学模型和语言模型概率乘积的最大值：


$$
\text{最终得分} \propto P(\text{音频} | \text{词语}) \times P(\text{词语})
$$


在此，$P(\text{音频} | \text{词语})$ 代表声学模型的得分，$P(\text{词语})$ 是语言模型的概率。系统选择能使此组合得分尽可能高的词语序列。

### 实际例子

让我们回到熟悉的例子：音频听起来可能是“recognize speech”（识别语音）或“wreck a nice beach”（破坏一片漂亮海滩）。

1. **声学模型的评估：** 声学模型处理音频后，发现这两个短语在声学上都非常接近。如果说话者的发音碰巧与第二个短语更吻合，它甚至可能给第二个短语更高的分数。

   - “recognize speech”的声学得分：0.85
   - “wreck a nice beach”的声学得分：0.88

   仅根据声学，“wreck a nice beach”是领先者。
2. **语言模型的输入：** 现在，语言模型评估这些短语的可能性。由于它经过海量文本训练，它知道“recognize speech”是一个常见且合乎逻辑的短语，尤其是在技术背景下。相比之下，“wreck a nice beach”在语法上是有效的，但其出现的可能性极低。

   - “recognize speech”的语言模型概率：高（例如，0.7）
   - “wreck a nice beach”的语言模型概率：极低（例如，0.001）
3. **计算最终得分：** 解码器组合这些分数以找出赢家。

   - **“recognize speech”的最终得分**： $0.85 \times 0.7 = 0.595$
   - **“wreck a nice beach”的最终得分**： $0.88 \times 0.001 = 0.00088$

结果很清楚。语言模型的高概率极大地提高了“recognize speech”的得分，使其轻松获胜，尽管其声学得分略低。语言模型通过提供重要的语言学语境，有效地推翻了声学上模棱两可的结果。

> 解码器权衡来自声学模型和语言模型的证据，以选择最有可能的转录。

通过增加这一层语言学验证，语言模型显著减少了错误。它引导ASR系统生成不仅声学上合理，而且语法正确、语义合理的转录。声学模型与语言模型之间的这种合作对几乎所有现代语音识别系统的准确性都非常重要。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Pearson
  这本书是语音识别的全面参考资料，涵盖了声学模型、语言模型及其在自动语音识别系统中的整合等基本概念。
- [Statistical Methods for Speech Recognition](https://mitpress.mit.edu/books/statistical-methods-speech-recognition) — Fred Jelinek (1998)
  Publisher: The MIT Press
  这本书介绍了语音识别的基础统计方法，详细解释了声学模型、语言模型及其组合以实现最佳转录。
- [Deep Learning for Speech Recognition: An Overview](https://www.researchgate.net/publication/262557438_Deep_Learning_for_Speech_Recognition_An_Overview) — Li Deng and Dong Yu (2014)
  Journal: IEEE Signal Processing Magazine; Publisher: IEEE; Volume: 31; Pages: 82-96; DOI: [10.1109/MSP.2013.2290903](https://doi.org/10.1109/MSP.2013.2290903)
  这篇概述文章讨论了深度学习方法在语音识别中的应用，展示了神经网络如何增强声学和语言模型组件以提高准确性。
