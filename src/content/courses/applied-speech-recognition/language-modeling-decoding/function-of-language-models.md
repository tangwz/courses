---
course: "applied-speech-recognition"
chapter: "language-modeling-decoding"
lesson: "function-of-language-models"
sourceId: 7150
sourceUrl: "https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding/function-of-language-models"
title: "语言模型在自动语音识别中的作用"
description: "了解语言模型如何为声学模型的输出添加语言语境，以生成更连贯的文本。"
order: 1
plots: []
sourceHash: "6bd1924a8d4379cdbf182fb5b0fe211dddb66cdfd8a65ce7b92ee4c1a3569f02"
sourceCorrections: []
---

声学模型擅长其特定任务：将音频特征映射到可能的字符或音素序列。然而，它缺乏对语法、语境或常识的理解。这就是为什么它会轻易地将“recognize speech”与“wreck a nice beach”混淆。这两个短语在语音上相似，从模型的角度来看，它们同样有效。缺失的部分是语言语境。

语言模型（LM）提供这种语境。它的基本作用是量化 (quantization)给定词序列的可能性。它回答的问题是：“这在英语中是一个合理的句子吗？”通过结合声学模型的“听起来是什么”分析与语言模型的“什么有意义”分析，自动语音识别系统可以做出更明智的最终判断。

### 语言概率的作用

声学模型产生的是原始、未经处理的输出。可以将其看作是一个可能性列表，其中每个可能性都是一个可以合理匹配输入音频的词序列。语言模型充当这些可能性的过滤器或仲裁者。它审查每个候选转录，并根据其流畅度和自然度为其分配概率分数。

请看以下图表，它说明了语言模型如何融入自动语音识别流程。

> 声学模型根据声音生成多个假设。语言模型对每个假设进行语言合理性评分，使解码器能够选择最合理的转录。

语言模型本身通常通过大量的文本数据进行训练，例如书籍、文章和网页。从这些数据中，它学习词语之间的统计关系。它学习到“recognize”经常跟在“speech”后面，而“wreck”很少以那个精确顺序跟在“a nice beach”后面，尽管两者在语法上都可行。因此，它会为 $P(\text{"recognize speech"})$ 分配高得多的概率，而不是 $P(\text{"wreck a nice beach"})$。

### 组合分数以做出更好的判断

语言模型的整合发生在解码阶段，系统在此阶段使这种决策过程规范化。如引言中提到的，目标是找到使组合分数最大的词序列 $W$。让我们再次查看该公式：


$$
\text{得分}(W) = \log P_{\text{声学}}(X|W) + \alpha \log P_{\text{语言模型}}(W)
$$


分解如下：

- **$\log P_{\text{声学}}(X|W)$**：该项来源于声学模型。它表示音频输入 $X$ 由词序列 $W$ 生成的概率。高分表示声学匹配度强。我们使用对数概率来避免数值下溢，并将概率的乘积转换为更简单的和。
- **$\log P_{\text{语言模型}}(W)$**：这是语言模型的贡献。它是词序列 $W$ 出现的先验概率。高分表示该序列在语言上常见且格式良好。
- **$\alpha$**：这是一个可调节的超参数 (parameter) (hyperparameter)，作为权重 (weight)。它平衡语言模型与声学模型的影响。如果将 $\alpha$ 设置得过高，系统可能会倾向于语法完美但声学不准确的句子。如果设置得过低，你就会回到“wreck a nice beach”的问题。找到 $\alpha$ 的合适值通常通过在验证数据集上测试不同值来凭经验完成。

### 搜索空间的剪枝

如果没有语言模型，尝试寻找最佳转录的解码器将有大量的路径需要检查。在每个时间步，声学模型可能会建议几个可能的字符或词语，导致潜在句子数量的指数级增长。

语言模型在此搜索中充当重要指引。当解码器考虑扩展部分句子时，它可以使用语言模型来检查新的、更长句子的概率。如果特定路径开始形成无意义短语（例如：“speech recognize a”），语言模型会给它分配非常低的概率。解码器便可以安全地“剪枝”或丢弃这条路径，使其能够将计算资源集中在更有希望的候选上。

在接下来的部分中，我们会介绍如何构建一个简单而有效的 n-gram 语言模型，然后将其整合到束搜索解码器中，这是一种有效遍历此搜索空间的实用算法。

## 参考资料

- [Speech and Language Processing (3rd Edition Draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Online Draft
  一本全面且广泛使用的教材，详细介绍了语言模型、声学模型以及它们在自动语音识别系统中的整合。
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) — Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin (2003)
  Journal: Journal of Machine Learning Research; Volume: 3; Pages: 1137-1155; DOI: [10.1162/153244303322753381](https://doi.org/10.1162/153244303322753381)
  这篇奠基性论文引入了基于神经网络的语言模型，通过学习词语的分布式表示并捕捉更复杂的语言模式，超越了传统的N-gram模型。
- [CS224S: Spoken Language Processing](https://web.stanford.edu/class/cs224s/) — Andrew Maas, Tolúlá» páº¹Ì  Ogunremi (2025)
  Publisher: Stanford University
  一门高级大学课程，提供关于语音识别理论和实践方面的深入讲座和材料，包括语言建模和解码。
