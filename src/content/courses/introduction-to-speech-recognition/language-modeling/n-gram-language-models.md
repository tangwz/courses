---
course: "introduction-to-speech-recognition"
chapter: "language-modeling"
lesson: "n-gram-language-models"
sourceId: 7116
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-4-language-modeling/n-gram-language-models"
title: "N-gram 语言模型：二元语法和三元语法"
description: "了解经典的 N-gram 模型，它们根据前 N-1 个词来预测下一个词。"
order: 3
plots: ["plots/7116-0.json"]
sourceHash: "bf149292e36cc84bc000b5de34a117d22a6a758962ce125b5443aac4a71551be"
sourceCorrections: []
---

为解决“recognize speech”（识别语音）和“wreck a nice beach”（毁掉一个漂亮海滩）等短语之间的模糊性，语言模型必须判断哪个词序列的可能性更大。然而，从头计算整个句子的概率是一项复杂的工作。对于一个包含 $k$ 个词的句子 $w_1, w_2, \dots, w_k$，概率的链式法则如下：


$$
P(w_1, w_2, \dots, w_k) = P(w_1) \times P(w_2 | w_1) \times P(w_3 | w_1, w_2) \times \dots \times P(w_k | w_1, \dots, w_{k-1})
$$


可以看出，预测每个后续词所需的上下文 (context)变得越来越长。这种方法不仅计算量大，而且需要海量数据才能准确估计每个可能词序列的概率，这几乎是不可能实现的。

N-gram 模型通过一个简化假设提供了一个实际的解决方案：一个词的概率仅依赖于前面少量固定数量的词，而不是整个序列。这被称为**马尔可夫假设**。

### 二元语法模型 (N=2)

最直接的 N-gram 模型是**二元语法模型**，这种模型假设一个词的概率仅取决于它前面的一个词。两个相邻词的序列称为二元语法。

词 $w_i$ 在给定前一个词 $w_{i-1}$ 条件下的概率表示为 $P(w_i | w_{i-1})$。为了找到整个句子的概率，我们将构成它的二元语法的概率相乘。

让我们将此应用于我们的例子，“recognize speech”：


$$
P(\text{recognize speech}) \approx P(\text{recognize}) \times P(\text{speech} | \text{recognize})
$$


我们通过统计一个庞大的文本数据集（称为语料库）中词的出现次数来估计这些概率。条件概率的计算式为：


$$
P(\text{speech} | \text{recognize}) = \frac{\text{Count}(\text{"recognize speech"})}{\text{Count}(\text{"recognize"})}
$$


如果“recognize”在我们的语料库中出现 1,000 次，并且其后跟着“speech”的有 400 次，那么 $P(\text{speech} | \text{recognize}) = 400 / 1000 = 0.4$。如果其后跟着“a”只有 5 次，那么 $P(\text{a} | \text{recognize}) = 5 / 1000 = 0.005$。二元语法模型会了解到，“speech”比“a”更有可能跟在“recognize”之后。



![二元语法概率比较](plots/7116-0.json)



> 概率取自文本语料库。“recognize speech”这个短语比“nice beach”是更常见的搭配，这使得它在二元语法模型中可能性更大。

### 三元语法模型 (N=3)

二元语法模型有作用，但有时一个词的上下文 (context)不足够。**三元语法模型**通过假设一个词的概率取决于前面两个词来扩展上下文。三个相邻词的序列称为三元语法。

概率表示为 $P(w_i | w_{i-2}, w_{i-1})$。

让我们看看短语“a nice beach”。三元语法模型将根据其前面的两个词来计算“beach”的概率：


$$
P(\text{beach} | \text{a nice}) = \frac{\text{Count}(\text{"a nice beach"})}{\text{Count}(\text{"a nice"})}
$$


这种额外的上下文信息很有用。词“nice”后面可以跟许多词语（“nice day”、“nice car”、“nice person”）。然而，“a nice”这个序列提供了更强的提示，表明“beach”可能紧随其后，这与二元语法模型仅凭“nice”看到的情况不同。这有助于 ASR 系统做出更准确的判断。

> 二元语法和三元语法模型的依存关系图。三元语法模型使用更宽泛的两个词的上下文来预测下一个词，这能够更好地反映语言的细节模式。

### 三元语法与稀疏性问题

你可以将这种模式扩展到四元语法（使用 3 个词的上下文 (context)）、五元语法（使用 4 个词），以此类推。然而，随着 N 的增加，一个重要问题会浮现：**数据稀疏性**。

对于像“recognize speech”这样的二元语法，你可能会在一个文本语料库中找到数千个例子。对于像“to recognize speech”这样的三元语法，你会找到更少的例子。对于像“I am trying to recognize speech”这样的五元语法，你可能找到的例子会非常少，甚至没有。当一个特定的 N-gram 没有出现在训练数据中时，其计数为零，其概率无法正确计算。由于这种权衡，二元语法和三元语法模型在上下文和可靠性之间一直是一个很好的平衡点。

#### 关于句子开头的说明

如何计算句子中第一个词的概率呢？它没有前面的词。为解决此问题，一个特殊的“句子开始”标记 (token)（通常写作 `<s>`）会被添加到训练数据中每个句子的开头。第一个词“wreck”的概率随后作为二元语法 $P(\text{wreck} | \text{<s>})$ 来计算。同样，第二个词“a”的概率将作为三元语法 $P(\text{a} | \text{<s>}, \text{wreck})$ 来计算。这保证了每个词都有所需的上下文量。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University (online edition)
  一本广泛使用的综合性教科书，详细解释了N-gram语言模型、其数学基础以及在语音识别中的应用。
- [CS224N: Natural Language Processing with Deep Learning - Lecture 2: N-gram Models and Neural Network Models](https://www.youtube.com/watch?v=kYJ5k8B-qA4) — Christopher Manning (2024)
  Publisher: Stanford University
  一所知名大学课程的讲义，涵盖了作为自然语言处理基础概念的N-gram模型。
- [Foundations of Statistical Natural Language Processing](https://mitpress.mit.edu/9780262133609/foundations-of-statistical-natural-language-processing/) — Christopher Manning and Hinrich Schütze (1999)
  Publisher: The MIT Press
  一本有影响力的教科书，介绍了包括N-gram模型在内的统计方法，这些方法是早期自然语言处理研究的基础。
