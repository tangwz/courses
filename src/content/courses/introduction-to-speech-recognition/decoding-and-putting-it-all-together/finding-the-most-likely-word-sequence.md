---
course: "introduction-to-speech-recognition"
chapter: "decoding-and-putting-it-all-together"
lesson: "finding-the-most-likely-word-sequence"
sourceId: 7127
sourceUrl: "https://apxml.com/zh/courses/introduction-to-speech-recognition/chapter-5-decoding-and-putting-it-all-together/finding-the-most-likely-word-sequence"
title: "确定最可能的词语序列"
description: "了解解码器如何结合声学模型和语言模型的概率来评估候选假设。"
order: 2
plots: []
sourceHash: "34d408a4e3639d2705c2fac6e149eded6b7c2de13652e9388a8a9058b558023a"
sourceCorrections: []
---

解码器的主要作用是找出与输入音频特征 ($O$) 最匹配的单个词语序列 ($W$)。这不仅仅是找出听起来正确的内容，更是要找出听起来正确*且*在语言上有意义的内容。解码器通过结合声学模型和语言模型各自的优势来达成此目的。

此过程通过语音识别的基本公式得到很好的体现：


$$
\hat{W} = \underset{W}{\mathrm{argmax}} \, P(O|W) \times P(W)
$$


我们来具体说明这意味着什么。

### 平衡两种依据来源

想象一下，ASR 系统正在听一段听起来像“wreck a nice beach”的语音。

1. **声学模型的贡献 ($P(O|W)$)**：声学模型听取音频特征 ($O$)，并计算这些声音由候选句子 ($W$) 中的词语产生的概率。对于短语“wreck a nice beach”，声学模型可能会返回高概率。声音与词语匹配良好。然而，对于短语“recognize speech”，声学模型*也*可能返回高概率，因为这两个短语在声学上非常相似（同音异义词）。声学模型单独工作时容易混淆。
2. **语言模型的贡献 ($P(W)$)**：语言模型不了解音频内容。它的作用是确定词语序列 ($W$) 在语言中出现的概率。它通过大量文本训练，知道短语“recognize speech”很常见，而“wreck a nice beach”则没有意义且极其罕见。因此，它会给“recognize speech”分配高概率，给“wreck a nice beach”分配接近零的概率。

解码器的作用是生成这些可能的句子，称为**假设**，然后扮演判断者，权衡来自两个模型的依据以做出最终决定。

### 结合得分以得出最终判断

解码器将每个假设的声学模型得分与语言模型得分相乘。综合得分最高的假设获胜。

- **假设 1**：“wreck a nice beach”

  - 声学得分：高（例如，0.9）
  - 语言模型得分：非常低（例如，0.0001）
  - 综合得分：$0.9 \times 0.0001 = 0.00009$
- **假设 2**：“recognize speech”

  - 声学得分：高（例如，0.88）
  - 语言模型得分：高（例如，0.1）
  - 综合得分：$0.88 \times 0.1 = 0.088$

即使声学得分非常接近，语言模型也扮演了强大的决胜者角色，使“recognize speech”成为明显的胜出者。

> **使用对数概率**
>
> 在实际系统中，将许多小概率相乘可能导致一个称为**数值下溢**的问题，即结果变得如此之小，以至于计算机将其视为零。为避免此问题，系统会使用对数概率。通过取概率的对数，乘法变为加法，这在计算上更快、更稳定。
>
> 公式因此变为：
>
> 
> $$
> \hat{W} = \underset{W}{\mathrm{argmax}} \, (\log P(O|W) + \log P(W))
> $$
> 
>
> 目标不变：找出得分最高的假设。由于概率（一个介于 0 和 1 之间的数字）的对数总是负数，这等同于找出最接近零的得分。

下图说明了解码器如何使用来自两个模型的得分来解决歧义。

> 解码过程示意图。解码器接收两个声学上合理的假设。语言模型为无意义的短语分配了非常低的得分，使解码器能够选择综合得分更高的正确转录。

归根结底，解码器面临的难题是可能的词语序列数量可能极其庞大。它不能简单地测试英语中的每一个可能句子。相反，它必须使用高效的搜索算法来在这个空间中前行并找出最佳路径。我们将在下一节中查看它是如何做到这一点的。

## 参考资料

- [Spoken Language Processing: A Guide to Theory, Algorithm and System Development](https://books.google.com/books/about/Spoken_Language_Processing.html?id=o4iNAAAAMAAJ) — Xuedong Huang, Alex Acero, Hsiao-Wuen Hon (2001)
  Publisher: Prentice Hall PTR; Pages: xxvi, 980 p.
  深入阐述统计语音识别（包括解码过程及底层模型）的经典教科书。
