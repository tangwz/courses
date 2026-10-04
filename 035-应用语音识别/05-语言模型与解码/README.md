# 第 5 章：语言模型与解码

来源：[原章节](https://apxml.com/zh/courses/applied-speech-recognition/chapter-5-language-modeling-decoding)

[返回课程目录](../README.md)

在前面的章节中，我们构建了将音频特征映射到字符概率序列的声学模型。虽然这些模型在识别语音内容方面很有效，但它们的输出可能在声学上听起来合理，但在语言上不正确。例如，模型可能会将“recognize speech”转录为发音相似的“wreck a nice beach”。

这就是引入语言模型 (LM) 的地方。语言模型根据其语法结构和出现可能性来评估一个词语序列，帮助系统区分合理与不合理的转录。使用语言模型指导从声学模型预测中选择最终文本的过程称为**解码**。解码器的目标是找到使组合得分最大化的词语序列 $W$，这通常是声学和语言模型概率的加权和：

$$
\text{score}(W) = \log P_{\text{Acoustic}}(X|W) + \alpha \log P_{\text{Language Model}}(W)
$$

这里，$P_{\text{Acoustic}}(X|W)$ 是声学模型在给定词语序列 $W$ 时分配给音频特征 $X$ 的概率。$P_{\text{Language Model}}(W)$ 项是词语序列本身的概率，而 $\alpha$ 是一个平衡两种模型影响的权重。

本章讲述了将语言模型整合到 ASR 系统中的理论与实践。您将学习：

*   阐述语言模型在 ASR 流程中的作用。
*   使用 KenLM 工具包从文本语料库构建统计 N-gram 语言模型。
*   比较简单的贪心搜索解码与更有效的束搜索算法。
*   实现一个束搜索解码器，该解码器引入外部语言模型的评分以提高转录准确性。

## 小节

- 1. [语言模型在自动语音识别中的作用](01-%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9C%A8%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
- 2. [N-gram 语言模型](02-N-gram%20%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B.md)
- 3. [使用 KenLM 构建 N-gram 模型](03-%E4%BD%BF%E7%94%A8%20KenLM%20%E6%9E%84%E5%BB%BA%20N-gram%20%E6%A8%A1%E5%9E%8B.md)
- 4. [模型整合的解码图](04-%E6%A8%A1%E5%9E%8B%E6%95%B4%E5%90%88%E7%9A%84%E8%A7%A3%E7%A0%81%E5%9B%BE.md)
- 5. [解码算法：贪心搜索与集束搜索对比](05-%E8%A7%A3%E7%A0%81%E7%AE%97%E6%B3%95%EF%BC%9A%E8%B4%AA%E5%BF%83%E6%90%9C%E7%B4%A2%E4%B8%8E%E9%9B%86%E6%9D%9F%E6%90%9C%E7%B4%A2%E5%AF%B9%E6%AF%94.md)
- 6. [结合语言模型实现束搜索](06-%E7%BB%93%E5%90%88%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%AE%9E%E7%8E%B0%E6%9D%9F%E6%90%9C%E7%B4%A2.md)
- 7. [动手实践：将语言模型集成到 CTC 解码器中](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%86%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E9%9B%86%E6%88%90%E5%88%B0%20CTC%20%E8%A7%A3%E7%A0%81%E5%99%A8%E4%B8%AD.md)
