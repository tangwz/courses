# 第 4 章：高级声学模型与架构

来源：[原章节](https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models)

[返回课程目录](../README.md)

上一章中，我们使用LSTM和连接时序分类（CTC）损失函数构建了声学模型。尽管CTC是一种可行的方法，但它做了一个强假设，即每个时间步的输出预测是条件独立的。本章将介绍现代架构，这些架构直接建模输出字符或词语间的依赖关系，从而带来更准确、考虑语境的转录。

您将从注意力机制开始学习，它允许模型在生成转录本的每个部分时，选择性地关注输入音频中的相关片段。在此基础上，您将了解序列到序列（Seq2Seq）架构，例如Listen, Attend, and Spell (LAS) 模型，这些模型能将输入音频序列直接映射到输出文本序列。

随后，我们将介绍Transformer架构及其在音频处理中自注意力机制的使用。您将了解到这种设计如何捕捉整个音频输入中的依赖关系。我们还将考察Conformer模型，它是一种混合架构，结合了用于局部特征提取的卷积和Transformer的全局语境建模能力。

本章最后将概述Wav2Vec 2.0等大型预训练ASR模型。您将学习如何在特定数据集上微调这些模型，这是一种获得高表现的常用且有效方法。实践部分将指导您完成从Hugging Face库微调预训练模型的过程，让您直接体验领先的ASR工作流程。

## 小节

- 1. [用于语音识别的注意力机制](01-%E7%94%A8%E4%BA%8E%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
- 2. [自动语音识别 (ASR) 中的序列到序列 (Seq2Seq) 模型](02-%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%20%28ASR%29%20%E4%B8%AD%E7%9A%84%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%20%28Seq2Seq%29%20%E6%A8%A1%E5%9E%8B.md)
- 3. [听觉、注意与拼写 (LAS) 架构](03-%E5%90%AC%E8%A7%89%E3%80%81%E6%B3%A8%E6%84%8F%E4%B8%8E%E6%8B%BC%E5%86%99%20%28LAS%29%20%E6%9E%B6%E6%9E%84.md)
- 4. [自动语音识别中的 Transformer 模型概述](04-%E8%87%AA%E5%8A%A8%E8%AF%AD%E9%9F%B3%E8%AF%86%E5%88%AB%E4%B8%AD%E7%9A%84%20Transformer%20%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
- 5. [Conformer：结合卷积神经网络与Transformer](05-Conformer%EF%BC%9A%E7%BB%93%E5%90%88%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8ETransformer.md)
- 6. [预训练ASR模型概述](06-%E9%A2%84%E8%AE%AD%E7%BB%83ASR%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0.md)
- 7. [实践：微调预训练ASR模型](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BE%AE%E8%B0%83%E9%A2%84%E8%AE%AD%E7%BB%83ASR%E6%A8%A1%E5%9E%8B.md)
