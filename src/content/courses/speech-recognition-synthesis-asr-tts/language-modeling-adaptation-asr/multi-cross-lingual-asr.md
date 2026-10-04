---
course: "speech-recognition-synthesis-asr-tts"
chapter: "language-modeling-adaptation-asr"
lesson: "multi-cross-lingual-asr"
sourceId: 3159
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/multi-cross-lingual-asr"
title: "多语言和跨语言自动语音识别"
description: "构建处理多种语言或利用其他语言数据的系统的方法。"
order: 7
plots: []
sourceHash: "3a72955049725d7cd2595780c35023c980a1a3eb9885dc58bca13d84452f47a5"
sourceCorrections: []
---

处理语言多样性是自动语音识别（ASR）中的一个主要挑战。构建高性能ASR系统通常需要为每种目标语言提供大量转录音频数据。然而，为每种语言开发单独模型效率不高，且通常不可行，特别是对于资源有限的语言。多语言和跨语言ASR方法提供了构建更通用和数据高效系统的策略。

**多语言ASR** 旨在创建一个能识别两种或更多语言语音的单一ASR系统。这对于服务不同人群或处理语码转换（在同一个话语中混合语言）的应用特别有用，尽管语码转换仍是一个重要难题。

**跨语言ASR** 侧重于借助资源丰富语言（源语言）的数据，以提升数据有限语言（目标语言）的ASR性能。这是一种跨越语言界限的迁移学习 (transfer learning)形式。

### 多语言ASR方法

多语言ASR的核心思想通常是模型架构中跨语言的资源共享。以下是常见策略：

1. **数据合并：** 最简单的方法是将所有目标语言的训练数据合并，并训练一个单一模型。尽管直接，但这需要仔细处理输出层。如果使用音素，可以使用涵盖所有语言的统一音素集（例如国际音标IPA，或自定义映射）。对于基于字符或子词 (subword)的端到端模型（CTC、RNN-T、Attention），所有语言的输出单元（字符、SentencePiece标记 (token)）会合并成一个单一输出词表。性能可能对训练数据中的语言不平衡敏感，这可能偏向于高资源语言。
2. **带有语言识别（LID）的共享组件：** 为明确引导模型，可以加入语言身份信息。

   - **语言ID输入：** 语言标识符（例如，独热向量 (vector)或习得的语言嵌入 (embedding)）可以作为额外的输入特征提供给声学编码器或解码器。这会告知模型预期语言。
   - **分层模型：** 一些架构首先预测语言，然后使用特定语言的组件（如解码器或输出层）进行转录。
   - **示例结构：** 常见的架构包括一个共享的声学编码器，它不分语言地处理音频特征，然后是解码前或解码期间引入语言身份的机制。

> 显示多语言ASR架构的图表，其中包含一个共享声学编码器和解码器中特定语言的条件化（使用语言ID）。

3. **共享音素表示：** 如果使用混合HMM-DNN系统或预测音素目标的模型，将特定语言的音素映射到共享音素空间可以促进知识转移。声音相似的语言获益最大。然而，定义一个真正通用且有效的共享音素集具有挑战性，并且不同音素库之间可能发生干扰。

这些方法的有效性取决于语言数量、它们的类型学相似性（例如，语音重叠、语法结构）以及每种语言的可用数据量等因素。

### 跨语言ASR方法

为低资源语言构建ASR系统时，借助高资源语言的数据可以明显提升性能，相较于仅用有限的目标数据训练。

1. **通过微调 (fine-tuning)进行迁移学习 (transfer learning)：** 这可以说是最常见的方法。

   - **预训练 (pre-training)：** 使用一个或多个源语言的大型数据集来训练ASR模型（通常是声学编码器或整个端到端模型）。
   - **微调：** 使用目标低资源语言的可用转录数据调整预训练模型。相关决定包括：
     - 哪些层要冻结（保持与预训练时不变）以及哪些层要微调。通常，编码器中捕获通用声学模式的较低层会被冻结或以较小学习率微调，而较高层和解码器则会更积极地进行调整。
     - 是否调整输出层。如果源语言和目标语言使用不同的文字系统或音素集，最终输出层通常需要替换或大幅度调整。使用在合并数据上训练的子词 (subword)单元（如BPE或WordPiece）有时可以缓解这种情况。
2. **多语言模型作为预训练：** 不在单一源语言上预训练，而是在多个高资源语言上预训练一个多语言模型（如上所述）。这可以提供更通用的声学表示，有助于后续在低资源目标上的微调。
3. **共享表示：** 类似于多语言ASR，在一个合并的高资源和低资源数据上训练模型，鼓励模型学习共享的底层表示，通过接触更多样化的声学现象来帮助低资源语言。
4. **辅助任务：** 在源语言预训练期间使用辅助任务，例如预测发音特征或预期更具语言独立性的声学属性，有时可以提高可迁移性。

跨语言迁移的成功通常取决于源语言和目标语言之间的声学和语音相似性。从英语到德语（均为日耳曼语言）的迁移通常比从英语到普通话（类型学上差异很大）更有效。

### 架构考量与评估

现代架构如Transformer非常适合多语言和跨语言任务。它们的自注意力 (self-attention)机制 (attention mechanism)可能学习跨语言的声学-音素关联。语言嵌入 (embedding)可以轻易地加入到输入给Transformer编码器或解码器的序列中。

相较于传统的混合系统，端到端模型简化了跨语言调整，因为调整独立的声学、发音和语言模型更为复杂。然而，确保输出层（例如，CTC或RNN-T模型的词表）恰当处理多种文字系统或音素集需要细致的设计，通常倾向于共享子词 (subword)分词 (tokenization)。

评估多语言系统通常涉及单独测量每种支持语言的性能（例如，词错误率WER）。对于跨语言ASR，主要衡量标准是在应用迁移技术后，目标低资源语言上的性能。

构建能够跨越语言界限的系统是一个活跃的研究方向，推动更普遍的语音处理能力，使ASR技术适用于更广泛的语言。这些技术是减少全球构建语音识别系统数据瓶颈的重要步骤。

## 参考资料

- [Massively Multilingual Speech Recognition with a Single Transformer](https://www.isca-speech.org/archive/interspeech_2020/chung20_interspeech.html) — Yu-An Chung, Wei-Ning Hsu, Alexis Conneau, and James Glass (2020)
  Journal: Interspeech 2020; Publisher: International Speech Communication Association (ISCA); Pages: 3775-3779; DOI: [10.21437/Interspeech.2020-2212](https://doi.org/10.21437/Interspeech.2020-2212)
  一篇研究论文，展示了单一Transformer模型在多语言ASR中的有效性，内容涵盖共享架构和数据池概念。
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Volume: 33; Pages: 12449-12460; DOI: [10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477)
  本文介绍了一种广泛采用的自监督预训练方法，它是通过对有限目标语言数据进行微调，实现现代跨语言ASR的基础。
- [Unsupervised Cross-Lingual Representation Learning for Speech Recognition](https://arxiv.org/abs/2006.13979) — Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli (2020)
  Journal: arXiv preprint arXiv:2006.13979; DOI: [10.48550/arXiv.2006.13979](https://doi.org/10.48550/arXiv.2006.13979)
  本文将wav2vec 2.0框架专门扩展到无监督跨语言语音表示学习，为低资源ASR带来了显著改进。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本全面的教科书，对语音识别原理进行了透彻介绍，涵盖了传统方法和深度学习的近期进展。相关章节讨论了语言建模和自适应技术。
