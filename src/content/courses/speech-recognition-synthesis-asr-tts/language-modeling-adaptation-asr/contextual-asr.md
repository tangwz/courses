---
course: "speech-recognition-synthesis-asr-tts"
chapter: "language-modeling-adaptation-asr"
lesson: "contextual-asr"
sourceId: 3149
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/contextual-asr"
title: "语境ASR"
description: "使ASR系统偏向语境相关词语或短语的技术。"
order: 3
plots: []
sourceHash: "a962d8d5d74a861e7f9abaef0e9a9b7fc3433a2fe07e8ff42ccc52dbe1b43526"
sourceCorrections: []
---

通用语言模型能捕捉普遍语言模式，但在识别特定用户、对话或情境的词语或短语时常有困难。例如，识别联系人列表中的姓名、项目团队特有的术语，或者你刚刚让智能音箱查找的歌曲名称。这些词语在通用语言模型的训练数据中可能很少出现，即使声学信号清晰，也可能导致识别错误。

语境自动语音识别（ASR）旨在提高对特定用户、对话或领域的词语的识别准确性，例如联系人姓名、项目术语或歌曲标题。传统语言模型通常难以识别这些语境依赖的词语，即使声学信号清晰。语境ASR通过将外部相关信息直接纳入识别过程来应对这一难题。它不同于仅依赖声学输入 $X$ 和词序列的通用概率 $P(W)$ 的识别方法，而是旨在使 $P(W|X, C)$ 最大化，其中 $C$ 表示可用的语境。

### 解码时运用语境

存在多种策略来注入语境信息，它们在复杂性和融入ASR流程的位置上有所不同。

#### 浅层融合与语境偏置 (bias)

一种常见且实用的方法是，在束搜索过程中，使解码器偏向预定义语境列表中存在的词语或短语。这是一种浅层融合，来自不同来源（声学模型、语言模型、语境列表）的信息在得分层面进行组合。

在束搜索中，扩展假设时，候选词 $w$ 的得分通常计算为声学模型得分和语言模型得分的加权和。通过语境偏置，如果词语（或已完成的短语）在语境列表中，则会添加一个额外的奖励项：


$$
score(path) = score_{AM}(X|path) + \lambda \cdot score_{LM}(path) + \beta \cdot score_{Context}(path)
$$


在这里，$score_{Context}(path)$ 提供一个正向偏置，如果 `path` 所代表的词序列（或添加到其中的最后一个词）与语境列表 $C$ 中的一个条目匹配。$\beta$ 是一个控制此偏置强度的超参数 (parameter) (hyperparameter)。

- **实现方式：** 通常涉及在假设语境列表中的词语时添加一个固定奖励 $\beta$。更精细的方法可能仅在形成完整语境短语时应用偏置，或根据先验概率或实体类型使用可变偏置强度。语境通常以简单的字符串列表形式提供。
- **优点：** 相对容易在现有解码器上实现，无需重新训练核心ASR模型。有助于提升特定已知实体（如名称或命令）的识别。
- **缺点：** 需要仔细调整偏置权重 (weight) $\beta$，以避免**过度偏置**（即使语音中不存在也识别出语境词语）。在解码过程中高效处理大型语境列表可能具有难度。对多词短语应用偏置需要在解码器中进行仔细的状态管理。

#### 动态语言模型与WFST调整

另一种方法涉及动态修改语言模型概率或用于解码的加权有限状态转换器（WFST）的结构。

- **工作方式：** 对于给定语境，特定词语或短语的概率可以在LM中暂时提升。在基于WFST的解码器中，这可以涉及即时创建专门的语法WFST，包含高概率（低权重）的语境短语，并将其与主解码图组合。
- **示例：** 设想构建一个仅包含用户联系人列表中姓名的WFST。这个“语境图”可以与主识别图组合，为识别这些特定名称提供低成本路径。
- **优点：** 提供了一种更一体化的方式来在现有解码框架中处理语境。如果WFST构建得当，可以自然地处理多词短语。
- **缺点：** 动态更新LM或WFST可能计算成本高且复杂，特别是对于快速变化的语境或大词汇量系统。可能需要专门的解码器支持。

#### 深度语境化（端到端整合）

近期方法将语境信息直接整合到端到端ASR模型（如基于注意力的编码器-解码器或转换器）的神经网络 (neural network)架构中。这常被称为“深度语境”或“语境聆听”。

- **工作方式：** 语境信息，常表示为嵌入 (embedding)（embeddings），作为额外输入提供给ASR模型，通常是解码器。模型学习在预测输出序列时关注或整合这些语境信息。
  - **语境编码器：** 一个辅助编码器可能将语境（例如，姓名列表、文档文本）处理成固定大小的嵌入向量 (vector)或一组向量。
  - **解码器整合：** 这些语境嵌入可以在每一步与标准解码器输入连接，用于初始化解码器状态，或融入注意力机制 (attention mechanism)（例如，Contextual Listen, Attend and Spell - CLAS）。
- **优点：** 允许模型学习声学、通用语言模式和特定语境之间的复杂相互作用，可能带来优于浅层融合的性能。如果训练得当，可以隐式处理语境词语的变体或相关形式。
- **缺点：** 需要修改模型架构并重新训练。有效表示和注入多种类型语境的机制是一个活跃的研究方向。推理 (inference)可能略微复杂。

> 语境ASR中浅层融合与深度融合的比较。浅层融合在解码时添加偏置得分，而深度融合则将语境嵌入直接整合到神经网络架构中。

#### 后处理重打分

一种更简单但通常效果较差的方法是使用语境对初始ASR解码通过产生的N-best列表进行重打分。

- **工作方式：** 解码器生成多个候选转录。一个独立的模块随后对这些假设进行重新排序，优先选择包含语境列表中词语或短语的假设。
- **优点：** 实现非常简单；将语境化与核心ASR引擎分离。
- **缺点：** 受初始N-best列表质量的限制。如果由于束搜索中的剪枝，正确的语境词语未出现在前N个假设中，则重打分无法找回它。

### 挑战与考量

- **语境表示：** 语境应如何提供？简单的列表很常见，但结构化信息（例如，带有“姓”和“名”字段的联系人条目）或大型文档的嵌入 (embedding)可能更有效。
- **可扩展性：** 即使有数千个语境条目，带有偏置 (bias)或动态图的解码也需要高效。对于浅层融合，通常需要前缀树（tries）或优化的查找结构等技术。
- **发音生成：** 如果一个语境词语对于主系统是真正的OOV（不仅仅是罕见词），ASR系统需要一种方法来生成其发音，通常使用字素到音素（G2P）模型。这个发音随后可以动态添加到词典或WFST中。
- **评估：** 衡量语境化的影响需要包含目标语境现象的特定测试集。不仅要评估语境词语的改进，还要确保通用识别准确性不会受到负面影响（避免过度偏置）。

“语境ASR是一种重要技术，可使语音识别系统在应用中更实用和准确，使其能够动态适应与当前情况或用户相关的特定实体和术语。”

## 参考资料

- [Large-Scale Contextual Speech Recognition with a Recurrent Neural Network Language Model](https://doi.org/10.21437/Interspeech.2018-1241) — Anjishnu Kumar, Ganesh Venkatesh, Kevin Kilgour, Taras Halych, Ruoming Pang, Bo Li, Ron Weiss, Jason Riesa, Pedro J. Moreno (2018)
  Journal: Interspeech 2018; Publisher: International Speech Communication Association (ISCA); Pages: 1241-1245; DOI: [10.21437/Interspeech.2018-1241](https://doi.org/10.21437/Interspeech.2018-1241)
  本文介绍了一种在语境ASR中进行浅层融合的方法，通过使用特定语境列表对循环神经网络语言模型进行偏置。
- [Contextual Listen, Attend and Spell](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/1784.pdf) — Etienne Bruguier, Olivier Siohan, Anjishnu Kumar, George Tucker (2019)
  Journal: Interspeech 2019; Publisher: ISCA (International Speech Communication Association); Pages: 4545-4549; DOI: [10.21437/Interspeech.2019-1784](https://doi.org/10.21437/Interspeech.2019-1784)
  本文详细介绍了“Contextual Listen, Attend and Spell (CLAS)”模型，这是一种将上下文融入基于注意力的端到端ASR系统的特定神经网络方法。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University; Pages: Chapter 7, Chapter 8
  一本内容广泛的教材，为语音识别提供了宽泛的理论基础，包括语言模型概念和自适应策略。
