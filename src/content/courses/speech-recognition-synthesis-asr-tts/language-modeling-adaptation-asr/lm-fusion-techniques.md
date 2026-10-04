---
course: "speech-recognition-synthesis-asr-tts"
chapter: "language-modeling-adaptation-asr"
lesson: "lm-fusion-techniques"
sourceId: 3147
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-3-language-modeling-adaptation-asr/lm-fusion-techniques"
title: "浅层集成与深度集成"
description: "比较解码过程中结合声学模型和语言模型分数的方法。"
order: 2
plots: []
sourceHash: "bd8490d5b8667c59801a7568d40546d127fc59d8a900b4ff9c53dc946d6e4c40"
sourceCorrections: []
---

尽管现代端到端自动语音识别（ASR）系统，例如基于CTC、注意力机制 (attention mechanism)或RNN-Transducer的系统，本身就能从成对的语音-文本数据中学习序列层面的关联，有效包含了一个隐式语言模型，但通过整合一个外部的、显式训练的语言模型（LM），其性能往往能得到大幅提升。这个外部LM通常在大量的纯文本数据上训练，能够捕获在声学训练集中可能稀疏表示的更全面的语言模式、语法结构和特定范围的词汇。

ASR解码的基本目标是找到使给定声学观测序列$X$下的后验概率最大的词序列$\hat{W}$：


$$
\hat{W} = \arg \max_{W} P(W|X)
$$


使用贝叶斯定理，这可以改写为：


$$
\hat{W} = \arg \max_{W} \frac{P(X|W) P(W)}{P(X)}
$$


由于$P(X)$对于给定输入是常数，最大化简化为：


$$
\hat{W} = \arg \max_{W} P(X|W) P(W)
$$


此处，$P(X|W)$是声学模型（AM）分配的概率，$P(W)$是语言模型分配的先验概率。在实践中，解码通常在对数域进行，以实现数值稳定性并将乘积转换为和：


$$
\hat{W} = \arg \max_{W} \log P_{AM}(X|W) + \log P_{LM}(W)
$$


然而，AM和LM对数概率的量级可能显著不同，并且它们的相对重要性会因任务和数据的不同而变化。因此，通常会引入一个可调的插值权重 (weight)$\lambda$：


$$
\hat{W} = \arg \max_{W} \log P_{AM}(X|W) + \lambda \log P_{LM}(W)
$$


通常，还会添加一个由权重$\beta$控制的词插入惩罚或奖励，以管理假设序列的长度：


$$
\hat{W} = \arg \max_{W} \log P_{AM}(X|W) + \lambda \log P_{LM}(W) + \beta \cdot \text{长度}(W)
$$


主要难点在于如何在解码过程中有效结合$\log P_{AM}(X|W)$和$\log P_{LM}(W)$，这通常通过束搜索实现。这种集成主要有两种策略：浅层集成和深度集成。

### 浅层集成

浅层集成可以说是将外部LM集成到端到端ASR系统中最常见、最直接的方法。其核心思想是将AM和LM视为独立模块。AM根据声学信息生成分数（对数概率），LM根据文本历史生成分数。这些分数由解码器*外部*结合，通常在束搜索过程中，以重新排序假设。

**机制：**

1. **独立训练：** AM（例如，基于注意力的编码器-解码器、CTC模型或Transducer）和LM（例如，RNN-LM或Transformer-LM）通常在其各自的数据集上分别训练（AM使用语音-文本对，LM使用纯文本）。
2. **束搜索集成：** 在束搜索过程中，在生成输出序列的每一步，候选假设都会被扩展。对于每个候选假设$W_{prefix}$，解码器会计算一个组合分数，该分数包含AM对下一个标记 (token)/词的预测以及LM对扩展序列的概率。
3. **分数组合：** 如果$y_t$表示AM在步骤$t$预测的标记（字符、子词 (subword)、词），并且$W_{t} = W_{prefix} + y_t$是扩展假设，则分数更新通常如下所示：
   
   $$
   score(W_t) = score(W_{prefix}) + \log P_{AM}(y_t | X, W_{prefix}) + \lambda \log P_{LM}(y_t | W_{prefix})
   $$
   
   （确切的公式取决于AM架构，例如，对于CTC或Transducer，分数在对齐 (alignment)方式上累积不同，但添加加权LM分数的原理不变）。LM分数$\log P_{LM}(y_t | W_{prefix})$通过查询预训练 (pre-training)的LM获得。
4. **分词 (tokenization)处理 (tokenization)：** 如果AM和LM使用不同的标记单位（例如，AM使用字节对编码（BPE）子词，而LM使用词），就会出现一个实际问题。解码器需要逻辑将AM的子词输出映射到词，以便适当查询词级别的LM，通常是累积AM的子词概率，直到形成一个完整的词，然后才添加LM分数部分。

> 浅层集成图示。声学模型和语言模型独立运行，提供分数，这些分数在束搜索解码过程中在外部结合。

**优点：**

- **模块化：** AM和LM可以独立开发、训练和更新。您可以轻松替换不同的LM，而无需重新训练AM。
- **简便性：** 与修改模型架构相比，解码器中的集成逻辑相对简便。
- **效果好：** 通常能显著降低词错误率（WER），特别是当LM是在大型、范围匹配的文本语料库上训练时。

**缺点：**

- **后期集成：** LM只影响主要基于声学证据生成的假设*之间*的选择。它不引导AM本身的内部表示或预测。
- **分词不匹配：** 处理AM/LM分词差异会增加解码器的复杂性。
- **超参数 (parameter) (hyperparameter)调优：** LM权重 (weight)$\lambda$（以及可能的词插入惩罚$\beta$）需要在开发集上仔细调优。

### 深度集成

深度集成试图通过在ASR模型的最终预测层*之前*结合AM和LM的内部表示，实现两者之间更紧密的集成。其想法是让LM提供的语言上下文 (context)能够更直接地影响AM的预测。

**机制：**

1. **架构修改：** 深度集成需要修改ASR模型架构，通常是解码器部分（在基于注意力或Transducer的模型中）。
2. **状态组合：** 在每个解码步骤$t$，ASR解码器的隐藏状态（例如$h_{AM}$）和一个单独计算的LM的隐藏状态（例如$h_{LM}$，源自假设前缀$W_{prefix}$）被组合。
3. **集成函数：** 这种组合可以通过各种函数实现，例如：
   - **拼接：** 拼接$h_{AM}$和$h_{LM}$并将其通过一个线性层：$h_{fused} = W [h_{AM}; h_{LM}] + b$。
   - **门控：** 使用一个门控机制（可能是可学习的）来控制从$h_{AM}$和$h_{LM}$的信息流：$h_{fused} = gate \odot h_{AM} + (1 - gate) \odot h_{LM}$，其中门本身可能取决于$h_{AM}$和$h_{LM}$。
4. **最终预测：** 融合的隐藏状态$h_{fused}$随后被使用，通常通过一个最终线性层和softmax函数，来预测下一个输出标记 (token)$y_t$的概率分布。
   
   $$
   P(y_t | X, W_{prefix}) = \text{softmax}(Linear(h_{fused}))
   $$
   
5. **训练：** 训练可能很复杂。通常，LM在文本数据上预训练 (pre-training)并保持固定（或用小的学习率进行微调 (fine-tuning)）。ASR模型（包括集成组件）随后在语音-文本对上训练。在集成框架内对AM和LM组件进行联合端到端训练也是可能的，但可能难以稳定。

> 深度集成图示。声学模型解码器 ($h_{AM}$) 和语言模型 ($h_{LM}$) 的内部状态在网络架构内部结合（例如，通过集成层或门），然后在最终输出预测之前使用。

**优点：**

- **更紧密的集成：** 来自LM的语言上下文可能在过程中更直接、更早地影响AM的内部表示和概率估计。
- **潜在的更好性能：** 在某些情况下，深度集成可能比浅层集成更好地捕获复杂的AM-LM交互，从而提升准确性。

**缺点：**

- **复杂性增加：** 需要修改核心ASR模型架构，并使系统模块化程度降低。
- **训练上的挑战：** 联合训练或微调这些组件可能很困难，需要仔细平衡梯度和学习率。如果LM被冻结，可能会限制集成机制的适应性。
- **推理 (inference)成本：** 在每一步同步运行LM以获得$h_{LM}$可能会增加推理延迟，相比之下，一些浅层集成实现中LM评分有时可以部分批处理或优化。
- **LM替换：** 替换LM通常需要重新训练至少集成组件，甚至更多。

### 浅层集成与深度集成的选择

浅层集成因其简便性、模块化和已验证的有效性，通常是默认选择。它允许相对轻松地使用独立训练的强大LM。尝试不同的LM并调整$\lambda$权重 (weight)非常直接。

深度集成代表一种更复杂的做法。它可以在以下情况时考虑：

- 浅层集成的结果不足以满足目标应用。
- 存在一种假设，认为语言信息的更紧密、更早集成能显著有利于特定的ASR任务（例如，处理高度模糊的声学输入，这类输入中上下文 (context)早期便很重要）。
- 计算开销和实现复杂性可接受。

值得注意的是，像RNN-Transducer这样的架构本身就执行一种内部预测组合（结合声学编码器输出和预测文本历史），这与深度集成原理共享一些特点，尽管通常没有在状态级别集成一个独立的、大型的外部LM。

最终，浅层集成和深度集成的选择取决于ASR系统的具体需求、可用于实现和训练的资源，以及在相关开发数据上观察到的经验性能。这两种方法都是将重要语言知识融入现代语音识别流程的有价值的技术。

## 参考资料

- [Listen, Attend and Spell: A Neural Network for Large Vocabulary Conversational Speech Recognition](https://arxiv.org/abs/1508.01211) — William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals (2016)
  Journal: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP); Publisher: IEEE; Pages: 4960-4964; DOI: [10.1109/ICASSP.2016.7472659](https://doi.org/10.1109/ICASSP.2016.7472659)
  这篇开创性论文介绍了“听、注意和拼写”（LAS）模型，这是一种基础的端到端ASR架构，并展示了通过浅层融合整合外部语言模型以提高识别性能的有效性。
- [Streaming End-to-End ASR with Shallow-Fused Contextual Biasing](https://doi.org/10.21437/Interspeech.2019-2646) — Wenkuan Fang, Jiahong Yuan, Yu Zhang, and Shinji Watanabe (2019)
  Journal: Interspeech; Publisher: ISCA (International Speech Communication Association); Pages: 2646-2650; DOI: [10.21437/Interspeech.2019-2646](https://doi.org/10.21437/Interspeech.2019-2646)
  本文详细介绍了一种实用的浅层融合方法，用于将外部语言模型与上下文偏置相结合，集成到流式端到端ASR系统中，展示了其对识别性能的影响。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本权威教科书提供了语音识别的全面理论背景，涵盖了声学模型和语言模型的基础概念，以及在解码过程中结合它们的数学框架。
