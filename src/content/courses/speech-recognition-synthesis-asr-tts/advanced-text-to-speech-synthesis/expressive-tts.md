---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-text-to-speech-synthesis"
lesson: "expressive-tts"
sourceId: 3182
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-4-advanced-text-to-speech-synthesis/expressive-tts"
title: "富有表现力的语音合成"
description: "生成带有不同情感、风格或说话特征的语音。"
order: 6
plots: []
sourceHash: "98405bb7d75e184620cc6d7b83cadae47a5afc93e1334292e24cd6a2da687662"
sourceCorrections: []
---

虽然生成清晰可懂的语音是文本转语音（TTS）系统的基本成果，但要使语音听起来*自然*且*吸引人*，则不能仅仅停留在中性、单调的表达上。富有表现力的语音合成旨在为合成语音注入情感（如喜悦、悲伤、愤怒）、说话风格（如叙述、对话、宣告）或其他人类自然使用的副语言特征。这项能力显著提升了虚拟助手、有声读物叙述、游戏角色配音和辅助技术等应用中的用户体验。

实现语音表现力通常涉及根据所需风格或情感的某种表示来调节TTS模型。我们可以大致将所用方法分类如下：

### 使用明确标签控制表现力

最直接的方法需要使用标有所需表现力类别的训练数据。例如，数据集可能包含标有“高兴”、“悲伤”、“兴奋”或“低语”等标签的录音。

#### 风格嵌入 (embedding)

一种直接的技术是为训练数据中存在的每个预定义风格或情感标签学习一个独特的嵌入向量 (vector)。在训练过程中，TTS模型接收文本输入以及相应的风格嵌入。这个嵌入作为额外的调节信号，影响模型生成的声学特征。

风格嵌入 $e_{style}$ 的常见集成点包括：

1. **与文本嵌入拼接：** 风格嵌入在输入文本编码器之前与输入文本嵌入拼接。
2. **添加到编码器输出：** 风格嵌入（逐元素或通过拼接和投影）添加到文本编码器的输出序列中。
3. **调节解码器：** 风格嵌入被用作初始状态或自回归 (autoregressive)解码器每一步的额外输入（或影响非自回归模型中的时长/音高预测器）。

模型学习将特定的声学特征（音高轮廓、能量水平、语速、频谱特征）与每个风格嵌入关联起来。在推理 (inference)时，您选择与所需风格对应的嵌入来生成富有表现力的语音。

> 一个简化流程图，展示了风格标签如何转换为嵌入，并用于调节TTS声学模型。调节可以在不同点进行，例如影响文本编码器或解码器。

#### 全局风格标记 (token)（GSTs）

与预定义的独热标签不同，全局风格标记提供了一种更灵活、数据驱动的方法。GSTs涉及TTS模型内的一个“风格编码器”模块，该模块直接从训练音频中学习一组具有代表性的风格嵌入（即“标记”），通常以无监督或半监督方式进行。

在训练期间，这个风格编码器将声学特征（如梅尔频谱）作为输入，并使用注意力机制 (attention mechanism)计算学习到的风格标记的权重 (weight)。这些标记的加权和构成话语的风格嵌入。然后，这个嵌入用于调节主TTS模型，类似于上面描述的明确风格嵌入。

主要优势在于模型能从数据本身学习到有意义的风格聚类，可能捕获到预定义标签之外的细节。在推理时，您可以向风格编码器提供参考音频，或直接操作学习到的标记的权重来控制输出风格，尽管控制特定标记可能需要分析或额外的技术。

#### 潜在变量模型（例如，VAEs）

变分自编码器（VAEs）可以用来学习一个表示风格变化的连续潜在空间。一个VAE包含一个编码器和一个解码器。编码器将输入音频（或韵律等派生特征）映射到潜在空间 $z$ 中的一个分布。解码器根据从该潜在空间中抽取的样本重构输入。

在富有表现力的TTS中，VAEs可以在不同种类的语音数据上进行训练。TTS声学模型随后根据从VAE先验分布（通常是标准高斯分布 $N(0, I)$）中抽样的潜在向量 $z$，或通过编码参考音频样本获得的后验分布进行调节。通过操作潜在向量 $z$，这允许对风格进行精细控制和插值。模型学习将潜在空间的不同区域与不同表现特征关联起来。


$$
\text{训练:} \quad z \sim \text{VAE\_编码器}(\text{音频特征}) \\
\text{声学特征}_{\text{目标}} = \text{TTS\_解码器}(\text{文本}, z)
$$


$$
\text{推理（采样）:} \quad z \sim N(0, I) \\
\text{声学特征}_{\text{合成}} = \text{TTS\_解码器}(\text{文本}, z)
$$


$$
\text{推理（重构/风格迁移）:} \quad z = \text{VAE\_编码器}(\text{参考音频特征}) \\
\text{声学特征}_{\text{合成}} = \text{TTS\_解码器}(\text{目标文本}, z)
$$


### 使用参考音频控制表现力

这种方法旨在合成与所提供参考音频话语风格匹配的语音，不一定需要预定义标签。它对于即时模仿特定表达风格特别有用。

#### 风格编码器

与GSTs或VAEs中使用的编码器类似，一个专用的风格编码器网络被训练用于直接从参考音频波形或其频谱图中提取固定维度的风格嵌入 (embedding)。该编码器通常使用RNN、CNN或Transformer等架构，旨在概括相关风格信息（韵律、特征共振峰偏移等），同时理想情况下忽略语音内容和说话人身份（尽管解耦可能具有挑战性）。

从参考音频中提取的风格嵌入 $e_{ref\_style}$ 随后用于在合成目标文本时调节主TTS模型：


$$
e_{ref\_style} = \text{风格编码器}(\text{参考音频}) \\
\text{声学特征}_{\text{合成}} = \text{TTS\_模型}(\text{目标文本}, e_{ref\_style})
$$


这使得“零样本”风格迁移成为可能，即模型可以模仿参考音频样本的风格，即使训练期间没有明确见过该特定风格，前提是风格编码器泛化良好。

### 富有表现力合成的挑战

1. **数据获取：** 获取大量高质量、一致标注的富有表现力的语音数据是一个显著的瓶颈。这通常需要专业配音演员，使其成本高昂且耗时。基于参考的方法减轻了对标签的需求，但仍需要多样化的训练数据以使风格编码器泛化。
2. **评估：** 评估表现力本质上是主观的。虽然客观指标（音高范围、语速）可以提供一些见解，但平均意见分数（MOS）测试仍是黄金标准。设计有效的、评估合成表现力对于给定文本和语境的适当性和自然度的主观测试是很重要的。
3. **解耦：** 使用参考音频时，将风格与风格嵌入 (embedding)中的说话人身份和语音内容分离很困难。解耦不佳可能导致合成语音听起来像参考说话人，或存在与参考文本相关的伪影。
4. **可控性与自然度：** 高度可控的系统（例如，精细的VAE控制）在推向极端时有时可能产生不自然或易产生伪影的结果。平衡控制程度与合成流程的鲁棒性和自然度是一个持续的研究区域。

生成富有表现力的语音为TTS系统增添了丰富性和真实感。通过根据明确标签、参考音频或学习到的潜在表示来调节合成，Tacotron 2、FastSpeech 2和Transformer TTS等模型可以从中性表达中得到提升，使得人机交互更具吸引力且更符合语境。理解这些技术对于构建复杂的TTS应用很重要。

## 参考资料

- [Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis](https://arxiv.org/abs/1806.04558) — Ye Jia, Yu Zhang, Ron J. Weiss, Quan Wang, Jonathan Shen, Fei Ren, Zhifeng Chen, Patrick Nguyen, Ruoming Pang, Ignacio Lopez Moreno, Yonghui Wu (2018)
  Journal: Advances in Neural Information Processing Systems 31 (NeurIPS 2018); Publisher: Neural Information Processing Systems Foundation, Inc. (NeurIPS)
  本文提出了一种稳健的多说话人表达性文本转语音方法，通过训练风格编码器从参考音频中提取固定维度的嵌入，实现零样本风格迁移。
- [Expressive Neural Speech Synthesis by Learning a Disentangled Style Latent Space](https://dl.acm.org/doi/10.1145/3394171.3413988) — Shengqiang Sun, Qinghua Zheng, Fanglei Sun, Yujia Li, Ying Qin (2020)
  Journal: Proceedings of the 28th ACM International Conference on Multimedia (MM '20); Publisher: ACM; Pages: 1274-1282; DOI: [10.1145/3394171.3413988](https://doi.org/10.1145/3394171.3413988)
  探讨使用变分自编码器（VAEs）学习表达性语音合成的解耦潜在空间，从而独立控制各种风格特征。
