# ASR/TTS 的知识蒸馏

来源：[原文](https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-6-optimization-deployment-toolkits/knowledge-distillation-speech)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然量化 (quantization)和剪枝等技术直接修改模型的结构或权重 (weight)以降低复杂度，但知识蒸馏 (knowledge distillation)（KD）提供了一种不同的方法。它侧重于将大型、复杂模型（“教师模型”）的能力迁移到更小、更高效的模型（“学生模型”）中。这使我们能够创建明显更快、更轻量且可部署的语音识别（ASR）和语音合成（TTS）模型，同时保留其大型对应模型的大部分性能。核心思想是，教师模型通过在大量数据集上的训练，学到了丰富的表示和决策边界，而较小的学生模型仅使用真实标签可能难以独立习得这些内容。

### 师生模型方法

知识蒸馏 (knowledge distillation)基于一个简单原理：训练一个紧凑的学生模型来模仿预训练 (pre-training)、高性能教师模型的行为。

1. **教师模型：** 这通常是一个先进的ASR或TTS模型，具有高准确度但计算成本也高。它可能是一个深度Transformer、一个大型Conformer或一个复杂序列到序列架构。其参数 (parameter)在蒸馏过程中通常是固定的。
2. **学生模型：** 这是旨在用于部署的模型。它的架构明显更小（层数更少，隐藏维度更小，不同模型类型如CNN/RNN而非Transformer），设计时考虑了推理 (inference)速度、内存占用和功耗方面的效率。

目标是使学生模型尽可能地复现教师模型的输出函数，有效地将教师模型所学知识“蒸馏”到较小的学生网络中。

### 知识迁移：软目标与特征匹配

学生如何向教师学习？最常用的方法是匹配教师模型的输出或内部表示。

#### 软目标（Logit 匹配）

知识蒸馏 (knowledge distillation)（KD）通常不只是单独使用“硬”真实标签（例如，表示正确单词或音素的独热向量 (vector)）来训练学生模型，而是使用教师模型生成的完整概率分布作为“软”目标。理由是教师模型的输出分布包含的信息比单一正确标签更多；它呈现出教师模型如何理解不同类别*之间*的关系（例如，哪些不正确的音素在声学上与正确的音素相似）。

为促进这种信息迁移，教师模型和学生模型的输出（logits，$z$）在计算蒸馏损失之前，通常会在softmax函数中使用一个温度参数 (parameter)（$T > 1$）进行软化：


$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$


更高的温度 $T$ 会在类别上产生更平滑的概率分布，放大较小的logit值，从而为学生模型提供更丰富的监督信号。

然后训练学生模型以最小化一个损失函数 (loss function)，该函数促使其软化预测与教师模型的软化预测匹配。常用选择是 Kullback-Leibler (KL) 散度：


$$
L_{KD} = T^2 \cdot KL( \text{softmax}(z_{\text{学生}}/T) || \text{softmax}(z_{\text{教师}}/T) )
$$


$T^2$ 缩放因子确保在温度软化输出时，梯度幅度保持恰当。

通常，这个蒸馏损失会与标准损失函数（例如，分类任务的交叉熵，或频谱图预测等回归任务的均方误差）结合使用，后者是利用硬真实标签（$y_{\text{真实}}$）和学生模型的标准（非软化，$T=1$）预测（$p_{\text{学生}}$）计算的。一个加权因子 $\alpha$ 平衡这两个目标：


$$
L_{\text{总}} = \alpha L_{\text{标准}}(p_{\text{学生}}, y_{\text{真实}}) + (1 - \alpha) L_{KD}
$$


这鼓励学生模型既要模仿教师行为，又要正确预测真实值。

> 使用软目标的典型知识蒸馏设置。学生模型学习模仿大型教师模型的软化输出，通常还会通过标准损失函数从原始硬标签中学习。

#### 中间表示匹配（特征匹配）

知识也可以存在于教师模型内部的中间激活或隐藏状态中。知识蒸馏可以涉及训练学生模型去匹配这些特定层级的内部表示。例如，学生模型在第 $l$ 层的隐藏状态（$h_{\text{学生}}^{(l)}$）可以通过最小化均方误差（MSE）或其他距离度量来训练，使其与教师模型在对应层（或如果维度不同，则为其投影）的隐藏状态（$h_{\text{教师}}^{(l)}$）相匹配：


$$
L_{\text{特征}} = \sum_{l \in L_{\text{匹配}}} || f_{\text{投影}}(h_{\text{学生}}^{(l)}) - h_{\text{教师}}^{(l)} ||_2^2
$$


这里，$L_{\text{匹配}}$ 是选定用于匹配的层集合，$f_{\text{投影}}$ 是一个可选的投影层（例如，线性层）以匹配维度。这种方法有助于迁移教师模型学到的结构知识、注意力模式或中间声学/语言特征。

### 在语音识别中的应用

知识蒸馏 (knowledge distillation)被广泛用于构建高效的语音识别模型：

- **声学模型：** 在数千小时数据上训练的大型教师模型（例如，Conformer, RNN-T, 基于注意力的编码器-解码器）可以蒸馏成更小的学生模型（例如，更小的Conformer, LSTM, GRU, CNN）。学生模型在教师模型的软预测引导下，学习预测上下文 (context)相关的音素状态或字符。编码器层的特征匹配也能提升学生模型的性能。
- **语言模型：** 用于重评分语音识别假设的基于Transformer的大型语言模型计算成本高昂。知识蒸馏可以训练更小的语言模型（如BERT或GPT的蒸馏版本）来近似大型语言模型的评分行为，从而实现更快的二遍重评分，甚至通过浅层/深层融合集成到一遍解码中。

### 在语音合成中的应用

同样，知识蒸馏 (knowledge distillation)有助于构建快速、高质量的语音合成系统：

- **声学模型（频谱图预测）：** 缓慢但高质量的自回归 (autoregressive)模型，如 Tacotron 2 或 Transformer TTS，可作为优秀的教师模型。它们的知识可以蒸馏到快速、并行的非自回归学生模型中，如 FastSpeech 2 或 ParaNet。学生模型学习基于输入文本预测梅尔频谱图，常使用教师模型预测的频谱图作为目标（与真实值相比，有时被认为是“软”目标）。从教师模型蒸馏注意力对齐 (alignment)或时长预测也常见。
- **声码器：** 将声学特征（如梅尔频谱图）转换为音频波形可能是一个瓶颈。缓慢、高保真的自回归声码器（如 WaveNet）或复杂的流/扩散基声码器可以蒸馏到快得多的基于GAN的声码器（如 HiFi-GAN, MelGAN）或并行WaveNet变体中。学生声码器学习生成感知上与教师模型生成的波形相似的波形，通常通过匹配输出分布或中间特征。

### 实际考量

- **教师模型质量：** 被蒸馏的学生模型性能通常受限于教师模型的质量。更好的教师模型通常会带来更好的学生模型。
- **学生模型架构：** 学生模型架构的选择很重要。它必须比教师模型明显更高效，但仍具备学习蒸馏知识的足够能力。
- **蒸馏策略：** 在软目标、特征匹配或两者结合之间选择，取决于特定任务和模型。调整超参数 (parameter) (hyperparameter)（如温度 ($T$) 和损失权重 (weight) ($\alpha$)）是必需的。
- **自蒸馏：** 一种有趣的变体，其中教师模型和学生模型拥有*相同*的架构。训练一个模型来模仿其自身的软化预测（来自之前的检查点或平均权重），有时能提升泛化能力和鲁棒性，与标准训练相比。
- **数据：** 知识蒸馏 (knowledge distillation)通常使用与教师模型相同的训练数据，但有时也可以利用未标记 (token)数据，因为蒸馏损失项只需要教师模型的预测。

总而言之，知识蒸馏是语音处理优化工具包中一种强大且多用途的技术。通过将知识从大型、复杂的模型迁移到更小、高效的模型中，它使得高性能的语音识别和语音合成系统得以在资源受限的环境中部署，补充了量化 (quantization)和剪枝等方法。

## 参考资料

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Geoffrey Hinton, Oriol Vinyals, Jeff Dean (2015)
  Journal: NIPS 2014 Deep Learning Workshop; DOI: [10.48550/arXiv.1503.02531](https://doi.org/10.48550/arXiv.1503.02531)
  这篇开创性论文介绍了知识蒸馏技术，详细阐述了如何使用软目标和温度缩放将知识从大型教师模型迁移到小型学生模型。
- [Knowledge Distillation: A Survey](http://dx.doi.org/10.1007/s11263-021-01453-z) — Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao (2021)
  Journal: International Journal of Computer Vision; Publisher: Springer Science and Business Media LLC; Volume: 129; Pages: 1789-1819; DOI: [10.1007/s11263-021-01453-z](https://doi.org/10.1007/s11263-021-01453-z)
  这是一篇关于知识蒸馏方法、理论及其在不同领域（包括语音）应用的全面综述。
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09292) — Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: Neural Information Processing Systems; Volume: 32; DOI: [10.48550/arXiv.1905.09292](https://doi.org/10.48550/arXiv.1905.09292)
  这篇论文提出了一种使用知识蒸馏从自回归教师模型训练的非自回归文本到语音模型，实现了显著的速度提升并保持了高质量。

---

[上一节](02-%E6%A8%A1%E5%9E%8B%E5%89%AA%E6%9E%9D%E4%B8%8E%E7%A8%80%E7%96%8F%E5%8C%96.md) · [下一节](04-%E4%BC%98%E5%8C%96%E6%8E%A8%E7%90%86%E5%BC%95%E6%93%8E%EF%BC%88ONNX%20Runtime%2C%20TensorRT%EF%BC%89.md)
