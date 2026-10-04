---
course: "speech-recognition-synthesis-asr-tts"
chapter: "advanced-acoustic-modeling-asr"
lesson: "advanced-asr-training"
sourceId: 3134
sourceUrl: "https://apxml.com/zh/courses/speech-recognition-synthesis-asr-tts/chapter-2-advanced-acoustic-modeling-asr/advanced-asr-training"
title: "高级训练方法"
description: "正如之前所说，选择RNN-T或Transformer等强大架构是重要的，但训练过程本身也为提高模型性能、鲁棒性和泛化能力提供了很大的空间。简单地训练一个复杂模型…"
order: 6
plots: []
sourceHash: "8f758ddad1d752962cf4ca3130e0ed286e7bd76da9abc39c6c5cc3118e79b271"
sourceCorrections: []
---

选择RNN-T或Transformer等强大架构对于自动语音识别（ASR）系统是基础性的，但训练过程本身也为提高模型性能、鲁棒性和泛化能力提供了很大的空间。简单地用原始数据训练复杂模型通常会导致过拟合 (overfitting)或在多样的实际语音上表现不佳。训练阶段应用的一些高级技术可以缓解这些问题并构建更有效的ASR系统。

### 特征空间的数据增强：SpecAugment

ASR中最有影响力的策略之一，不是作用于原始音频波形，而是直接作用于输入特征，通常是对数梅尔滤波器组能量（频谱图）。SpecAugment是一种常用技术，它对这些特征表示引入变形，迫使模型对语音和噪声的变化保持不变性。

它包含三个主要组成部分，在训练期间依次应用：

1. **时间扭曲：** 在频谱图的时间轴上选择一个随机点。该点之前或之后的所有时间步都向前或向后扭曲随机距离 $w$，该距离从 $[0, W]$ 中均匀选择，$W$ 是一个时间扭曲参数 (parameter)。这模拟了语速的变化。
2. **频率遮蔽：** 连续的梅尔频率通道块 $[f\_0, f\_0 + f)$ 被遮蔽，这意味着它们的值被设为零（或平均值）。频率遮蔽宽度 $f$ 从均匀分布 $[0, F]$ 中选择，起始通道 $f\_0$ 从 $[0, \\nu - f]$ 中选择，其中 $\\nu$ 是梅尔频率通道的总数，$F$ 是最大频率遮蔽宽度参数。这使得模型能够处理频率信息的部分丢失，例如电话线失真或特定噪声类型。
3. **时间遮蔽：** 连续的时间步块 $[t\_0, t\_0 + t)$ 在所有频率通道上被遮蔽。时间遮蔽宽度 $t$ 从 $[0, T]$ 中选择，其起始位置 $t\_0$ 从 $[0, \\tau - t]$ 中选择，其中 $\\tau$ 是输入中的总时间步数，$T$ 是最大时间遮蔽宽度参数。这有助于模型处理时域中的遮挡或短暂噪声爆发。

SpecAugment 在训练期间动态应用，这意味着每个 epoch 模型都会看到输入频谱图的略微不同版本。这显著减少了过拟合 (overfitting)，并提高了对未见过说话者、口音和声学条件的泛化能力，而无需额外的转录数据。参数（$W$, $F$, $T$，以及可能应用的频率和时间遮蔽的数量）是在模型开发过程中调整的超参数 (hyperparameter)。

> SpecAugment 操作应用于输入频谱图后，再将其馈入ASR模型的简化流程。

### 多任务学习 (MTL)

多任务学习不是仅仅在语音到文本转录这一主要任务上训练模型，而是让模型同时执行一个或多个辅助任务。主要观点是这些任务共享表示层（例如，编码器-解码器模型中的编码器），迫使这些层学习对所有任务都有益的特征，从而提高主要任务的泛化能力。

ASR中常见的辅助任务包括：

- **音素识别：** 除了字符或单词外，还预测音素序列。
- **说话者识别：** 从语音中识别说话者身份。
- **性别识别：** 识别说话者的性别。
- **噪声类型分类：** 识别存在的背景噪声类型。
- **语言识别：** 对于多语言系统，识别所说的语言。

**实现方式：** 通常，共享编码器处理输入语音特征。编码器（或中间层）的输出随后被馈入单独的、任务特定的“头部”或解码器。例如，一个头部可以是用于转录的CTC或注意力解码器，而另一个可以是用于说话者识别的简单分类器网络。

总损失函数 (loss function)通常是每个任务损失的加权和：
$L_{总} = w_{主要} L_{主要} + \sum_{i} w_{辅助, i} L_{辅助, i}$
其中 $L\_{主要}$ 是主ASR任务的损失（例如，CTC损失、交叉熵损失），$L\_{辅助, i}$ 是第 $i$ 个辅助任务的损失，而 $w\_{主要}$ 和 $w\_{辅助, i}$ 是平衡每个任务贡献的权重 (weight)。这些权重是重要的超参数 (parameter) (hyperparameter)。如果辅助任务权重过高，它们可能会损害主要任务的性能；如果过低，它们可能无法提供足够的正则化 (regularization)益处。

MTL 作为一种隐式正则化形式，鼓励共享层学习更通用的表示。

### 课程学习

受人类学习方式启发，课程学习涉及以特定顺序向模型呈现训练样本，通常从较简单的例子开始，并逐渐增加难度。“简单”和“困难”的定义可能有所不同：

- **语音长度：** 从较短的语音开始，逐步引入较长的语音。较短的语音通常具有更清晰的对齐 (alignment)路径（对于CTC/RNN-T），或者只需要在较短的上下文 (context)中进行关注。
- **信噪比 (SNR)：** 从干净的语音（高信噪比）开始，逐步添加具有更多背景噪声（低信噪比）的样本。
- **语速：** 首先在语速适中且清晰的语音上训练，然后引入更快或更慢的语音。
- **复杂性度量：** 根据音素复杂性、词汇难度或来自先前模型的对齐置信度分数定义自定义度量。

核心思想是，通过最初专注于更简单的例子，模型可以在处理更具挑战性的数据之前在参数 (parameter)空间中建立一个好的起点。这可以导致更快的收敛速度和可能更好的最终性能，特别是对于复杂模型或优化困难的任务。实施课程学习需要一个机制来根据所选难度度量对训练数据进行排序或分桶，以及一个在训练 epoch 中引入更难例子的时间表。

### 学习率调度与优化器

虽然Adam或带有动量的SGD等标准优化器常用，但在训练过程中有效地调度学习率对于ASR中使用的大型深度学习 (deep learning)模型很重要。简单的固定学习率通常表现不佳。常见策略包括：

- **热身：** 从一个非常小的学习率开始，并在最初的几千次更新或初始 epoch 中逐渐增加。这可以防止训练开始时的大梯度使模型不稳定，特别是对于Transformer等架构。
- **衰减：** 在热身阶段之后（或从初始峰值学习率开始），在训练的其余部分逐渐降低学习率。常见的衰减调度包括线性衰减、指数衰减或余弦退火。这允许模型在接近收敛时进行更精细的调整。
- **组合调度：** 一种流行的方法是“先热身后衰减”，通常是线性热身接线性或余弦衰减。

选择合适的优化器参数 (parameter)（例如，Adam的 $\\beta\_1$, $\\beta\_2$，SGD的动量）和合适的学习率调度，包括热身步数和峰值学习率，是获得最佳结果需仔细调整的重要超参数 (hyperparameter)。

这些高级训练技术补充了之前讨论的架构选择。通过加入SpecAugment、多任务学习和仔细的学习率调度等方法，您可以显著提高ASR模型的鲁棒性和准确性，使其在处理实际语音数据中固有的变异性时更有效。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  介绍了Transformer架构，该架构极大地影响了ASR模型，并描述了其相关的带有预热和衰减的学习率调度。
- [Curriculum Learning](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG8Z5pix83ZxATklqy57SYnwzfotQHm8yQpF_yAyqQhd8gTCNxJo4xBFrkIGPyDnyv3E_F-YHRYoVkRFpV5Vg6kv6buE6t-LaMSK5L9l9R-AhlTwcgjfdobCG4Kf2nMWJ10IZPvT0XQ6Ke3qzzGiPxyYtA=) — Yoshua Bengio, Jérôme Louradour, Ronan Collobert, Jason Weston (2009)
  Journal: Proceedings of the 26th Annual International Conference on Machine Learning - ICML '09; Publisher: ACM Press; Pages: 41-48; DOI: [10.1145/1553374.1553381](https://doi.org/10.1145/1553374.1553381)
  介绍了课程学习概念的基础论文，这是一种将训练样本从易到难排序的训练策略。
- [An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf) — Sebastian Ruder (2017)
  Journal: arXiv preprint arXiv:1706.05098
  一份全面综述，概述了深度神经网络背景下多任务学习的原理、应用和益处。
