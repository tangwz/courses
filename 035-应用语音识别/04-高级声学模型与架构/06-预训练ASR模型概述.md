# 预训练ASR模型概述

来源：[原文](https://apxml.com/zh/courses/applied-speech-recognition/chapter-4-advanced-acoustic-models/overview-pretrained-asr-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

从头构建大型Transformer或Conformer模型可以取得显著成果，但这种方式需要两种稀缺资源：海量的标注训练数据和庞大的计算能力。对于许多组织和开发者来说，在这些限制下创建高质量的ASR系统是不切实际的。这一挑战促使该领域向使用预训练 (pre-training)模型发生了重大转变。

这些模型背后的核心思想是：首先在一个不需要手动创建标注的通用、大规模任务上训练一个大型神经网络 (neural network)，然后将这个已学习的模型应用于特定的、较小的标注数据集。这种两阶段过程，被称为预训练和微调 (fine-tuning)，已成为在ASR中实现优异性能的标准做法。

### 语音自监督学习 (supervised learning) (self-supervised learning)的兴起

促成语音预训练 (pre-training)的创新是**自监督学习（SSL）**。与需要成对输入和输出（音频及其文本）的监督学习不同，自监督学习直接从输入数据生成自己的标注。这使得模型能够从大量未标注音频中学习到丰富且有意义的表征，而未标注音频比已转录音频更易获得。

一般做法是向模型提供音频样本的修改版本，并训练它预测原始、未修改的版本。通过解决这个人工设定的问题，模型被迫学习人类语音的底层结构，例如语音学、协同发音和韵律，而无需看到任何文本标注。

### Wav2Vec 2.0：从原始音频中学习

在语音方面最具影响力的自监督模型之一是`Wav2Vec 2.0`。它直接从原始音频波形中学习强大的语音表征。其架构由三个主要部分组成：

1. **特征编码器：** 一个多层卷积神经网络 (neural network)（CNN）处理原始音频波形。它的任务是分割音频并将其转换为一系列潜在特征表征，通常以每25毫秒生成一个向量 (vector)的速率。
2. **量化 (quantization)模块：** 来自编码器的连续特征向量被离散化为有限的“语音单元”集合。这一步对自监督任务很重要，因为它将预测问题简化为分类任务。
3. **语境化器：** 一个大型Transformer网络接收特征表征序列作为输入。在预训练 (pre-training)期间，其中一些输入向量被遮蔽（置零）。Transformer的目标是利用周围语境预测被遮蔽位置的正确量化语音单元。

这一过程类似于BERT等文本模型中的掩码语言建模。通过学习填补音频中的空白，Transformer的语境化器变得非常擅长理解语音话语不同部分之间的关系。

### 从预训练 (pre-training)到微调 (fine-tuning)

模型在数千小时的未标注音频上预训练完成后，可以将其适配特定ASR任务，例如转录英语电话通话。这第二个阶段被称为**微调**。

过程简单明了：

1. **加载预训练模型：** 使用`Wav2Vec 2.0`模型及其自监督阶段学习到的权重 (weight)。量化 (quantization)模块被舍弃。
2. **添加分类头：** 在Transformer之上添加一个新的、随机初始化的线性层。该层将Transformer的输出表征映射到你的目标词汇（例如，字符或音素）的概率。
3. **在标注数据上训练：** 使用标准的ASR损失函数 (loss function)（例如CTC损失）在你的较小标注数据集上训练整个模型。由于模型的核心层已经理解语音，它能非常快速有效地学习到到文本字符的映射。

微调阶段所需数据和计算量比从头训练显著减少，但它持续产生优异结果。

> 构建现代ASR系统的两阶段工作流程。首先，模型从未标注数据中学习通用语音特征。其次，该预训练模型使用较小规模的标注数据适配特定转录任务。

### 其他重要预训练 (pre-training)模型

虽然`Wav2Vec 2.0`是一个基础模型，但该领域持续演进。以下是另外几个需要了解的重要模型：

- **HuBERT (Hidden-Unit BERT)：** 与`Wav2Vec 2.0`在思想上相似，`HuBERT`也使用掩码预测任务。其主要区别在于它如何为被遮蔽的步骤生成目标标注。它使用离线聚类步骤首先发现离散隐藏单元，使学习目标在训练期间更具一致性。
- **Whisper：** 由OpenAI开发，`Whisper`模型代表了一种不同的方法。它们并非纯粹的自监督，而是在一个庞大且多样化的数据集上以“弱监督”方式训练，该数据集包含来自网络的680,000小时音频，且已与文本配对。由于这些数据涵盖多种语言、主题、口音和声学环境，`Whisper`模型极其强大，无需任何微调 (fine-tuning)即可在各种任务上表现良好。它们是多语言和多任务的，能够执行转录和翻译。

使用`Wav2Vec 2.0`、`HuBERT`或`Whisper`等预训练模型极大地降低了构建高质量ASR系统的进入门槛。在即将进行的实践部分，你会看到这种方法有多么有效，我们将为自定义语音识别任务微调来自Hugging Face Hub的预训练模型。

## 参考资料

- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) — Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed (2021)
  Journal: IEEE/ACM Transactions on Audio, Speech, and Language Processing; Volume: 30; Pages: 1191-1205; DOI: [10.48550/arXiv.2106.07447](https://doi.org/10.48550/arXiv.2106.07447)
  这篇论文介绍了HuBERT，这是一种替代的自监督方法，它使用离线聚类步骤生成离散的隐藏单元作为掩码预测的目标，为语音表征学习提供了一种不同的方案。
- [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf) — Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever (2022)
  Publisher: OpenAI; DOI: [https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)
  这份报告介绍了Whisper模型系列，该系列通过在海量多样的音频-文本对数据集上进行弱监督训练，实现了优异性能，具备多语言和多任务能力。

---

[上一节](05-Conformer%EF%BC%9A%E7%BB%93%E5%90%88%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%8ETransformer.md) · [下一节](07-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%BE%AE%E8%B0%83%E9%A2%84%E8%AE%AD%E7%BB%83ASR%E6%A8%A1%E5%9E%8B.md)
