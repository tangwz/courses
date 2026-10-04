---
course: "rnns-and-sequence-modeling"
chapter: "introduction-sequential-data"
lesson: "common-sequence-modeling-tasks"
sourceId: 2523
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-1-introduction-sequential-data/common-sequence-modeling-tasks"
title: "常见的序列建模任务"
description: "序列预测、分类和生成等任务的概览。"
order: 5
plots: []
sourceHash: "43911b6010ed2ce05e2fd97480d084110e6b99de5377b7583b7d6230a365f1ba"
sourceCorrections: []
---

既然我们已明确了序列数据的特性以及标准前馈网络的局限，现在来看看序列模型旨在解决的各类问题。了解这些常见任务能为我们后续学习的架构和技术提供背景。序列建模问题通常分为几大类，主要区别在于它们的输入和输出结构。

## 序列预测与预报

最直接的用途也许是预测序列中的后续内容。给定一系列观测值 $x_1, x_2, ..., x_t$，目标是预测 $x_{t+1}$ 的值，或者预测一系列未来值 $x_{t+1}, ..., x_{t+k}$。

"这类包含许多应用："

- **时间序列预报：** 根据历史趋势预测未来股价，根据过去气象数据预报天气模式，或根据之前的消耗估计能源需求。
- **下一步预测：** 在自然语言处理 (NLP) 中，这可能涉及为自动补全预测句子中的下一个词，或在推荐系统中建议用户可能交互的下一个项目。

主要理念是学习序列内部的时间关联，从而对未来做出明智的预测。根据您是预测单个下一步还是多个步骤，这可视为“多对一”（从多个输入预测一个输出）或“多对多”（从多个输入预测多个输出）的问题结构。

> 依据前序序列 ($x_1, x_2, x_3$) 预测下一个元素 ($x_4$) 的流程。每个步骤处理一个输入并更新表示序列历史的内部状态。

## 序列分类

在序列分类中，目标是为整个输入序列分配一个单一的类别标签。我们不是预测下一个元素，而是希望理解整个序列所代表的整体含义或属性。

常见例子有：

- **情感分析：** 将文本评论（词语序列）分类为积极、消极或中性。
- **主题识别：** 根据文章或文档的文本来确定其主题。
- **活动识别：** 根据可穿戴设备的传感器读数序列来识别一项活动（如行走、跑步、坐立）。
- **垃圾邮件检测：** 根据电子邮件中包含的词语或字符序列对其进行分类。

通常，模型处理整个序列 $x_1, ..., x_T$，然后生成一个单一的输出分类 $y$。这符合“多对一”的结构，即多个输入映射到一个单一的类别输出。

> 对整个序列 ($x_1$ 到 $x_T$) 进行分类的流程。信息按顺序处理，最终的分类判断基于整个序列汇总的信息做出。

## 序列生成

序列生成涉及依据学习到的特定模式创建新的序列。与预测不同，预测的目标通常是*下一步*，而生成则常旨在生成更长、连贯的序列。

应用包括：

- **文本生成：** 创建类似人类的文本，例如写作辅助、为聊天机器人生成对话，或创作诗歌或代码。
- **音乐生成：** 以特定风格创作新颖的音乐作品。
- **图像描述生成：** 为给定图像生成文本描述（词语序列）（图像常被视为单一输入上下文 (context)）。

生成模型学习训练数据中序列的概率分布。它们可以从该分布中抽样以创建新序列。这有时可以从一个“种子”输入（例如一个句子的开头或一个音乐乐句）开始，然后继续一个接一个地生成元素。此任务可涉及“一对多”（从单一起点或上下文生成序列）或“多对多”结构（生成迭代进行，可能基于先前的输入或输出）。

> 生成序列 ($x_1, x_2, x_3, ...$) 的流程。从一个初始信号开始，模型生成第一个元素 ($x_1$)。这个输出可以作为输入（虚线）反馈，以生成下一个元素 ($x_2$)，依此类推。

## 序列到序列 (Seq2Seq) 任务

一个更复杂的类别涉及将输入序列转换为不同的输出序列。这些任务的关键在于，输入和输出序列通常具有不同的长度和结构。

例子有：

- **机器翻译：** 将一个语言的句子（输入序列）翻译成另一个语言的句子（输出序列）。
- **摘要生成：** 将长文档（输入序列）凝练成更短的摘要（输出序列）。
- **问答：** 根据给定的上下文 (context)文档和问题（输入序列）生成答案（输出序列）。

这些任务通常需要更精密的架构，常包含两个主要组成部分：一个“编码器”，将输入序列处理成固定大小的表示；一个“解码器”，从该表示生成输出序列。我们将在本课程后面提及这些架构。

了解这些不同的序列建模任务是重要的第一步。您旨在解决的具体任务将显著影响您的模型设计、数据准备方式以及性能评估方法。在后续章节中，我们将开始查看循环神经网络 (neural network) (RNN)，这是一类模型，专门用于处理这些任务中数据的序列特性。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供深度学习的全面介绍，包括关于序列建模、循环神经网络及其在预测和生成等各种任务中应用的专门章节。
- [Sequence to Sequence Learning with Neural Networks](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) — Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2014)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 27; Pages: 3104-3112
  一篇开创性的论文，介绍了序列到序列模型，该模型成为机器翻译和摘要等任务的标准，其中输入和输出序列长度不同。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473) — Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014)
  Journal: International Conference on Learning Representations (ICLR 2015); DOI: [10.48550/arXiv.1409.0473](https://doi.org/10.48550/arXiv.1409.0473)
  引入了注意力机制与序列到序列模型，显著推动了神经机器翻译的发展，并影响了其他各种序列建模任务。
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Stanford University (2023)
  Publisher: Stanford University
  一门广受认可的大学课程，提供关于自然语言处理深度学习方法的全面讲座和材料，涵盖各种序列建模任务。
