---
course: "llm-model-sizes-hardware"
chapter: "intro-llms-model-size"
lesson: "model-parameters"
sourceId: 4195
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-1-intro-llms-model-size/model-parameters"
title: "了解模型参数"
description: "学习参数在神经网络和大型语言模型中的含义。"
order: 2
plots: []
sourceHash: "197a790bb3d1789f13b54623ffdc73b0325a0631ad87e383f8a96396be123a3d"
sourceCorrections: []
---

我们来思考一下大型语言模型处理和生成语言的原理。它们究竟是如何真正地*学会*这些的呢？答案在于一个被称为**参数 (parameter)**的东西。

可以把大型语言模型想象成一个非常复杂的网络，它的连接方式大致受到了大脑的启发。这个网络由相互连接的“节点”或“神经元”层组成。当你训练一个大型语言模型时，你会向它输入海量的文本数据。训练过程会调整这些神经元之间连接的“强度”或“权重 (weight)”。这些可调整的连接强度就是**参数**。

### 什么是参数 (parameter)？

在机器学习 (machine learning)，特别是大型语言模型的背景下，参数是模型内部的变量，其数值在训练过程中从数据中习得。它们主要决定了模型在特定任务上的能力，例如理解语法、事实、推理 (inference)能力以及不同文本风格等。

你可以将参数想象成模型内部数百万（甚至数十亿！）个微小的刻度盘。每个刻度盘的设置都会在训练过程中根据模型看到的例子进行调整。所有这些刻度盘的最终设置代表了模型所获得的“知识”。

神经网络 (neural network)中常见的参数类型包括：

- **权重 (weight)：** 它们代表两个神经元之间连接的强度。权重越大，一个神经元发出的信号对下一个神经元的影响就越强。
- **偏差：** 这些是应用于神经元的额外数值，可以改变它们的输出，帮助模型更好地适应数据。可以将其视为对神经元活动的一种基线调整。

对于大型语言模型而言，这些参数共同捕捉了其训练所用语言的模式、结构和细微差别。

### 为什么参数 (parameter)决定模型大小

这些可学习参数（权重 (weight)和偏差的总和）的总数量，是我们衡量大型语言模型“大小”的标准方式。为什么呢？

1. **与复杂性直接关联：** 通常来说，参数越多的模型，学习复杂模式和从训练数据中存储更多信息的能力就越强。它有更多的“刻度盘”可以调整，从而能更细致地表示语言。
2. **对资源的影响：** 参数数量直接影响所需的计算资源。每个参数都需要存储在内存中（我们稍后会讨论显存 (VRAM)），并在训练和使用（推理 (inference)）过程中进行计算。参数越多意味着需要更多的内存和更多的计算量。

当你听说一个模型拥有“70亿参数”（通常写作7B）时，这意味着该模型拥有7,000,000,000个在训练期间习得的可调整权重和偏差。这个数字为我们提供了模型潜在能力的大致但有用的估计，并且也表明了其硬件需求。

> 这是不同层中神经元（节点）之间连接（边）的一个高度简化视图。标签“w”表示权重（参数），表明每个连接的强度。偏差（也是参数）与隐藏层和输出层中的神经元相关联。所有“w”值和偏差的总和即为参数数量。

理解参数很重要，因为这个数量是决定大型语言模型需要多少内存和计算能力的主要因素，我们将在接下来的章节中进一步查看。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本介绍深度学习概念的综合教材，涵盖神经网络的数学基础、参数、权重和偏置。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  这篇开创性论文介绍了Transformer架构，它是大多数现代大型语言模型的基础，并定义了它们的参数结构。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; Pages: 1-19; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  一篇研究模型性能如何随参数数量、数据集大小和计算资源扩展的论文，提供了LLM规模影响的见解。
