---
course: "introduction-to-llm-fine-tuning"
chapter: "foundations-of-model-customization"
lesson: "role-of-transfer-learning"
sourceId: 7266
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-1-foundations-of-model-customization/role-of-transfer-learning"
title: "迁移学习在大型语言模型中的作用"
description: "了解迁移学习的原理如何被应用于调整通用大型语言模型以适应特定的下游任务。"
order: 5
plots: []
sourceHash: "517116ddb48154d71fff1b8f550b84c5e1bb11bc820ae6dc8ecb23f57315d2a3"
sourceCorrections: []
---

微调 (fine-tuning)大型语言模型是迁移学习 (transfer learning)的一个直接且有效的应用。在机器学习 (machine learning)中，迁移学习利用从解决一个问题（*源任务*）中获得的知识，并将其应用于另一个不同但相关的问题（*目标任务*）。对于大型语言模型而言，源任务是初始的预训练 (pre-training)阶段，而目标任务则是您的具体应用场景。

当基础模型进行预训练时，它处理的是一个海量且多样化的文本语料库。这个过程不仅仅是教会模型预测下一个词；它还促使模型构建一个精细的内部语言表示。这包括语法、句法、语义关系以及大量的知识。所有这些信息都编码在模型的参数 (parameter)或权重 (weight)中。这些权重并非随机数，它们代表了模型所见数据中提炼出的知识。

微调利用这份丰富的、预先存在的知识作为高效的起点。我们并非为您的任务从随机权重开始初始化新模型并从头训练，而是以基础模型已完成训练的权重为起始。然后，训练过程在一个与您的目标任务相关的、更小的专用数据集上继续进行。学习算法对这些权重进行增量更新，温和地调整模型的行为，使其与新数据中存在的模式、风格和信息对齐 (alignment)。

这个过程本质上是关于效率的。模型不是从零开始学习语言；它是在调整它已掌握的知识。

> 大型语言模型中的迁移学习过程。预训练获得的通用知识在微调过程中被迁移并调整，以适应专门任务。

### “热启动”的优势

从预训练 (pre-training)权重 (weight)开始为训练过程提供了常说的“热启动”。与从“冷启动”（随机初始化）训练模型相比，这种方法有多项显著益处。

- **计算和数据效率：** 从零开始训练大型模型需要庞大的计算资源和海量数据集，通常耗资数百万美元。迁移学习 (transfer learning)使您能够利用在基础模型预训练中已投入的巨大资源。微调 (fine-tuning)所需的数据和计算时间要少得多，因为主要目标是适应而非基础学习。您的任务可能只需要数千个高质量示例，而非数万亿个token。
- **更好的泛化能力：** 仅在小型专用数据集上从零开始训练的模型存在很高的过拟合 (overfitting)风险。它可能会完美地记住训练示例，但在新的、未见过的数据上表现不佳，因为它尚未学习语言的普遍原理。通过从预训练模型开始，您可以将微调过程锚定在对语言的普遍理解上。这有助于模型从您的少量数据集更好地泛化到更广泛的任务上。
- **更高的性能上限：** 预训练知识提供了更好的基础，通常能让模型在目标任务上获得更高的最终性能。模型可以将其有限的训练预算集中在学习新领域的具体内容上，而不是花费在学习“the”是限定词或问题通常以问号结尾这样的基础知识。

知识迁移的这一原则支撑着本课程中我们将涉及的所有微调技术。无论您在完全微调期间调整模型的每个参数 (parameter)，还是使用参数高效方法仅修改一小部分，您都始终在进行迁移和提炼现有知识的过程。理解这一联系有助于您做出明智决策，选择最适合您的目标和限制的微调策略。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Volume: 1; Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  这篇基础论文介绍了 BERT 模型，它推广了预训练和微调范式，展示了迁移学习在自然语言处理中的效率和性能提升。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  一份近期且全面的调查，提供了大型语言模型的广泛概述，包括它们的预训练、微调以及迁移学习的基本原理。
