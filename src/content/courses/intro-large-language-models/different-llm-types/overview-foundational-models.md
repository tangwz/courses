---
course: "intro-large-language-models"
chapter: "different-llm-types"
lesson: "overview-foundational-models"
sourceId: 3716
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-4-different-llm-types/overview-foundational-models"
title: "基础模型概览"
description: "介绍大型语言模型中的基础模型理念及其核心特性"
order: 1
plots: []
sourceHash: "18fc86054cf6c51f4ac9a2bb519b812eb43409d51beb2e0b6fb9552abb7a0337"
sourceCorrections: []
---

您已知道，大型语言模型是基于庞大的文本数据量进行训练的。这个训练过程计算成本高昂，需要大量资源。想象一下尝试阅读互联网上很大一部分内容，并学习语言的规律——这与这些模型所经历的过程类似。

这引出了基础模型的理念。可以把基础模型看作一个非常大型、多用途的大型语言模型，它经历了对广泛的通用文本数据进行的大规模初步训练。它并没有专门为某*一项*任务训练，而是从训练数据中习得了对语言、语法、事实、推理 (inference)能力等方面的广泛理解。

### 是什么让模型称为“基础”的？

几种特点可以界定这类模型：

- **训练数据广泛：** 它们是基于极其多样和庞大的数据集进行训练的，通常包括网页、书籍、文章和代码。这让它们拥有广泛的知识储备和语言能力。
- **通用能力：** 由于它们的广泛训练，基础模型通常可以“开箱即用”地完成许多不同任务，表现相当不错。这些任务可能包括回答问题、总结文本、翻译语言、编写不同类型的创意内容以及生成代码。
- **规模庞大：** 基础模型通常是最大型的大型语言模型之一，通常包含数十亿甚至数万亿个参数 (parameter)（我们曾在第二章简要提及参数）。这种规模让它们能够掌握语言中的复杂规律。
- **作为基础：** 也许最能界定其特点的是，基础模型作为*基础*或起点。其他更专业化的模型通常是通过使用一个预训练 (pre-training)的基础模型，并将其调整以适应特定任务或使用场景来创建的。这个过程通常被称为微调 (fine-tuning)（尽管我们不会在此处详细讲解）。

### 为什么基础模型如此重要？

从零开始创建一个如此规模的模型是一项巨大的工程。基础模型是这一初步的、资源密集型训练阶段的成果。

> 基础模型是通过对多样化数据进行广泛训练而创建的，并且可以作为创建更专业化模型的基础。

通过使用已有的基础模型，开发人员和研究人员可以：

1. **节省资源：** 避免初步训练所需的庞大计算成本和时间。
2. **利用现有知识：** 利用基础模型已掌握的广泛知识。
3. **开发特定应用：** 将精力放在调整模型以满足特定需求上，而不是从零开始。

可以把它想象成使用预制构件进行建造，而不是自己制作每一块砖。基础模型就是那个功能强大的通用构件。

这些基础模型是您可能接触到的许多大型语言模型的根本。然而，正如我们将在后续章节中看到的，模型在具体用途（通用型与专业型）、可用性（开放与封闭）以及整体规模上有所不同，这通常与其能力有关。了解基础模型的理念有助更好地理解这些差异。

## 参考资料

- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  全面回顾了大型语言模型，包括其发展、架构、训练方法（在海量数据上进行预训练）和适配技术，直接支持对基础模型的理解。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2018)
  Journal: Proceedings of NAACL-HLT 2019; Pages: 4171-4186; DOI: [10.48550/arXiv.1810.04805](https://doi.org/10.48550/arXiv.1810.04805)
  这篇论文介绍了 BERT 模型，建立了在大量数据上预训练大型通用语言模型，然后针对各种下游任务进行微调的范式，这是基础模型的核心概念。
- [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2023)
  Publisher: Stanford University
  这门大学课程提供了关于深度学习自然语言处理原理的深入讲座和材料，涵盖了大型语言模型架构、预训练和微调等主题，这些都是理解基础模型的基石。
