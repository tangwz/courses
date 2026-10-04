---
course: "introduction-to-llm-fine-tuning"
chapter: "foundations-of-model-customization"
lesson: "pre-training-vs-fine-tuning"
sourceId: 7263
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-1-foundations-of-model-customization/pre-training-vs-fine-tuning"
title: "预训练与微调的比较"
description: "比较从零开始预训练和微调现有模型的目标、数据需求和计算成本。"
order: 2
plots: []
sourceHash: "1786dd138ddcf925d247eda2d6feeae72f6e3706feb0680d65839357a2b8d843"
sourceCorrections: []
---

为有效定制大型语言模型，区分其两个主要学习阶段至关重要：预训练 (pre-training)和微调 (fine-tuning)。尽管两者都涉及训练神经网络 (neural network)，但它们的目标、数据需求和计算规模本质上不同。可将其视为模型生命周期中的两个不同阶段：第一个阶段建立广泛的知识基础，第二个阶段则为特定目的优化这些知识。

### 预训练 (pre-training)阶段：建立通用知识

预训练是生成GPT-3、Llama或Mistral等基础模型的工业级规模流程。其目标是让模型充分理解语言，包括语法、句法、推理 (inference)能力以及事实信息库。

- **目标：** 模型学习预测序列中的下一个词元 (token)（一个词或词的一部分）。通过在海量数据集上反复执行此操作，它隐式地学习了人类语言的统计规律。学习目标是通用且不局限于任何单一应用。
- **数据集：** 预训练使用海量、大部分非结构化的文本语料库，这些语料库抓取自公共互联网、书籍及其他来源。我们谈论的是数TB甚至数PB的数据，包含数十亿个句子。这些数据不为特定任务手动标注；任务是自监督的，因为句子中的下一个词作为它自己的标签。
- **计算成本：** 此阶段资源消耗极大。它需要数千块高端GPU并行运行数周或数月，计算时间成本达数百万美元。因此，预训练通常仅由大型、资金充足的组织和研究实验室执行。
- **结果：** 结果是**基础模型**。该模型是一个强大的通用型模型。它能回答问题、撰写文本，并有一定熟练度地完成多类任务，但它缺乏专业性。

### 微调 (fine-tuning)阶段：适应特定任务

微调始于预训练 (pre-training)之后。它接收一个强大的通用基础模型，并使其擅长特定任务或特定专业方面。它不是从零开始学习，而是优化模型参数 (parameter)中已有的知识。

- **目标：** 微调的目标是专业化。这可能意味着教模型采用特定对话风格（例如，专业客服代理），按特定格式执行指令（例如，从自然语言生成代码），或获得特定小众学科的专业知识（例如，法律或医学术语）。
- **数据集：** 微调依赖于小得多、高质量、经过整理的数据集。这些数据集通常仅几MB或几GB大小，并专为目标任务构建。例如，一个指令遵循数据集将包含数百或数千个`提示`/`完成`对。
- **计算成本：** 微调所需的资源比预训练小几个数量级。它通常只需一块或几块GPU，在数小时或数天内即可完成，这使个人开发者和小型组织也能使用。
- **结果：** 结果是**专业化模型**。该模型在目标任务上的表现相较于原始基础模型更优。

### 并排比较

下表总结了这两个过程的主要区别。

| 特性 | 预训练 (pre-training) | 微调 (fine-tuning) |
| --- | --- | --- |
| **目标** | 通用语言理解 | 任务专属性能或专业领域适应 |
| **数据规模** | 数TB至数PB（例如，整个互联网） | 数MB至数GB（精选示例） |
| **数据类型** | 非结构化、未标注文本 | 结构化、已标注示例（例如，提示/响应） |
| **计算规模** | 数千块GPU，数周至数月 | 1至8块以上GPU，数小时至数天 |
| **模型产物** | 基础模型（通用型） | 专业化模型（专长型） |
| **起点** | 随机初始化的权重 (weight) | 来自预训练模型的权重 |

下图说明了这一两阶段过程。预训练是一次性、大规模投入，生成一个多功能的基础模型。这个单一产物可以成为众多小型微调工作的起点，每个都为不同应用创建独立、专业化的模型。

> 模型从通用预训练到多项专业化微调应用的生命周期。

### 迁移学习 (transfer learning)的作用

整个过程是**迁移学习**的一项高效应用。高成本预训练 (pre-training)阶段获得的知识被“迁移”到微调 (fine-tuning)任务中。通过从预训练模型的权重 (weight)开始，您并非从零开始。相反，您是从一个已理解语法、语境并掌握大量相关信息的模型开始。

微调只是调整这些权重，使其更好地适应您的小型、特定任务数据集中的模式。这就是微调如此有效且高效的原因。它建立在预训练期间完成的大规模计算工作的基础之上，使您能在特定问题上获得高性能，仅需一小部分数据和计算资源。了解这种关联对于做出明智的定制LLM决策至关重要。

## 参考资料

- [Universal Language Model Fine-tuning for Text Classification](https://aclanthology.org/P18-1031/) — Jeremy Howard and Sebastian Ruder (2018)
  Journal: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Pages: 328-339; DOI: [10.18653/v1/P18-1031](https://doi.org/10.18653/v1/P18-1031)
  本文介绍了一种有效微调预训练语言模型以完成下游自然语言处理任务的方法，展示了迁移学习的效率和性能提升。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171-4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  这篇里程碑式的论文提出了一种强大的语言模型预训练技术和成功的微调策略，成为许多后续大型语言模型的标准。
- [Transfer Learning in Natural Language Processing](https://aclanthology.org/N19-5004/) — Sebastian Ruder, Matthew E. Peters, Swabha Swayamdipta, Thomas Wolf (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Tutorials; Publisher: Association for Computational Linguistics; Pages: 15-18; DOI: [10.18653/v1/N19-5004](https://doi.org/10.18653/v1/N19-5004)
  本教程全面概述了应用于自然语言处理的迁移学习技术，是理解其在大型语言模型中原理和应用的优质资源。
- [Exploring the Limits of Language Modeling](https://doi.org/10.48550/arXiv.1602.02410) — Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, Yonghui Wu (2016)
  Journal: arXiv preprint arXiv:1602.02410; DOI: [10.48550/arXiv.1602.02410](https://doi.org/10.48550/arXiv.1602.02410)
  本文展示了在海量数据集上进行大规模语言模型预训练的有效性，表明了模型在专业任务之前学习通用语言理解的潜力。
