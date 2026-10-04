---
course: "intro-large-language-models"
chapter: "using-pre-trained-llms"
lesson: "what-are-pre-trained-models"
sourceId: 3728
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-5-using-pre-trained-llms/what-are-pre-trained-models"
title: "什么是预训练模型？"
description: "了解预训练模型的含义，以及它们为何能帮助你快速上手。"
order: 1
plots: []
sourceHash: "82dbda43adf4c8fcf9c15869a3360bb60cd8831ed32ac581674db13ae3907e6a"
sourceCorrections: []
---

在之前的章节中，我们讲到大型语言模型通过处理海量文本数据来学习。这个训练过程计算量大，需要大量算力 (compute)、庞大的数据集和可观的时间。这是一项复杂的任务，通常由拥有雄厚资源的大型研究机构或公司来完成。

预训练 (pre-training)模型是大型语言模型（LLM）完成其训练阶段后的结果。可以把预训练模型看作一个已经完成了基础教育的LLM。它通过分析庞大的训练数据集，学习了语法、事实（如其训练数据中所示）、推理 (inference)能力和语言模式。

“预训练”这个标签表明，模型创建中最耗费资源的部分已经完成。开发者已经投入时间和计算资源，将这些基础知识构建到模型的参数 (parameter)中，即模型用于做出预测的内部变量。

### 为什么要用预训练 (pre-training)模型？

使用预训练模型有几个重要的好处，特别是对于初学者来说：

1. **易得性：** 它使得高级AI功能变得可用，而无需超级计算机或海量数据集。你可以直接使用大规模训练的成果，而无需自己进行训练。
2. **即时可用：** 你可以立即开始与模型交互并将其用于各种任务。你不需要自己制造引擎，可以直接开车。
3. **内置功能：** 这些模型自带在大量训练中学到的广泛语言理解和生成功能。它们通常可以开箱即用地出色完成翻译、摘要、问答和文本生成等任务。

设想一下，从零开始构建一个复杂的软件库（比如图形引擎），与使用一个已有的、完善的库，两者有何不同。预训练模型就像那些已建成的库；它们提供了一个强大的支撑，你可以在此基础上进行开发或直接使用。

### 理解“预训练 (pre-training)”的特性

理解“预训练”意味着模型具有坚实的通用基础，但它不一定立即专门针对你可能想到的每个特定任务。它的知识通常也停留在其训练数据收集的时间点；它通常无法获取实时信息，除非是为此特别设计的（例如与搜索引擎集成）。

主要观点是，学习语言基本原理的艰巨工作已经完成。你作为与预训练LLM交互的用户，你的任务通常是使用提示词 (prompt)将现有知识引向你的具体目标，这我们在第三章中讲过。

在接下来的章节中，我们将了解如何通过各种服务和平台找到这些预训练模型，以及如何使用简单的工具（如网页界面或基本编程接口（API））开始与它们交互。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) — Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova (2019)
  Journal: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers); Publisher: Association for Computational Linguistics; Pages: 4171–4186; DOI: [10.18653/v1/N19-1423](https://doi.org/10.18653/v1/N19-1423)
  本文介绍了BERT，一个基于Transformer的模型，显著推动了自然语言理解任务的预训练范式。它描述了模型如何通过广泛的预训练获得通用语言知识。
- [Natural Language Processing with Transformers: Building Innovative Applications with Deep Learning Models](https://www.oreilly.com/library/view/natural-language-processing-with/9781098136796/) — Lewis Tunstall, Leandro von Werra, and Thomas Wolf (2022)
  Publisher: O'Reilly Media; Pages: 406
  本书提供了使用Hugging Face Transformers库的实用指导，该库提供了广泛预训练模型的访问。它介绍了如何将这些模型用于各种自然语言处理任务，展示了它们的易用性和实用性。
