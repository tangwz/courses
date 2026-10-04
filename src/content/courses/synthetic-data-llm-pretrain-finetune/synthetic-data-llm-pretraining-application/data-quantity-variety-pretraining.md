---
course: "synthetic-data-llm-pretrain-finetune"
chapter: "synthetic-data-llm-pretraining-application"
lesson: "data-quantity-variety-pretraining"
sourceId: 6127
sourceUrl: "https://apxml.com/zh/courses/synthetic-data-llm-pretrain-finetune/chapter-3-synthetic-data-llm-pretraining-application/data-quantity-variety-pretraining"
title: "基础模型训练中的数据量与多样性"
description: "了解数据量和多样性对基础大型语言模型预训练的影响。"
order: 1
plots: ["plots/6127-0.json"]
sourceHash: "f7fea2b8c386f6a6ead68495693343e5695e546b99017f9da1d695be2bb1f3a9"
sourceCorrections: []
---

大型语言模型预训练 (pre-training)阶段的有效性主要取决于所用数据的特点。这不只是普通数据，而是具有足够规模和多样性的数据，使得这些模型能够形成其突出的能力。纯粹的数据量（通常表示为 $V_{data}$）及其多样性对成功的基础模型训练是必不可少的。

### 规模要求：越多通常越好

在大型语言模型的发展中，一个持续观察到的现象是“缩放法则”所描述的。简单来说，这些经验性发现表明，模型的性能（通常通过其准确预测文本的能力衡量，例如测试集上更低的困惑度或损失）会随着三个主要因素的增加而可预测地提升：模型大小（参数 (parameter)数量）、训练所用的计算资源量，以及对我们讨论而言很重要的——训练数据集的大小。

对于预训练 (pre-training)而言，这意味着让模型接触更大容量的文本 $V_{data}$ 通常会带来更好的结果。为何会如此？

- **获取更广泛的知识**：更大的数据集自然包含关于众多主题的更多事实、概念和信息。模型有更多学习这些内容的机会。
- **模式识别**：大型语言模型通过识别语言中的统计模式来学习。更多数据提供了更丰富的示例集合，使模型能够学习更精细复杂的模式，从简单语法到复杂的叙事结构或逻辑联系。
  "\* **更好的泛化能力**：拥有更多数据，模型不太可能仅仅记忆特定示例，能够更好地将所学模式泛化到新的、未见过的文本。虽然在预训练中使用的大规模数据集不太需要担心传统过拟合 (overfitting)，但数据不足可能导致模型在多样化输入上表现不佳。"

这意味着很清楚：要构建有能力的基础模型，我们通常需要真正大规模的数据集，有时以数千亿甚至数万亿个token来衡量。



![大型语言模型性能与训练数据量对比](plots/6127-0.json)



> 常见关系：随着训练数据量的增加，模型的测试损失通常会降低，表明性能有所提升。

### 数据多样性的作用

尽管数据量是性能的主要驱动因素，但数据内部的多样性或差异性也同等重要，甚至更重要。由重复或非常狭窄内容组成的大规模数据集，将无法产出有能力且多功能的LLM。预训练 (pre-training)数据的多样性体现在几个维度：

- **主题覆盖**：数据应涵盖广泛的主题，包括科学、文学、新闻、历史、艺术等。这使模型能够对不同领域有广泛的认识。
- **风格变化**：语言在不同语境中用法不同。一个好的预训练语料库应包含多样化的风格：正式学术论文、非正式博客文章、结构化技术文档、对话、创意小说，甚至源代码。
- **结构多样性**：文本以各种格式存在。接触文章、书籍、问答对、列表、表格和脚本有助于模型理解并生成这些不同结构下的文本。
- **语言多样性**：尽管许多基础模型主要以英语训练，但如果需要多语言能力，整合多种语言的数据，或同一语言内的不同方言和社群方言，可以增加适用性和对不同输入的适应性。

高数据多样性的好处是多方面的：

- **更好的泛化能力**：在多样化数据上训练的模型，更能应对以前从未遇到的更广泛的提示和任务。
- **更复杂的理解**：接触不同的视角、论点和表达思想的方式，有助于模型形成对语言和各种主题更复杂的理解。
- **减少偏见**：虽然不是一个完整的解决方案，但更具多样性的数据集可以帮助降低模型学习和放大狭窄数据源中存在的偏见的风险。然而，仔细整理仍然必不可少。
- **涌现 (emergence)能力**：大型语言模型的一些高级能力似乎在模型于非常大且多样化的数据集上训练时出现，这表明不同类型信息的互动促成了新形式的“理解”或推理 (inference)。

> 一个多样化的预训练语料库由多种来源构建，每种来源贡献不同类型的信息和风格。合成数据可以补充这些来源以增加多样性。

### 互动：数量放大多样性作用

数据量和多样性并非独立；它们相互影响。一个真正大的数据集 $V_{data}$，使得纳入大量多样化内容成为可能。许多有价值的数据类型，例如专业科学文本、特定编程语言或复杂的哲学讨论，相比于一般网页文本可能相对稀有。在较小的数据集中，这些“长尾”来源可能表示不足，难以对模型的学习产生有意义的影响。然而，在一个数万亿token的数据集中，即使是这些较稀有的数据类型也可以以足够大的量包含在内，从而有助于模型的知识和能力。

本质上，大数据量为容纳广泛的信息范围提供了“空间”。预训练 (pre-training)目标，通常是下一token预测或掩码语言建模，在此基础上表现良好。模型通过识别这组文本中的模式，学习预测接下来是什么（或缺失什么）。它遇到的模式越多样，看到的每个模式的示例越多，其语言理解就越具适应性和普遍性。

### 合成数据在满足这些需求中的作用

这正是合成数据（本课程的重点）对预训练 (pre-training)特别重要的原因。虽然理想情况是拥有丰富、高质量、多样化的数据，但这并非总能实现，原因在于：

- **稀缺性**：某些类型的知识或文本风格可能确实稀有或难以获取。
- **许可和版权**：许多现有文本并非免费可用于模型训练。
- **隐私问题**：包含个人信息的数据不能使用。
- **质量问题**：数据可能嘈杂、不一致或包含不良内容。

合成数据生成技术通过以下方式提供了一种增强预训练语料库的途径：

1. **增加数量 ($V_{data}$)**：生成大量文本，特别是在有特定需求引导的情况下。
2. **提升多样性**：创建现有数据集中代表性不足的特定特征数据。例如，可以合成生成更多复杂的指导文本、特定领域知识（如法律或医学文本，并采取适当保护措施），或低资源语言中的数据。

理解基础模型训练中数据量和多样性的基本需求是第一步。它为理解经过深思熟虑生成的合成数据如何成为构建更强大、更全面的大型语言模型的有效工具奠定了基础，正如我们将在后续章节中涵盖的那样。

## 参考资料

- [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) — Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy (2021)
  Journal: arXiv preprint arXiv:2101.00027; DOI: [10.48550/arXiv.2101.00027](https://doi.org/10.48550/arXiv.2101.00027)
  介绍了专门为提高通用语言模型能力而设计的大型多样化数据集，强调了来自多个来源的数据多样性的重要性。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  介绍了GPT-3模型，强调了大规模预训练数据和模型尺寸在实现强大的少样本学习和涌现能力方面的重要贡献。
