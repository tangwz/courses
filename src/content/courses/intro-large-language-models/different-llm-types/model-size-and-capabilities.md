---
course: "intro-large-language-models"
chapter: "different-llm-types"
lesson: "model-size-and-capabilities"
sourceId: 3723
sourceUrl: "https://apxml.com/zh/courses/intro-large-language-models/chapter-4-different-llm-types/model-size-and-capabilities"
title: "理解模型大小与能力"
description: "了解语言大模型的规模（参数数量）如何与其能力关联。"
order: 4
plots: ["plots/3723-0.json"]
sourceHash: "8ee682398521dd6477ade451be76a3e806d867c7e4d89d1b7b98985010d06c88"
sourceCorrections: []
---

比较不同的语言大模型时，它们的大小是常被谈及的一个特点。但在此处，“大小”究竟指什么？它又与模型实际能做的事情有什么关联呢？

### 模型大小的决定因素：参数 (parameter)

在之前的章节中，我们提及了语言大模型如何在训练过程中，根据处理的大量文本数据来调整其内部设置从而进行学习。这些可调整的设置称为**参数**。可以将参数想象成模型内部的小旋钮或刻度盘。在训练期间，这些旋钮的值会被调整，使模型更善于预测下一个词或理解语言模式。

语言大模型的“大小”通常通过**这些参数的总数量**来衡量。参数更多的模型有更多的“旋钮”可以调整，这通常使其能从训练数据中学到更复杂的模式和细节。模型大小差异很大：

- **小型模型**可能有数百万或数十亿参数。
- **中型模型**通常在数百亿参数的量级。
- **大型模型**，也就是常上新闻的那些，可以拥有数千亿甚至数万亿参数。

正是这种庞大的参数数量，构成了语言大模型中“大”的含义。

### 大小与能力之间的关联

通常来说，模型的参数 (parameter)数量与其能力之间存在关联。参数更多的模型通常表现出：

- **知识范围更广：** 它们经过更多样的数据训练，有更大的容量存储和回溯信息。
- **复杂任务表现更佳：** 需要较深层次推理 (inference)、创造性、多步骤指令或理解细微语境的任务，通常能从大型模型中获益。比如，创作诗歌、生成复杂代码或解释科学观念，可能由大型模型处理得更好。
- **语言理解能力提升：** 大型模型通常对语法、风格、语气和语境有更好的掌握。它们往往能更准确地遵循指令，并生成更连贯、更像人类的文本。

想象一下要建造一个复杂的东西。一个更大的工具箱（更多参数）会给你更多专业工具（学到的模式），以更高效地处理各种复杂的任务。一个较小的工具箱可能足以应付简单的任务，但应对高度复杂的项目时可能会遇到困难。



![通用关系：模型大小与能力](plots/3723-0.json)



> 此图表显示了一个普遍趋势，即参数越多的模型倾向于处理更复杂的任务并展现出更广泛的能力。请注意，这是一种简化表示；训练数据质量和模型架构等因素也扮演着重要角色。

### 重要的权衡

尽管大型模型通常拥有更强大的能力，但大小并非决定一切，它也伴随着重要的权衡：

1. **计算成本：** 训练大型模型需要巨大的计算能力（如GPU或TPU等专用硬件）、能源和时间，使其非常昂贵。运行（或*推理 (inference)*）这些大型模型也需要大量的计算资源。
2. **速度：** 相比更小、更灵活的模型，大型模型处理输入和生成输出通常需要更长时间。
3. **可访问性：** 训练甚至运行最大型模型所需的资源，可能会限制拥有大量资金和基础设施的组织才能使用它们。小型模型通常更容易部署在标准硬件甚至移动设备上。
4. **专用性：** 有时，专门为特定任务（如情感分析或两种特定语言之间的翻译）训练或微调 (fine-tuning)的小型模型，在该特定任务上的表现可能优于大型通用模型，同时效率也更高。

选择合适的模型需要平衡所需能力与这些实际考量。对于许多常见任务，小型或中型模型可能完全足够，并且比可用的最大选项更实用。当您与不同的语言大模型交互时，请思考它们的大小如何影响其性能以及有效使用它们所需的资源。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; Pages: 19; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  一篇开创性论文，量化了语言模型性能与模型大小、计算预算和数据集规模之间的关系。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems; Publisher: NeurIPS; Volume: 33; Pages: 1877-1901; DOI: [10.5591/978-0-9799981-0-1.vol33.abs175](https://doi.org/10.5591/978-0-9799981-0-1.vol33.abs175)
  介绍了GPT-3，一个拥有1750亿参数的模型，展示了大型模型如何通过少量样本（少样本学习）获得强大性能。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，全面解释了神经网络，包括参数在模型学习和表示中的作用。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  一份近期调查，概述了大型语言模型、其架构、训练方法、能力及相关挑战。
