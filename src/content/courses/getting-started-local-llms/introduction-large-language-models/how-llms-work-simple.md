---
course: "getting-started-local-llms"
chapter: "introduction-large-language-models"
lesson: "how-llms-work-simple"
sourceId: 4196
sourceUrl: "https://apxml.com/zh/courses/getting-started-local-llms/chapter-1-introduction-large-language-models/how-llms-work-simple"
title: "大语言模型如何运作的简单视图"
description: "获得关于 LLM 如何处理输入和生成输出的非技术性简要概览。"
order: 2
plots: []
sourceHash: "8bf1880f4638b543b37d3ee4a31792d7edb883b773fddc6566d8314130703c47"
sourceCorrections: []
---

大语言模型（LLM）是一种旨在理解并生成类似人类文本的人工智能。但它究竟是如何*运行*的呢？我们先从一个简化的视角来看，暂时不涉及复杂的数学。

可以将 LLM 看作是一个高度先进的模式匹配机器，并结合了一个精密的预测引擎。它在海量的文本数据上进行了训练——这些数据包括书籍、文章、网站、代码等等。在这个训练阶段，模型并非像人类那样通过理解意义来“学习”事实。相反，它学习的是单词之间以及单词序列之间的统计关系。它根据无数的例子，找出了在不同语境下哪些单词可能会跟着其他单词出现。

例如，在训练数据中看到“The quick brown fox jumps over the lazy...”这个短语数百万次之后，模型会知道“dog”这个词极有可能紧随其后。它学习语法规则、常用短语、事物间的联系（例如“天空”和“蓝色”），甚至写作风格，所有这些都是从数据中得出的模式。

因此，当你给 LLM 一个提示（输入文本）时，它并非以人类的方式理解你的请求。相反，它执行以下步骤：

1. **分析输入：** 它查看你提供的单词序列。
2. **查询已学模式：** 它使用在训练期间学到的庞大统计模式网络，根据之前处理过的所有内容，确定哪些词最可能接着你的输入序列出现。
3. **预测下一个词：** 它计算潜在下一个词的概率，并通常选择最可能的那个（或其中一个高度可能的词，这允许一定的创造性）。
4. **追加并重复：** 它将这个预测的词添加到序列中，然后重复这个过程：它查看*新的*、更长的序列，并预测*下一个*最可能的词，依此类推。

它一次构建一部分响应，根据提示和它迄今已生成的文本，不断预测接下来应该出现什么。

> 一个简化的流程图，展示了 LLM 如何处理提示以生成文本。

这种预测性的、逐步进行的流程，就是 LLM 常被描述为大规模“下一个词预测器”的原因。它们生成连贯、与语境相关且常常出人意料地富有创造性的文本的能力，源于其训练数据的庞大规模以及它们学到的模式的复杂性，而非源于真正的理解或意识。在下一节中，我们将更仔细地查看这些模型所处理的“部分”，它们被称为 token。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  一本涵盖自然语言处理基础和高级主题的综合性教材，包括语言模型和LLM背后的统计学习原理。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  一项详细综述，提供了对大型语言模型的概览，涵盖了它们的架构、训练方法和主要能力，有助于读者理解该领域。
