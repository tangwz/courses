---
course: "prompt-engineering-llm-application-development"
chapter: "integrating-llms-external-data-rag"
lesson: "limitations-standard-llm-knowledge"
sourceId: 3056
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-6-integrating-llms-external-data-rag/limitations-standard-llm-knowledge"
title: "标准大语言模型知识的局限"
description: "了解大语言模型为何需要访问外部、最新或私有信息。"
order: 1
plots: []
sourceHash: "10398c71afa11609a3737c389d8b22acb766ca372fdbb951864c16ab74860f5e"
sourceCorrections: []
---

大语言模型 (LLM)，例如您通过API接触到的那些，是在包含互联网文本、代码和数字化书籍的海量数据集上训练的。此训练过程使它们获得了对语言、事实、推理 (inference)模式以及不同方面的全面认识。然而，这些知识基本是静态的，反映的是*训练数据收集时的*信息状态。这种特性在构建实际应用程序时，带来多项主要局限。

### 知识截止日期问题

每个预训练 (pre-training)的大语言模型 (LLM)都有一个“知识截止日期”。这是指模型在训练阶段之后不再接触新信息的时间点。您可以将其想象成信息的快照，固定在某个特定时刻。因此，模型不了解以下内容：

- **近期事件：** 在模型训练完成后发生的主要政治事件、科学发现或文化动态。
- **新数据：** 截止日期之后出现的网上发布信息、新的研究论文或更新的统计数据。
- **变化趋势：** 比其知识库更新的公众意见变化、市场走向或技术进步。

如果您询问模型上周发生的事情，或者查询昨天发布产品的最新规格，它很可能没有答案。好一点的情况，它会说明其知识局限；差一点的情况，它可能会试图根据旧数据推断或猜测，这可能导致不准确或过时的回复。

请设想，询问一个知识截止日期为2023年初的大语言模型，关于2023年末举行的一场重要选举的获胜者。它根本不会知道，因为该信息不存在于其训练语料库中。

### 无法获取实时信息

与知识截止日期相关的是，标准大语言模型 (LLM)无法访问实时、动态的数据流。它们不能：

- 查看当前股价。
- 提供即时天气预报。
- 报告实时体育比赛分数。
- 获取突发新闻。

需要真正最新信息的应用程序不能仅依赖于大语言模型的内部知识。

### 无法访问私有或专有数据

在企业或个性化应用程序开发中，最常遇到的局限可能是大语言模型 (LLM)本身无法访问私有数据源。标准模型没有内置的查询能力，无法查询：

- 公司内部数据库或知识库。
- 私有代码库。
- 用户专属文档或电子邮件。
- 专有研究数据。
- 您组织专属的客户支持日志。

设想您正在构建一个客户支持机器人。虽然通用大语言模型知道如何礼貌地对话和回答常见问题，但它无法访问*您*公司的具体产品手册、故障排除指南或客户历史数据库，从而无法提供定制化、准确的支持。由于上下文 (context)窗口大小限制和安全问题，将所有可能相关的私有数据作为提示每次都输入，通常不现实。

### 虚构（幻觉 (hallucination)）的可能性

当面对超出其知识范围或需要它们不具备信息（如近期或私有数据）的查询时，大语言模型 (LLM)有时会“虚构”或“产生幻觉”。这意味着它们生成的回复听起来合理且语法正确，但实际不准确或毫无意义。它们可能会编造细节，错误回忆训练数据中的事实，或以误导性方式组合不相关的信息。出现这种情况，是因为模型的目的通常是根据训练期间学到的模式生成连贯的文本序列，不一定保证事实准确性，特别是对于从未训练过的信息。

> 大语言模型的内部知识限于其训练数据，这在其所知内容与回答某些查询所需的外部环境实时、私有或近期信息之间形成一个差异。红色虚线表示标准大语言模型通常无法获取的信息。

这些局限表明，需要有机制让大语言模型在生成过程中查阅外部信息源。仅依赖于模型预训练 (pre-training)的参数 (parameter)，不足以完成要求当前、特定或专有知识的任务。这正是检索增强生成（RAG）要解决的问题，也是本章的侧重内容。通过首先获取相关外部信息，然后将其作为上下文 (context)提供给大语言模型，RAG系统使模型能够生成更准确、及时和相关的回复。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11472) — Zheng Ge, Zequn Jie, Xin Huang, Chengzheng Li, Osamu Yoshie (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: arXiv; DOI: [10.48550/arXiv.2005.11472](https://doi.org/10.48550/arXiv.2005.11472)
  介绍了检索增强生成（RAG）方法，该方法通过利用外部信息来解决大型语言模型静态知识的局限性。
- [A Survey of Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions](https://dl.acm.org/doi/10.1145/3631481) — Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Li, Yong Ivanov, Samuel Albanie, Bo Li, Luc Van Gool (2023)
  Journal: ACM Computing Surveys; Publisher: Association for Computing Machinery; Volume: 56; Pages: 1-38; DOI: [10.1145/3631481](https://doi.org/10.1145/3631481)
  全面概述了大型语言模型中的幻觉现象，涵盖其成因、类型和缓解策略。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; Pages: 144; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  大型语言模型的广泛综述，其中包含关于事实不准确性、知识截止日期以及对外部信息依据的需求等章节。
