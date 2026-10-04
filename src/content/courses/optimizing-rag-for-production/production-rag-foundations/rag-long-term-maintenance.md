---
course: "optimizing-rag-for-production"
chapter: "production-rag-foundations"
lesson: "rag-long-term-maintenance"
sourceId: 6162
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-1-production-rag-foundations/rag-long-term-maintenance"
title: "RAG系统长期维护的难题"
description: "了解生产环境中RAG系统长期维护、更新和演变相关的难题。"
order: 4
plots: []
sourceHash: "02c9c296df4483869090507394c109dace4c35acc6c22166b63adf04d814d3f5"
sourceCorrections: []
---

检索增强生成（RAG）系统的初始设置和部署，通常只是漫长运行周期的一个起点。与静态软件不同，RAG系统是动态的，它会与不断演变的数据、模型和用户期望进行交互。忽视长期维护可能导致性能下降、成本上升，并最终造成用户不满意。在长期保持生产环境RAG系统良好运行和有效性方面，存在一些持续存在的难题。

### 不断变化的知识库

任何RAG系统的核心都是其知识库。这个语料库，无论是文档集合、数据库还是网页内容，都很少是静态的。新信息不断产生，现有数据变得过时，相关性也会转移。

- **数据过时与准确性：** 最明显的难题是确保RAG系统检索并用于生成的信息保持最新和准确。过时的产品规格、旧的政策文件或已作废的新闻文章可能导致不正确或误导性的回复。这需要强大的数据管道，用于检测源数据变化、摄取更新和重新索引嵌入 (embedding)。这些更新的频率和方法（例如，完整重新索引与增量更新）将取决于数据的波动性和对过时信息的容忍度。
- **模式演变：** 数据源不仅可能内容发生变化，结构也可能改变。如果您的RAG系统依赖特定的元数据字段或文档结构以实现最佳的分块和检索，那么这些模式的变化可能破坏数据摄取管道或降低检索质量。对数据模式进行版本控制并具备可适应的解析逻辑变得很重要。
- **重新索引的成本和停机时间：** 重新处理和重新索引大型知识库可能计算成本高且耗时。对于需要高可用性的系统，在没有明显停机时间或性能下降的情况下执行这些更新需要精心规划，可能涉及向量 (vector)索引的蓝绿部署或实时索引交换技术。

考虑财务影响：如果每月重新嵌入整个10TB知识库需要24小时的GPU时间，这将成为一项可预测的运营开销，需要计入您的总拥有成本 (TCO)。

### 模型退化与漂移

RAG系统依赖至少两种机器学习 (machine learning)模型：用于检索的嵌入 (embedding)模型和用于生成的语言模型（LLM）。两者都随着时间推移易发生某种形式的漂移。

- **嵌入模型漂移：** 术语和概念的语义含义可能演变，或者查询分布的性质可能变化。基于昨日数据微调 (fine-tuning)的嵌入模型可能无法最佳地表示今日的细节。这可能表现为检索相关性的逐渐下降。监控检索指标并定期评估是否需要重新微调甚至替换嵌入模型，这一点很重要。更换嵌入模型也意味着重新索引整个知识库，这是一项重要的工作。
- **LLM性能漂移：** 如果您依赖第三方LLM API，底层模型可能由提供商更新。虽然通常有益，但这些更新有时可能导致特定提示和使用场景下的输出风格、冗长程度甚至事实准确性上发生细微变化。对于自托管LLMs，更新甚至服务基础设施的微小变化都可能产生类似的影响。在推出任何LLM变更之前，针对由提示和预期输出组成的“黄金数据集”进行严格的回归测试非常重要。
- **用户查询语义漂移：** 用户提问的方式或他们感兴趣的话题可能发生变化。如果您的RAG系统最初针对特定查询风格或主题集进行优化，用户行为的转变可能导致性能下降。这通常需要重新评估提示工程 (prompt engineering)策略，甚至重新训练检索组件。

以下图表说明了知识库或模型中未被察觉的漂移如何随时间推移拉大用户意图与RAG系统输出之间的差距。

> 随着时间推移，由于知识库、模型或用户查询模式中未解决的漂移，可能导致用户意图与RAG输出质量之间出现潜在差异。

### 管理复杂的依赖关系网

生产RAG系统很少是单一的。它们是相互关联的组件生态系统：

- 向量 (vector)数据库（例如，Pinecone, Weaviate, FAISS）
- 嵌入 (embedding)模型提供商或库（例如，Hugging Face Transformers, OpenAI Embeddings）
- LLM提供商或库（例如，Anthropic, Cohere, 本地TGI实例）
- 编排框架（例如，LangChain, LlamaIndex）
- 数据摄取管道和ETL工具
- 监控和日志服务

这些依赖中的每一个都有自己的发布周期、潜在的API变更、安全漏洞和性能特征。

- **版本锁定与保持最新：** 您将面临经典的二难选择：为求稳定而锁定依赖版本，但可能错过安全补丁或有益更新；或者尝试保持最新，但可能带来集成难题和潜在回归。一种平衡的方法通常涉及在将更新推广到生产环境之前，在预生产环境中进行彻底测试。
- **安全补丁：** 任何组件都可能出现漏洞。向量数据库库中的CVE或LLM服务器所用Python版本中的安全问题都需要及时关注。这意味着需要有流程来监控漏洞披露并迅速测试和部署补丁。
- **级联故障：** 一个组件的故障或性能下降可能影响整个RAG管道。例如，LLM API提供商的延迟增加将直接影响您系统的响应速度。错误处理、重试和熔断机制变得极其重要。

### 不断变化的评估与监控需求

随着RAG系统成熟和其使用模式演变，您在启动时建立的指标和监控策略可能不足。

- **关注高级指标：** 初步评估可能侧重于检索精确度/召回率和基本的生成质量。随着时间推移，您可能需要事实性、语气一致性、无偏见或用户参与度的详细指标。收集这些指标的框架（例如，人工标注、基于模型的评估）也需要维护和可能升级。
- **评估数据中的漂移：** 用于离线评估的“黄金数据集”也可能变得过时或不能很好地代表当前的用户查询或文档特征。定期刷新和扩充这些数据集本身就是一项维护工作。
- **新的故障模式：** 随着系统在生产环境中遇到更多样化的输入和边缘情况，新颖的故障模式可能出现。您的监控必须足够灵活以检测这些新模式，并且您的调试工具集必须能够诊断它们。例如，知识库中一种新型的格式错误的文档可能导致索引错误，而这些错误只有在特定用户查询后才会出现。

### 跟上规模与成本

对1,000用户和10,000文档有效的方法，可能在100,000用户和1,000万文档的负载下力不从心。

- **架构限制：** 最初足够用的组件（例如，单节点向量 (vector)数据库、简单队列系统）可能成为瓶颈。为更大规模重新设计架构（例如，分片数据库、分布式任务队列）可能是一项大量的工程投入。
- **成本悄然增加：** 随着使用量的增长，LLM API调用、向量数据库托管、嵌入 (embedding)计算和数据存储等相关成本也随之增加。如果没有勤奋的监控和优化（第5章将介绍），这些成本可能意外地迅速增长。例如，低效的提示工程 (prompt engineering)导致过多的token使用，可能随着时间推移显著增加LLM账单。
- **规模化下的性能下降：** 数据量增加可能减慢检索速度。更高的查询量可能使服务基础设施承受压力。随着系统规模扩大，保持低延迟和高吞吐量 (throughput)需要持续的性能调优和容量规划。

### 用户期望与使用场景的变化

用户对RAG系统的理解和期望会发生变化。他们可能发现新的应用或要求更高的复杂程度。

- **适应新的查询类型：** 如果用户开始提出更复杂的、多跳的问题，或需要跨多个文档进行综合的查询，您的最初检索和生成策略可能不足。这可能需要引入更高级的技术，例如查询分解或迭代检索。
- **新功能请求：** 用户可能请求诸如对话记忆、与其他工具的集成或不同输出格式的功能。满足这些需求需要持续开发并引入新的维护考量。

### 文档与知识转移

最后，一个技术性不那么强但同样重要的难题是维护全面的文档并确保团队成员变动时的顺畅知识转移。一个只有一两个人了解的RAG系统是脆弱的。

- **实时更新的文档：** 系统架构、数据管道图、模型版本历史、故障排除指南和值班流程需要保持最新。
- **交叉培训：** 确保多名团队成员熟悉RAG系统操作和维护的不同方面，以降低风险。

成功应对这些长期维护难题需要积极主动的心态。它涉及建立实践、持续监控、定期性能审查，以及愿意根据需要调整和重构组件。后续章节将详细介绍具体的优化技术，这些技术不仅提高初始性能，而且从长远来看有助于RAG系统更易于维护且更具弹性。

## 参考资料

- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于构建、部署和维护机器学习系统的实用指南，涵盖数据管理、模型监控和MLOps等主题。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2024)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  对RAG系统进行全面综述，讨论其架构、方法和挑战，为操作问题提供背景。
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research (TMLR); DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提出一个在不同场景下评估语言模型的框架，有助于识别性能下降和长期漂移。
