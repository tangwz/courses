---
course: "prompt-engineering-llm-application-development"
chapter: "prompt-design-iteration-evaluation"
lesson: "evaluating-prompt-performance"
sourceId: 3008
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-3-prompt-design-iteration-evaluation/evaluating-prompt-performance"
title: "评估提示词表现"
description: "评估特定提示词生成响应的质量和可靠性的方法。"
order: 4
plots: ["plots/3008-0.json"]
sourceHash: "ddd270700a8d110f2416501df4976e4652a5da6cd78689b7329daa168abf82ee"
sourceCorrections: []
---

如前所述，撰写和优化提示词 (prompt)是一个迭代过程。但您如何客观地判断优化后的提示词是否真的比之前的版本*更好*？仅仅查看少量输出不足以构建可靠的应用。这就是系统化提示词评估的用处所在。它涉及为您的特定任务界定“好”的含义，然后衡量您的提示词在各种输入下实现该目标的程度。

有效的评估超越了主观印象，提供具体数据来指导您的提示词优化工作。如果没有评估，您可能会冒着在某种输入上提升表现，却在其他输入上降低表现的风险，或者优化了一个与应用需求不符的质量。

### 界定成功标准

要评估提示词 (prompt)，首先必须界定什么是成功的输出。这些标准很大程度上取决于LLM正在执行的任务。以下是需要考虑的常见维度：

1. **准确性和正确性：** 所提供的信息是否事实正确？输出是否准确反映了输入数据？这对于问答、数据提取和基于所提供情境的总结等任务很重要。
2. **相关性：** 输出是否与输入查询或指令相关？它是否直接满足了用户的需求？这对于搜索、问答和推荐系统来说很重要。
3. **连贯性和流畅性：** 生成的文本是否语法正确、易于理解且听起来自然？这适用于大多数生成式任务，如写作辅助或聊天机器人回复。
4. **格式遵循度：** 输出是否符合要求的结构（例如，JSON、Markdown列表、特定XML模式）？当LLM输出需要被下游系统以程序方式解析时，这一点非常必要。
5. **完整性：** 输出是否包含基于提示词和情境的所有必需信息？
6. **简洁性：** 输出是否适当简洁，避免不必要的冗余？有时少即是多。
7. **安全性和无害性：** 输出是否避免生成有害、有偏见或不当内容？这需要仔细监控，特别是对于面向用户的应用。
8. **指令遵循度：** LLM对提示词中指定的所有限制和指令遵循得如何？

您可能需要根据应用的目标优先考虑这些标准中的一部分。例如，对于API调用生成器而言，严格的格式遵循可能比流畅性更重要，而对于创意写作助手而言则可能相反。

### 评估方法

一旦您界定了成功标准，就需要衡量对照这些标准的表现的方法。评估技术从人工评估到全自动化指标不等。

#### 人工评估

直接人工判断通常被认为是黄金标准，特别是对于连贯性、相关性或语气等主观质量。常见方法包括：

- **李克特量表：** 评估员根据数字量表（例如，1-5分）对特定标准（如“准确性”或“帮助性”）对输出进行评分。
- **排名：** 评估员并排比较不同提示词 (prompt)版本的输出，并从最好到最差进行排名。
- **二元选择：** 评估员只需根据特定标准选择两个输出中（例如，来自提示词A与提示词B）哪个更好。
- **合格/不合格：** 评估员判断输出是否达到特定标准的最低质量要求。

**优点：** 捕捉细节，很好地评估主观质量，适用于复杂标准。
**缺点：** 耗时、昂贵，可能存在评估员间差异性（不同人评分不同），难以大规模应用。

为了提高人工评估的一致性，请向您的评估员提供清晰的评分指南和示例（评分标准）。

#### 自动化指标

自动化指标具有可扩展性和速度，使其适用于评估大量输出或集成到CI/CD流程中。

- **精确匹配 (EM)：** 检查生成的输出是否与预定义参考答案完美匹配。适用于约束性很强的任务（例如，提取特定日期），但对生成式任务通常过于严格。
- **关键词识别 / 正则表达式：** 检查输出中是否存在特定关键词、模式或值。可用于验证某些实体的包含或遵循简单格式。
- **语义相似度：** 使用文本嵌入 (embedding)来衡量生成输出与参考答案或理想响应档案之间的意义相似度。Sentence-BERT等模型可以生成嵌入，通常使用余弦相似度来比较它们。这超越了表面匹配。
  - *示例：* 计算LLM生成的摘要嵌入与人工编写的参考摘要嵌入之间的余弦相似度。分数越高（$0$到$1$），表示语义重叠度越大。

  
  $$
  \text{相似度} = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}
  $$
  

  > 这里 $\vec{A}$ 和 $\vec{B}$ 是生成文本和参考文本的嵌入向量 (vector)。
- **格式验证：** 使用库（如Python中的Pydantic用于JSON）或模式验证器以程序方式检查输出是否符合所需结构。这直接衡量格式遵循度。
- **特定任务指标：** 针对特定NLP任务存在已有指标，例如用于摘要生成的ROUGE（面向召回的概要评估替代表）或用于机器翻译的BLEU（双语评估替代表）。尽管它们在其特定领域中功能强大，但可能并不总是与您的总体应用目标完美契合。

**优点：** 速度快、可扩展、客观（结果一致）、设置后成本效益高。
**缺点：** 可能无法很好地捕捉细节（嵌入式方法除外），可能会被“钻空子”（输出是为了指标优化而非质量），需要参考答案或特定验证逻辑。

#### LLM作为评判者

一种越来越常见的方法是使用另一个LLM（通常是一个更大、能力更强的LLM）来评估主LLM的输出。您向评估LLM提供原始输入、生成的输出以及定义评估标准的提示（例如，“根据所提供的情境，对以下回复的事实准确性按1-5分制评分。解释您的理由。”）。

**优点：** 比简单的自动化指标能更好地评估主观质量，大规模应用时可能比人工评估更快、更便宜，通过提示可调整标准。
**缺点：** 评估质量很大程度上取决于评估LLM和评估提示，可能继承评估LLM的偏见，产生API费用，如果使用相同模型系列可能存在自我强化偏见。

### 建立评估数据集

仅凭一两个输入来评估提示词 (prompt)是不够的。您需要一组具有代表性的测试用例，通常称为“评估数据集”或“黄金数据集”，涵盖您的应用预期处理的各种输入。此数据集应包括：

- **典型案例：** 代表常见使用模式的输入。
- **边界案例：** 可能导致失败的具有挑战性或不寻常的输入（例如，模糊查询、信息缺失的输入、非常长的输入）。
- **负面案例：** 旨在测试安全性、拒绝能力或处理超出范围请求的输入。

对于评估数据集中的每个输入，您可能还需要定义参考输出或预期行为，以便与LLM的生成内容进行比较（特别是对于自动化指标）。随着您发现新的故障模式，随时间维护并可能扩展此评估数据集是很重要的。

### 组合方法

通常，最有效的评估策略结合多种方法。您可以使用：

1. **自动化指标**（如格式验证和语义相似度）在开发过程中对大型评估数据集提供快速反馈。
2. **有针对性的人工评估**（如并排比较）用于特定主观标准或较小、多样化的挑战性案例子集。
3. **LLM作为评判者**作为潜在的中间步骤，以便在投入大量人工审查之前获得大规模主观反馈。

目标是获得提示词 (prompt)在不同维度和输入下的全面视图。随着您迭代优化提示词，随时间跟踪这些评估结果可提供改进（或退步）的客观证据，并有助于验证设计决策。



![提示词表现对比（平均分）](plots/3008-0.json)



> 评估数据集上的平均得分显示，提示词V2在准确性和格式遵循度上相较于V1有显著提升，同时保持了相关性。

通过系统地界定标准、选择合适方法、使用评估数据集并跟踪结果，您可以从猜测变为清楚了解提示词的表现，并自信地改进它们。这种结构化方法是构建可靠有效LLM应用的核心所在。

## 参考资料

- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提供了一个全面的框架，用于评估大型语言模型在各种能力、场景和指标下的表现，为理解模型性能和局限性提供系统方法。
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://aclanthology.org/D19-1410/) — Nils Reimers and Iryna Gurevych (2019)
  Journal: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP); Publisher: Association for Computational Linguistics; Pages: 3982–3992; DOI: [10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410)
  提出了一种用于生成具有语义意义的句子嵌入的方法，可用于语义相似性等任务，直接支持了自动化评估指标部分。
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311–318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  一篇介绍BLEU指标的奠基性论文，该指标广泛用于评估机器翻译系统生成文本的质量，也适用于其他生成式任务。
