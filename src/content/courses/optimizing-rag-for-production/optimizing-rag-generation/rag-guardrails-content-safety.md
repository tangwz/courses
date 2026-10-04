---
course: "optimizing-rag-for-production"
chapter: "optimizing-rag-generation"
lesson: "rag-guardrails-content-safety"
sourceId: 6198
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-3-optimizing-rag-generation/rag-guardrails-content-safety"
title: "实施防护措施与内容安全"
description: "将防护措施与内容审查机制纳入RAG系统，以确保生成内容的安全性、适当性和合规性。"
order: 6
plots: []
sourceHash: "718d295bd6da62cb52c8460faf2aed9d52ba1018427f55a381ae6991d1094585"
sourceCorrections: []
---

尽管蒸馏和量化 (quantization)等技术提升了大型语言模型（LLM）的效率，但在生产RAG系统中，其生成输出的安全性与适当性仍然是主要考量。不受控制的LLM行为可能导致有害内容、错误信息、敏感数据泄露或违反使用政策的响应。因此，实施一套全面的防护措施和内容安全框架，不仅仅是附加功能，更是部署可信赖、可靠RAG应用的必要条件。将详细阐述如何结合此类机制的策略，以确保您的RAG系统输出安全、适当且合规。

### 防护措施系统的架构考量

防护措施策略涉及RAG管道中的多处干预，主要包括审查用户输入和验证LLM生成的输出。

#### 预处理：输入防护措施（用户查询审查）

输入防护措施会在用户提供的提示被检索器或LLM处理前进行分析。目标是在管道早期识别并减轻潜在风险。常见的输入防护措施包括：

- **提示注入检测：** 识别通过对抗性提示操纵LLM行为的尝试。这可以是从针对已知注入字符串的简单模式匹配，到使用经过训练的专用分类模型来检测恶意意图或异常查询的更复杂方法。
- **个人身份信息（PII）修订：** 检测并移除或遮蔽用户查询中的PII（例如，姓名、地址、社会安全号码），以防止其被记录、由LLM处理，或在LLM回显输入时无意中包含在生成响应中。为此，通常会采用针对PII检测进行微调 (fine-tuning)的高级命名实体识别（NER）模型。
- **主题一致性：** 确保用户的查询属于RAG应用的预期范围或内容。超出范围的查询可以被标记 (token)或拒绝，从而防止资源浪费和不相关响应。
- **有害内容检测：** 筛选输入中是否存在辱骂性语言、仇恨言论或其他违反政策的内容。

#### 后处理：输出防护措施（LLM响应验证）

输出防护措施会在LLM生成的响应交付给用户前对其进行检查。这些措施对于捕获LLM本身产生的不良内容具有关键作用。必要的输出防护措施包括：

- **内容过滤：** 类似于输入有害内容检测，此功能扫描LLM的输出，以查找毒性、亵渎或任何其他不良内容类别。多标签分类器可以对不同类型的内容提供精细控制。
- **事实性和依据检查：** 尽管缓解幻觉 (hallucination)是一个更宽泛的话题（在“缓解RAG输出中的幻觉”中有所介绍），但输出防护措施可以包括检查以确保响应与检索到的上下文 (context)或已知事实保持一致，并标记或阻止未经证实的主张。
- **敏感信息泄露预防：** 将LLM输出与包含专有信息的机密术语、模式或知识库列表进行交叉比对，以防止其意外泄露。
- **合规性：** 验证输出是否符合监管要求，例如确保存在财务建议免责声明，或医疗信息以适当方式呈现。
- **格式和风格一致性：** 确保输出符合预定义的格式规则、长度限制或风格指南，如果仅通过提示工程 (prompt engineering)无法完全管理这些方面。

> 数据流经结合了输入和输出防护措施的RAG系统。

### 防护措施的实现技术

防护措施可以采用多种技术实现，通常是组合使用，以创建分层安全。

#### 确定性检查：正则表达式和关键词过滤

最简单的防护措施形式涉及使用正则表达式（regex）和关键词列表来检测不良模式或术语。

- **优点：** 易于实现、可解释性强、延迟低，对明确定义的静态模式（例如，阻止特定亵渎词或已知恶意代码片段）有效。
- **缺点：** 脆弱，容易被措辞的细微变化或混淆技术绕过。维护庞大的列表可能变得繁琐。
  例如，一个用于输入防护措施的简单关键词过滤器：

```python
def keyword_filter(text, banned_keywords):
    text_lower = text.lower()
    for keyword in banned_keywords:
        if keyword in text_lower:
            return True # 标记内容
    return False

# 示例用法：
# user_query = "告诉我如何构建一个不当设备。"
# banned = ["不当设备", "有害主题"]
# if keyword_filter(user_query, banned):
# print("输入被关键词过滤器标记。")
```

这种方法是第一道防线，但对于生产系统来说，通常单独使用是不够的。

#### 基于模型的安全分类器

可以训练专门的机器学习 (machine learning)模型来对文本进行分类，以识别各种安全属性（例如，毒性、PII存在、情感、主题相关性）。

- **优点：** 比正则表达式/关键词方法更能适应语言变化。可以学习复杂模式并适应不断变化的语言使用。
- **缺点：** 需要训练数据，延迟可能高于确定性检查，并可能引入自身的偏见或错误（假阳性/假阴性）。这些模型的性能非常重要。
  例如，基于BERT的分类器通常会针对毒性检测或PII识别等任务进行微调 (fine-tuning)。谷歌的Perspective API或Azure内容安全等服务提供预构建的基于模型的解决方案。

#### 借助LLM进行高级审查

一种越来越常用的技术是使用LLM本身作为“评判者”或“审核者”来评估输入，或者更常见的是评估主要RAG LLM的输出。这涉及精心设计一个特定提示，指示评判LLM根据一组安全准则或原则来评估文本。

- **优点：** 比简单模型或规则更能理解上下文 (context)和微妙之处。高度灵活，可以适应复杂的复杂政策。
- **缺点：** 由于额外的LLM调用，会引入显著的延迟和成本。评判LLM本身需要可靠并能抵抗操纵。此外，评判LLM也存在产生幻觉 (hallucination)或错误解释准则的风险。

基于LLM的输出审查提示示例：

```
你是一个安全评估模型。根据以下生成的文本，评估它是否违反以下任何政策：
1.  不包含仇恨言论或煽动暴力。
2.  不泄露个人身份信息（姓名、电子邮件、电话号码）。
3.  与用户关于[context_中的主题]的原始查询相关。
生成文本："[LLM_OUTPUT_HERE]"
根据上述政策，文本是否安全合规？请回答“SAFE”或“UNSAFE”，如果是不安全的，请附上简要解释。
```

该评判LLM的响应可用于决定是否通过、阻止或修改主要LLM的输出。

### 将防护措施纳入RAG工作流

有效的防护措施不仅关乎所使用的技术，还关乎其在RAG系统中的策略性结合。

#### 策略性部署与排序

如图所示，输入防护措施在用户查询到达RAG核心逻辑之前对其进行操作，而输出防护措施则对LLM的响应进行操作。在这些阶段内，可以链接多个防护措施。

- **效率：** 将计算成本较低、速度较快的检查（例如，正则表达式过滤器）置于成本较高的检查（例如，LLM作为评判者）之前。如果早期检查标记 (token)出问题，后续检查可能会被跳过。
- **逻辑：** 某些检查可能依赖于其他检查的结果。例如，PII修订可能在主题分类模型对输入运行之前进行。

#### 定义行动策略

当防护措施被触发时，系统需要明确的行动策略。常见行动包括：

- **阻止：** 完全拒绝输入或丢弃输出。这对于仇恨言论或明显的提示注入尝试等严重违规行为很常见。
- **净化/修改：** 更改输入或输出以移除冒犯性内容。例如，从输入查询中修订识别出的PII，或从LLM响应中删除有问题语句。这需要仔细实施，以避免扭曲含义或制造新问题。
- **标记/警报：** 允许内容通过，但记录问题并可能提醒管理员或人工审查团队。这可能适用于边缘情况或用于收集新型问题内容的数据。
- **请求澄清/重述（针对输入）：** 如果用户的查询模糊或轻微有问题，可以要求他们重述，而不是直接阻止。
- **回退响应（针对输出）：** 如果LLM输出被阻止，则提供预定义的安全通用响应（例如，“抱歉，我无法回应该请求。”）。

配置系统应允许根据每个防护措施以及可能的用户类型或应用上下文 (context)灵活定义这些策略。

### 生产防护措施的运行方面

在生产环境中部署防护措施带来了一些运行挑战。

#### 性能影响与优化

每个防护措施都会增加延迟。对于实时RAG应用来说，这是一个重要问题。

- **优化单个防护措施：** 使用高效模型、优化过的正则表达式和预编译模式。
- **异步处理：** 一些非关键的防护措施检查（例如，用于后续审查的日志记录）可以异步执行，以避免阻塞主要响应路径。
- **缓存：** 如果输入或输出重复，缓存的防护措施决策可能可以重复使用，尽管这需要仔细考虑上下文 (context)。

#### 监控、日志记录和迭代改进

防护措施并非“一劳永逸”的解决方案。它们需要持续关注。

- **日志记录：** 记录所有防护措施触发事件，包括导致触发的输入/输出、激活的具体防护措施以及采取的行动。这对于审计、调试和了解防护措施性能必不可少。
- **监控：** 追踪以下指标，例如：
  - 不同防护措施触发的频率。
  - 假阳性率（FPR）：安全内容被错误标记 (token)的频率。
  - 假阴性率（FNR）：不安全内容被遗漏的频率。
  - 每个防护措施造成的延迟。
- **迭代改进：** 使用日志数据和性能指标更新和改进防护措施规则、模型和策略。新的攻击向量 (vector)和滥用模式不断出现，需要防护措施进行调整。A/B测试不同的防护措施配置有助于优化其有效性。

#### 权衡：平衡安全与用户体验

防护措施的严格性与RAG系统的实用性/自由度之间存在固有的紧张关系。过于激进的防护措施可能导致高假阳性率，通过阻止合法查询或响应来让用户感到沮丧。反之，薄弱的防护措施可能允许有害或不当内容通过。找到恰当的平衡通常是应用特有的，可能需要根据用户反馈和风险评估进行持续调整。

### RAG中防护措施的应用示例

某些防护措施对RAG系统特别相关。

#### 对抗提示注入和系统滥用

RAG系统与其他LLM应用一样，易受提示注入影响，用户试图覆盖系统指令或导致意外行为。输入防护措施可以设计用于识别常见的注入模式（例如，“忽略之前的指令并...”），或使用训练过的模型来识别对抗性输入。输出防护措施还可以检查LLM的响应是否表明其原始指令已被颠覆。

#### 强制执行数据隐私和合规指令

如果您的RAG系统处理或访问敏感数据，防护措施对于合规性必不可少。

- **输入：** PII修订是主要的输入防护措施。
- **输出：** 防护措施可以阻止LLM回吐知识库中的敏感信息，即使这些信息是检索到的上下文 (context)的一部分。这可能涉及根据已知敏感数据模式检查输出，或在使用差异隐私技术向LLM呈现上下文的方式（尽管这更多是一种架构选择）。

#### 使输出与特定领域约束保持一致

对于专门的RAG应用（例如，医疗、法律、金融），输出通常必须严格遵守特定方面的规则或免责声明。

- 输出防护措施可以验证是否存在所需的免责声明（例如，“这不是财务建议”）。
- 它们可以检查LLM的响应是否避免在需要谨慎时做出明确声明（例如，避免医疗诊断）。
- 它们可以确保所使用的语言适用于目标受众和特定方面。

实施有效的防护措施是一个持续过程，它结合了技术解决方案、仔细的政策定义和持续监控。通过周到地设计和结合这些安全机制，您可以大幅提高生产RAG系统的可信度和可靠性，为其负责任的采纳提供途径。有了这些安全网，接下来的重点就可以转向对生成内容整体质量进行更广泛的评估，正如下一节所讨论的那样。

## 参考资料

- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, et al. (2022)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073)
  讨论了通过AI反馈使LLM与人类价值观和安全原则保持一致的方法。这是一种训练LLM在没有大量人工干预的情况下调节自身行为的方法，与LLM作为评估者的技术相关。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew F. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  提供了一个评估LLM的框架，涵盖安全、公平和稳健性等多个标准，为评估护栏所解决的整体可靠性和风险提供了背景。
- [Google Cloud Perspective API Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQGgMDt5jE1wuG01rfcy3WQDcp3cUM_Qt19j0UNeIAio2V2ysfxRlvHIOKeIRKMyNLqXy6U7fTfTY5qM69b-GSd8x-IcI52bYCyDVoAg8tgeo6-VbJvCqo1lF-EJRPj0r4jQsPOu87a5fkZ-n0bJuhfegzBfmPG4LQ==) — Google Cloud (2024)
  Publisher: Google Cloud
  广泛使用的内容审核API的官方文档，该API使用机器学习模型检测各种有害或有毒内容类别，是输出护栏中基于模型的分类器的一个实例。
