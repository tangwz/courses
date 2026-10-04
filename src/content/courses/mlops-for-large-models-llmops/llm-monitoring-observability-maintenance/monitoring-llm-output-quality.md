---
course: "mlops-for-large-models-llmops"
chapter: "llm-monitoring-observability-maintenance"
lesson: "monitoring-llm-output-quality"
sourceId: 3203
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-5-llm-monitoring-observability-maintenance/monitoring-llm-output-quality"
title: "监控 LLM 输出质量（有害性、偏见）"
description: "实施策略以监控 LLM 输出的有害性、偏见和相关性等（不侧重于伦理）。"
order: 5
plots: ["plots/3203-0.json"]
sourceHash: "90903f794a4cb258cc9ddc8ee16fae5a0bc380b5e87e0a78ff02e1adc2c559cc"
sourceCorrections: []
---

虽然延迟和成本等运行指标是必需的，但它们不足以评估大型语言模型在生产环境中的实际用途和安全性。LLM 生成自由形式的文本，确保其输出符合预期的质量标准是一项持续的运行挑战。与生成分类或数值预测的传统模型不同，LLM 输出可能表现出细微而复杂的故障模式，包括生成有害内容或延续社会偏见。监控这些方面需要专门的策略，它们比标准的性能跟踪更进一步。

### 理解 LLM 输出中的有害性和偏见

在实施监控之前，定义我们要寻找什么很重要：

- **有害性：** 指粗鲁、不尊重或不合理的语言，可能导致某人离开对话。这包括仇恨言论、侮辱、威胁和露骨的性内容。具体定义通常需要根据应用的上下文 (context)和用户社区准则进行调整。监控有害性通常是维护平台安全和用户信任的必要条件。
- **偏见：** 包括延续刻板印象或与性别、种族、民族、宗教、年龄或其他特征相关的不公平表述。偏见可能表现得不明显，例如将某些职业主要与某一性别关联，或使用不同的形容词来描述不同人口群体执行的相似行为。发现和监控偏见对于公平性以及减轻表述方面的损害非常重要。

监控这些维度并非通过监控系统本身强制执行特定的伦理观点，而是为了根据预设的准则或预期特征，使模型的行为透明化，从而能就缓解或干预做出明智的决策。

### 监控输出质量的方法

可以采用多种技术方法，通常是结合使用，来监控 LLM 输出质量：

1. **辅助分类模型：**
   一种常用方法是训练独立的、较小的机器学习 (machine learning)模型，专门用于将文本片段分类为有害或无害，或识别潜在的偏见指标。

   - **实施：** 这些分类器可以将 LLM 输出（或其样本）作为输入。它们可能是较小语言模型（如 DistilBERT 或 RoBERTa）的微调 (fine-tuning)版本，或者是在文本嵌入 (embedding)上训练的更简单的模型。训练需要反映有害性或偏见目标定义的标注数据集。
   - **优点：** 可以通过关键词匹配捕捉模式。可以独立于主 LLM 进行更新。
   - **缺点：** 需要标注数据进行训练和评估。分类器本身可能存在偏见或局限性。增加了监控过程的计算开销和潜在延迟。分类器的维护（再训练、漂移检测）成为另一项 MLOps 任务。
2. **基于规则的系统和黑名单：**
   更简单的方法是使用预定义的规则、正则表达式或与不良内容相关的关键词/短语列表。

   - **实施：** 根据已知有害术语、侮辱性词语或表示刻板印象的模式列表检查 LLM 输出。
   - **优点：** 计算成本低，易于实施和理解。能有效阻止明确的有害语言。
   - **缺点：** 脆弱，容易被拼写错误、同义词或上下文 (context)绕过。跨语言和不断变化的俚语的列表维护负担重。可能导致高误报率（例如，阻止有关有害性的讨论）。
3. **统计偏见分析：**
   对于偏见监控，统计技术可以提供定量指标，尽管它们需要仔细应用和解释。

   - **表征指标：** 如果输入或输出与人口群体相关（例如，根据姓名生成传记），则跟踪提及或关联的频率。例如，将生成的男性化姓名与女性化姓名所描绘职业的分布与基线进行比较，并进行监控。
   - **关联测试：** 从词嵌入分析（如词嵌入关联测试 WEAT）中改编的技术可以应用于生成的文本语料库，以测量术语之间的关联（例如，性别术语和职业术语）。计算可能涉及比较生成文本中相关术语嵌入之间的余弦相似度。例如，测量 {$he, him, his$} 和 {$doctor, engineer$} 之间与 {$she, her, hers$} 和 {$doctor, engineer$} 之间的平均相似度。显著差异可能表示偏见。设 $A$ 和 $B$ 是两组目标术语（例如，男性/女性术语），$X$ 和 $Y$ 是两组属性术语（例如，职业/家庭术语）。简化的关联度量 $S(A, B, X, Y)$ 可以是：
     
     $$
     S(A, B, X, Y) = \sum_{a \in A} \text{平均}_{x \in X} \text{相似度}(a, x) - \sum_{a \in A} \text{平均}_{y \in Y} \text{相似度}(a, y) - (\sum_{b \in B} \text{平均}_{x \in X} \text{相似度}(b, x) - \sum_{b \in B} \text{平均}_{y \in Y} \text{相似度}(b, y))
     $$
     
     其中 $\text{相似度}$ 表示一种相似度指标，如从 LLM 输出或内部表示中获得的词嵌入之间的余弦相似度。非零值表明存在差异关联。
   - **优点：** 提供可随时间跟踪的定量指标。可以揭示系统性偏见。
   - **缺点：** 需要仔细定义群体和属性。对术语和指标的选择很敏感。解释可能复杂且依赖于上下文。可能需要访问嵌入或特定的模型内部。相关性不代表因果关系；这些指标是指示器，而不是有害偏见的最终证据。
4. **外部内容审核服务：**
   使用专门的第三方 API 进行内容审核可以减轻一些复杂性。

   - **实施：** 将 LLM 输出发送到外部 API 端点，该端点返回有害性、仇恨言论等的得分或标签。
   - **优点：** 能够访问可能复杂、在大型多样化数据集上训练的模型。减少内部开发工作。
   - **缺点：** 成本影响（按 API 调用计费）。潜在的延迟增加。数据隐私问题（将模型输出发送给第三方）。依赖供应商的定义和模型质量。

### 将质量监控集成到 LLMOps 流水线中

有效的监控需要将这些技术集成到您的运行工作流中：

- **采样：** 分析每一个 LLM 输出通常成本过高且不必要。实施智能采样策略：
  - **随机采样：** 分析固定比例的输出。
  - **分层采样：** 确保覆盖不同的用例、用户群或提示类型。
  - **异常/边缘情况采样：** 将分析重点放在异常长/短、生成概率得分低或触发其他警报的输出上。
- **异步处理：** 质量检查，特别是涉及分类器模型或外部 API 的检查，可能会增加延迟。在向用户发送主响应后，异步执行这些检查，并记录结果以供后续分析和汇总。
- **警报和仪表盘：** 当质量指标超过预设阈值时（例如，有害性率 > 1%，偏见指标显著变化）配置警报。在监控仪表盘上可视化趋势。一个简单的仪表盘可以跟踪有害性分类器标记 (token)的输出百分比随时间的变化。



![LLM 输出有害性率随时间变化](plots/3203-0.json)



> 跟踪有害性分类器标记的采样输出百分比有助于识别模型行为的趋势或退化。

### 输出质量监控中的挑战

监控 LLM 输出质量本身就具有挑战性：

- **上下文 (context)依赖性：** 有害性和偏见高度依赖于对话上下文、用户意图和文化规范。自动化系统在处理这些方面存在困难。
- **主观性：** 人工标注者通常对特定输出是否有害或有偏见存在分歧，这使得为训练或评估创建完全可靠的真实标签变得困难。
- **语言演变：** 俚语、暗语和对抗性措辞不断演变，需要持续更新基于规则的系统和分类器。
- **可扩展性：** 每天处理可能数百万或数十亿的输出需要高效且可扩展的监控基础设施。

“监控 LLM 输出质量并非一次性任务，而是一个持续的过程。它涉及选择合适的方法，将其集成到 LLMOps 流水线中，仔细解释结果，并利用所得信息来指导模型更新、提示词 (prompt)优化或安全机制调整，这些通常直接与后续讨论的反馈循环相关联。它使得人们能清楚地看到 LLM 在复杂的交互环境中是否按预期运行。”

## 参考资料

- [Semantics derived automatically from language corpora contain human-like biases](https://doi.org/10.1126/science.aal4230) — Aylin Caliskan, Joanna J Bryson, Arvind Narayanan (2017)
  Journal: Science; Publisher: American Association for the Advancement of Science; Volume: 356; Pages: 183-186; DOI: [10.1126/science.aal4230](https://doi.org/10.1126/science.aal4230)
  介绍了词嵌入关联测试 (WEAT)，并展示了广泛使用的词嵌入如何表现出类似人类的偏见，为文本中提到的定量偏见检测提供了基础方法。
- [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://doi.org/10.1145/3442188.3445922) — Emily M. Bender, Timnit Gebru, Angelina McMillan-Major, Shmargaret Shmitchell (2021)
  Journal: Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency; Publisher: Association for Computing Machinery; Pages: 610-623; DOI: [10.1145/3442188.3445922](https://doi.org/10.1145/3442188.3445922)
  一篇探讨大型语言模型伦理和社会风险的开创性论文，涵盖偏见延续和潜在有害输出，强调了监测的必要性。
- [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew T. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  介绍了HELM，一个用于评估语言模型多项标准的综合框架，包括公平性和毒性，为生产环境中的持续监测提供了相关方法。
