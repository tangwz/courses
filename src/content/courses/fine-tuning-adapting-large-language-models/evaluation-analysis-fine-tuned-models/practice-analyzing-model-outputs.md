---
course: "fine-tuning-adapting-large-language-models"
chapter: "evaluation-analysis-fine-tuned-models"
lesson: "practice-analyzing-model-outputs"
sourceId: 3080
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-6-evaluation-analysis-fine-tuned-models/practice-analyzing-model-outputs"
title: "实践：分析模型输出中的错误"
description: "对微调模型的输出进行定性分析，以找出规律和错误。"
order: 9
plots: ["plots/3080-0.json"]
sourceHash: "2d302e7fe51374dda3282c1f8d8dcc0aafa315e6168bb8dc82c6a0d3dcda396b"
sourceCorrections: []
---

定量指标能宏观地展现模型性能，但它们常掩盖具体的失效方式和细节。定性分析，即对单个模型输出进行系统性检查，对于理解微调 (fine-tuning)模型在特定输入上 *如何* 成功或 *为何* 失败非常重要。本实践部分将指导您建立并执行此类分析，以识别重复出现的错误样式。此过程直接为您的数据、微调策略或后处理步骤的迭代改进提供指导。

### 建立分析框架

在检查输出前，请定义分析的范围和目标。您主要关心的是：

1. **指令遵循情况：** 模型对提示中请求的特定限制或格式的遵循程度如何？
2. **事实准确性：** 模型生成的信息是否准确，特别是在特定应用中？
3. **安全性和偏见：** 模型是否产生有害、有偏见或不当内容？
4. **特定任务的质量：** 对于摘要或翻译等任务，输出是否达到所需的质量标准（例如，涵盖性、忠实度、流畅性）？

您的目标将指导数据选择和错误类型的优先级。

接下来，选择一份有代表性的评估数据样本，包括输入提示和相应的模型生成内容。简单地使用随机抽样可能不够。考虑以下方法：

- **随机抽样：** 获取整体概览的基础方法。
- **分层抽样：** 确保不同类型输入（例如，不同指令复杂性、不同主题）的代表性。
- **失败案例抽样：** 如果初步指标显示某些子集表现不佳，则有意地对这些案例进行过采样。
- **对抗性抽样：** 包含专门用于挑战模型设计的输入（例如，已知会引起幻觉 (hallucination)或拒绝的提示）。

目标是样本量足够大，足以发现规律，但便于人工检查。通常，检查50-200个例子可提供不少有价值的信息。

### 构建错误分类体系

一个结构化的分类系统（即分类体系）对于一致地识别错误是必要的。没有它，分析会变得主观且难以汇总。您的分类体系应根据模型任务和之前定义的目标进行定制。从宽泛的类别开始，在遇到特定错误类型时再进行细化。

这里提供一个示例分类体系，它借鉴了本章前面讨论的想法：

**常见LLM错误类别：**

1. **指令遵循错误：**
   - `忽略的限制`：未遵循特定指令（例如，长度限制、格式）。
   - `错误理解的指令`：错误理解了指令。
   - `部分遵循`：遵循了部分指令，但遗漏了其他指令。
2. **事实错误 / 幻觉 (hallucination)：**
   - `事实不准确`：陈述了与外部知识不符的错误信息。
   - `捏造`：生成了貌似合理但完全虚构的信息。
   - `过时信息`：提供了曾经正确但现已过时的信息。
3. **相关性和连贯性错误：**
   - `偏离主题`：回复与提示不相关。
   - `通用/无信息量`：输出过于模糊或未提供具体信息。
   - `重复`：不必要地重复短语或句子。
   - `不连贯`：文本没有意义或逻辑有误。
4. **流畅性错误：**
   - `语法错误`：语法、拼写或标点问题。
   - `措辞笨拙`：文本语法正确但表达不自然。
5. **偏见与安全错误：**
   - `有害内容`：宣扬非法行为、仇恨言论等。
   - `刻板印象`：延续有害的刻板印象。
   - `不公平偏见`：对某个群体表现出偏见。
6. **格式错误：**
   - `格式不正确`：未能以指定格式（例如，JSON、Markdown 表格）生成输出。

根据您的具体需求调整此列表。例如，代码生成模型可能需要`语法错误`、`代码逻辑错误`或`代码效率低下`等类别。

### 标注过程

准备好样本数据和错误分类体系后，开始审查过程。对于每个输入-输出对：

1. **仔细阅读输入提示：** 理解预期任务和任何限制。
2. **评估模型输出：** 将输出与输入以及您对预期正确/理想回复的认知进行比较。
3. **识别错误：** 根据您的分类体系，对观察到的任何错误进行分类。一个输出可以有多个错误。
4. **添加备注（可选但推荐）：** 简要描述具体错误或提供背景信息。这有助于后续分析。
5. **（可选）分配严重程度：** 您可以分配严重程度级别（例如，轻微、中等、严重）以优先处理修复。

您可以使用简单的工具（如电子表格，CSV/Excel）执行此标注，或者如果可用，使用专门的标注平台。一致性很重要，尤其是有多个审阅者参与时。考虑在数据的一个子集上计算标注者间一致性（IAA）分数（如Cohen's Kappa），以确保分类体系被统一应用。

**标注示例（在电子表格中）：**

| 输入提示 | 模型输出 | 错误类别 | 备注 | 严重程度 |
| --- | --- | --- | --- | --- |
| 将以下文本总结为正好50字 | 所提供的文档讨论了电池技术的进步... (75字) | `指令遵循错误` (忽略的限制) | 超过字数限制。 | 中等 |
| 澳大利亚的首都是哪里？ | 澳大利亚的首都是悉尼。 | `事实错误` (事实不准确) | 错误的首都。正确答案是堪培拉。 | 严重 |
| 简单解释量子纠缠。 | 量子纠缠是两个粒子以一种“幽灵般”的方式连接起来... | *无* | 简单、准确的解释。 | 不适用 |
| 生成Python代码来排序列表`x`。 | `sorted_list = x.sort()` | `流畅性错误` (语法错误) | `sort()` 会原地修改并返回 None，需要 `sorted(x)` | 中等 |
| 翻译成法语：“Hello world” | Salut monde | `流畅性错误` (措辞笨拙) | 正确的是“Bonjour le monde”。“Salut”在此处过于随意。 | 轻微 |

### 分析和可视化错误

标注完成后，汇总结果。计算每个错误类别的频率。这种定性结果的定量总结有助于找出最常见的问题。

例如，您可能会发现30%的错误是`事实不准确`，25%是`忽略的限制`，15%是`通用/无信息量`，其余分布在其他类别中。

可视化能使这些规律更清晰。一个简单的条形图常能有效显示错误类型的分布。



![抽样输出中的错误类型分布](plots/3080-0.json)



> 错误频率分析突出显示了评估样本中最常见的问题。

寻找相关性。某些类型的提示是否总是导致特定错误？例如，带有多个限制的复杂指令是否经常导致`部分遵循`错误？开放式问题是否常引发`通用/无信息量`的回复？这些信息非常有价值。

### 案例演练：医疗报告摘要器

假设您微调 (fine-tuning)了一个LLM来总结患者出院记录，要求将诊断、药物和后续医嘱提取为结构化格式。

**输入1 (提示)：**

```
将此出院记录中的重要信息总结为JSON：
患者因肺炎入院。每日静脉注射左氧氟沙星750mg，持续5天。反应良好。出院时口服左氧氟沙星500mg，每日一次，持续7天。1周后由家庭医生史密斯医生随访。建议休息和补充水分。
```

**模型输出1：**

```json
{
  "diagnosis": "Pneumonia",
  "medications": ["levofloxacin 750mg IV", "levofloxacin 500mg Oral"],
  "follow_up": "PCP Dr. Smith in 1 week",
  "advice": "Rest and hydration"
}
```

**分析1：** 优秀。遵循了指令，信息提取正确，使用了指定格式。**错误：无**。

**输入2 (提示)：**

```
创建一个JSON摘要：
患者因胸痛就诊，排除心肌梗死。诊断为肋软骨炎。处方布洛芬600mg，每日三次，必要时服用。无需随访，除非症状恶化。患者青霉素过敏。
```

**模型输出2：**

```json
{
  "diagnosis": "Costochondritis",
  "treatment": "Ibuprofen 600mg TID",
  "follow_up": "None specified"
}
```

**分析2：** 良好，但遗漏了重要信息。

- **错误类别：** `指令遵循错误` (部分遵循)。
- **备注：** 未能提取关于青霉素过敏的重要细节。这可能是很严重的问题。需要改进以捕捉提及的过敏或禁忌症。**严重程度：严重**。

**输入3 (提示)：**

```
使用JSON进行总结：
68岁男性，有高血压、2型糖尿病史。因COPD急性加重入院。接受类固醇、雾化器、BiPAP治疗。出院时服用泼尼松渐减剂量，使用信必可吸入器。2周后到呼吸科诊所随访。继续服用居家药物（赖诺普利、二甲双胍）。
```

**模型输出3：**

```json
{
  "patient_profile": "68yo male, Hx HTN, DM2",
  "admission_reason": "COPD Flare",
  "discharge_meds": ["prednisone", "Symbicort", "Lisinopril", "Metformin"],
  "plan": "Pulmonology F/U 2wk"
}
```

**分析3：** 大体正确，捕获了重要信息。

- **错误类别：** `事实错误` (轻微不准确/不精确)。
- **备注：** 将“泼尼松渐减剂量”简化为“泼尼松”。虽然不完全错误，但它损失了细节。可以在临床精确性上改进。**严重程度：轻微**。

### 综合发现并推动改进

这种定性分析，以结构化分类体系和仔细审查为基础，提供了纯粹基于指标的评估可能遗漏的可操作信息。在我们的医疗摘要案例中，我们识别了捕捉否定限制（过敏）和潜在临床细节丢失的问题。

这些发现表明了具体的下一步：

- **数据增强：** 在微调 (fine-tuning)数据集中添加更多明确包含过敏信息或要求捕获治疗细节（如渐减剂量）的示例。
- **提示工程 (prompt engineering)：** 修改指令模板，以便在提及过敏或禁忌症时专门要求捕获。
- **模型再微调：** 使用增强数据集进一步微调模型。
- **后处理规则：** 实施检查以确保如果源文本中出现关键词，则填充特定字段（如过敏）。

人工分析模型输出是一个迭代过程。随着您优化模型和数据，重复分析以跟踪改进并识别新的潜在问题。这是构建可靠且有效的微调模型的一个基本组成部分。

## 参考资料

- [Holistic Evaluation of Language Models](https://doi.org/10.48550/arXiv.2211.09110) — Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, Yuta Koreeda (2023)
  Journal: Transactions on Machine Learning Research; DOI: [10.48550/arXiv.2211.09110](https://doi.org/10.48550/arXiv.2211.09110)
  介绍了一种评估语言模型各项能力和风险的框架，涵盖了多种评估方法。
- [Natural Language Processing with Transformers: Building Innovative Applications with 🤗 Transformers](https://learning.oreilly.com/library/view/natural-language-processing/9781098136796/) — Lewis Tunstall, Leandro von Werra, Thomas Wolf (2022)
  Publisher: O'Reilly Media
  一本关于Transformer模型开发和评估的实用指南，涵盖了微调策略和评估方法。
