---
course: "fine-tuning-adapting-large-language-models"
chapter: "evaluation-analysis-fine-tuned-models"
lesson: "limitations-standard-metrics"
sourceId: 3063
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-6-evaluation-analysis-fine-tuned-models/limitations-standard-metrics"
title: "传统自然语言处理评估指标的局限性"
description: "批判性分析BLEU、ROUGE和准确率等指标在评估大型语言模型复杂能力时的不足。"
order: 1
plots: ["plots/3063-0.json"]
sourceHash: "63ac738f3b348d249ad91071975c124d28b8feca4a97be6416dae44d3cc76198"
sourceCorrections: []
---

尽管传统自然语言处理任务中常用的评估指标，如BLEU、ROUGE、准确率（Accuracy）和F1分数，在评估机器翻译或文本分类等任务时表现良好，但它们常常不足以评估微调 (fine-tuning)大型语言模型（LLM）的综合表现。不加分辨地使用它们可能导致对模型真实能力和不足的误判，尤其是在生成或指令遵循场景中。

### 基于参考的评估指标的不足之处：BLEU和ROUGE

BLEU（Bilingual Evaluation Understudy）和ROUGE（Recall-Oriented Understudy for Gisting Evaluation）等指标主要用于机器翻译和摘要任务。它们通过比较模型生成文本与一个或多个人工编写的参考文本之间的N-gram重叠来运作。


$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$


$$
\text{ROUGE-N} = \frac{\sum_{S \in \text{参考摘要}} \sum_{\text{gram}_n \in S} \text{匹配计数}(\text{gram}_n)}{\sum_{S \in \text{参考摘要}} \sum_{\text{gram}_n \in S} \text{计数}(\text{gram}_n)}
$$


尽管在衡量表面相似度方面有用，但它们对精确N-gram匹配的依赖在评估现代大型语言模型时带来一些问题：

1. **语义等效与词汇重叠：** 大型语言模型可以生成语义上正确且符合语境的输出，但使用与参考文本不同的词汇或表达方式。BLEU和ROUGE会严重惩罚这些有效变体，仅仅因为特定词序不匹配。例如，“The weather forecast predicts rain tomorrow”（天气预报说明天有雨）和“It's expected to rain tomorrow according to the forecast”（根据预报，明天预计有雨）传达相同意义，但N-gram重叠度会很低。
2. **参考范围受限：** 提供涵盖所有可接受变体的全面参考文本通常不切实际或不可能，尤其对于开放式生成任务。这些指标天生会惩罚回应中的创造性或多样性，如果它们偏离了预定义的有限“好”答案集。
3. **对意义扭曲不敏感：** 相反，一个生成内容可能通过重复参考中的关键词获得高重叠分数，但完全扭曲原文意义或未能捕捉到必要细节。
4. **短文本上表现不佳：** 对于涉及非常短的回答或特定格式指令的任务，N-gram重叠成为衡量质量的可靠性较低的指标。一个词语的差异可能显著改变分数，但未能反映质量的相应变化。



![BLEU 分数对措辞的敏感度](plots/3063-0.json)



> 生成1在词汇上相似，并获得不错的分数。生成2在语义上相似但词汇不同，得分较低。生成3不正确，得分接近零。BLEU在评估生成2的有效性时遇到困难。

### 分类评估指标的局限性：准确率、精确率、召回率、F1

准确率、精确率、召回率和F1分数等指标是分类任务的常用方法，模型在这些任务中从预定义集合中预测离散的类别标签。


$$
\text{准确率} = \frac{\text{正确预测数}}{\text{总预测数}}
$$


$$
\text{F1} = 2 \cdot \frac{\text{精确率} \cdot \text{召回率}}{\text{精确率} + \text{召回率}}
$$


试图将生成大型语言模型的评估生硬地套入此框架，暴露出显著的局限：

1. **不适用于开放式生成：** 大型语言模型的大多数微调 (fine-tuning)任务涉及文本生成，而非分类。对于生成的段落、摘要或对话回应，没有单一的“正确”标签。为准确率计算定义“正确性”变得模糊不清，或需要过于简化的替代指标。
2. **忽略输出质量：** 准确率对所有“不正确”的输出一视同仁。一个连贯、相关且仅略有不准确的生成回应，可能与无意义或完全偏离主题的输出受到相同的惩罚。这些指标未能捕捉到重要的特性，如流畅性、连贯性、创造性、风格遵循或事实准确性。
3. **任务过度简化：** 尽管某些大型语言模型应用可以被框定为分类（例如情感分析），但微调通常旨在实现更复杂的行为，例如遵循多步指令、扮演角色或综合信息。简单的分类指标无法衡量这些更复杂场景下的成功。

### 标准指标在微调 (fine-tuning)大型语言模型中的遗漏之处

鉴于这些基于参考和分类指标的局限，它们未能捕捉到微调大型语言模型表现特有的重要方面：

- **指令遵循：** 模型对提示或指令的具体要求遵循得如何？标准指标在这方面提供的信息很少。
- **事实准确性与“幻觉 (hallucination)”：** 生成文本可能流畅且语法正确（如果参考在风格上相似，BLEU/ROUGE得分会很高），但包含事实不准确或编造的信息（“幻觉”）。这些指标对事实性不敏感。
- **安全性、偏见与有害内容：** 模型可以生成与参考内容高度一致或获得高“准确率”的输出，但仍会产生有害、带有偏见或不当的内容。标准指标很少包含安全性维度。
- **连贯性与一致性：** 基于N-gram的指标主要评估局部流畅性。它们难以评估较长生成文本的逻辑流程、信息一致性和整体连贯性。
- **校准性：** 模型表达的置信度（例如，通过概率分数）与其实际正确性匹配得如何？标准指标不衡量这种可靠性。

因此，尽管这些传统指标有时能提供部分信号，尤其是在高度受限的任务中，或者作为更全面评估方案的一个组成部分，但仅依赖它们会提供不完整且可能具误导性的评估。有效评估需要超越这些方法，采用直接衡量指令遵循、事实依据、安全性、稳健性和其他定性方面的方法，正如我们将在后续章节中讨论的。

## 参考资料

- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics; Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍了BLEU分数，一种基于n-gram重叠的机器翻译自动评估基础指标。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out; Publisher: Association for Computational Linguistics; Pages: 74-81; DOI: [10.3115/1621876.1621880](https://doi.org/10.3115/1621876.1621880)
  提出了ROUGE指标套件，通过将生成文本与参考文本进行比较，广泛用于评估摘要和其他文本生成的质量。
