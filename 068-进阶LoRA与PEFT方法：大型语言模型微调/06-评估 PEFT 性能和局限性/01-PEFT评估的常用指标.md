# PEFT评估的常用指标

来源：[原文](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-evaluation-metrics)

[返回章节目录](README.md) · [返回课程目录](../README.md)

评估使用参数 (parameter)高效微调 (fine-tuning)（PEFT）技术调整过的模型性能，需要选择合适的衡量指标，以反映其在预期下游任务上的成效。仅仅训练一个模型是不够的；客观的衡量方式是必要的，以了解LoRA、QLoRA或适配器微调等方法彼此之间以及与传统完全微调相比的表现如何。衡量指标的选择很大程度上取决于具体的应用，无论是涉及理解语言（NLU）还是生成语言（NLG）。

## 自然语言理解（NLU）任务的衡量指标

NLU任务通常包括分类、序列标注或问答。PEFT方法通常在GLUE（通用语言理解评估）或SuperGLUE等既有基准上进行评估，这些基准涵盖了多种此类任务。

### 分类任务

对于情感分析、主题分类或自然语言推理 (inference)等任务，目标是将标签分配给给定输入文本，可应用标准分类衡量指标：

- **准确率：** 最简单的衡量指标，表示正确预测的比例。尽管直观，但在数据不平衡的数据集上可能会产生误导。
  $\text{准确率} = \frac{\text{正确预测的数量}}{\text{预测总数}}$
- **精确率、召回率和F1分数：** 这些衡量指标提供了更详细的视角，尤其适用于不平衡的类别。
  - **精确率：** 衡量积极预测的准确性。$\text{精确率} = \frac{TP}{TP + FP}$（其中TP = 真阳性，FP = 假阳性）。
  - **召回率（敏感度）：** 衡量实际阳性中被正确识别的比例。$\text{召回率} = \frac{TP}{TP + FN}$（其中FN = 假阴性）。
  - **F1分数：** 精确率和召回率的调和平均值，提供了一个平衡两者的单一分数。$\text{F1} = 2 \times \frac{\text{精确率} \times \text{召回率}}{\text{精确率} + \text{召回率}}$。这通常是基准中分类任务的主要衡量指标。
- **马修斯相关系数（MCC）：** 即使对于不平衡的类别，也被认为是一种平衡的衡量方式，取值范围从-1（完全不一致）到+1（完美预测）。
  $\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$
  （其中TN = 真阴性）。

评估PEFT时，我们使用这些衡量指标来比较例如LoRA适配器所达到的性能与完全微调 (fine-tuning)的基础模型的性能。目标通常是在使用少得多的可训练参数 (parameter)的同时，达到接近完全微调的性能（例如，F1分数相差1-2个百分点内）。

### 问答（QA）任务

对于抽取式问答任务（如SQuAD - 斯坦福问答数据集），答案是给定上下文 (context)中的一段文本，常用衡量指标包括：

- **精确匹配（EM）：** 衡量与真实答案完全匹配的预测百分比。这是一个严格的衡量指标。
- **F1分数：** 在词元 (token)级别计算，将预测和真实值视为词元集合。它衡量预测答案范围与真实答案范围之间的重叠度，为部分正确的答案提供部分得分。这通常被认为比EM更准确的衡量指标。

## 自然语言生成（NLG）任务的衡量指标

评估生成文本（例如，摘要、翻译、对话）本质上比评估NLU任务更为复杂，因为可能存在多个有效输出。衡量指标通常依赖于将生成文本与一个或多个参考文本进行比较。

- **BLEU（双语评估替补）：** 主要用于机器翻译，BLEU衡量生成文本与参考译文之间的n-gram精确率重叠度。它会惩罚过短的句子。分数越高表示与参考文本的相似度越好。
- **ROUGE（召回率导向的摘要评估替补）：** 通常用于摘要任务，ROUGE衡量n-gram召回率重叠度。其变体包括：
  - **ROUGE-N：** 衡量n-gram的重叠度（例如，ROUGE-1用于unigram，ROUGE-2用于bigram）。
  - **ROUGE-L：** 衡量生成摘要与参考摘要之间最长公共子序列（LCS），捕捉句子级别的结构相似性。
- **METEOR（明确排序翻译评估指标）：** 也用于翻译和生成任务，METEOR考虑精确匹配、词干匹配、同义词匹配和释义，根据这些标准对预测和参考进行对齐 (alignment)。它包含对错误词序的惩罚。
- **困惑度（PPL）：** 一种内在评估衡量指标，衡量概率模型预测样本的优劣。困惑度越低，表示模型对测试数据越不感到“惊讶”，暗示着更好的语言建模能力。虽然在训练期间有用，但它并不总是与人类对下游任务质量的判断完全相关。



[交互图表：NLU任务性能（F1分数）](https://apxml.com/zh/courses/lora-peft-efficient-llm-training/chapter-6-evaluating-peft-limitations/peft-evaluation-metrics#plot-14fbhcf)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "NLU任务性能（F1分数）",
    "xaxis": {
      "title": "方法"
    },
    "yaxis": {
      "title": "F1分数",
      "range": [
        0.8,
        0.95
      ]
    },
    "legend": {
      "traceorder": "normal"
    },
    "autosize": true,
    "colorway": [
      "#4263eb",
      "#12b886",
      "#fab005",
      "#f76707",
      "#ae3ec9"
    ]
  },
  "data": [
    {
      "type": "bar",
      "name": "完全微调",
      "x": [
        "情感分析"
      ],
      "y": [
        0.92
      ],
      "marker": {
        "color": "#4263eb"
      }
    },
    {
      "type": "bar",
      "name": "LoRA (r=16)",
      "x": [
        "情感分析"
      ],
      "y": [
        0.915
      ],
      "marker": {
        "color": "#12b886"
      }
    },
    {
      "type": "bar",
      "name": "QLoRA (r=16)",
      "x": [
        "情感分析"
      ],
      "y": [
        0.91
      ],
      "marker": {
        "color": "#fab005"
      }
    },
    {
      "type": "bar",
      "name": "适配器微调",
      "x": [
        "情感分析"
      ],
      "y": [
        0.905
      ],
      "marker": {
        "color": "#f76707"
      }
    },
    {
      "type": "bar",
      "name": "前缀微调",
      "x": [
        "情感分析"
      ],
      "y": [
        0.89
      ],
      "marker": {
        "color": "#ae3ec9"
      }
    }
  ]
}
```

</details>



> 使用F1分数比较不同PEFT方法在NLU任务上与完全微调 (fine-tuning)的表现。PEFT方法通常能接近完全微调的性能。

## PEFT评估的考量

评估PEFT涉及针对特定任务的额外考量：

- **性能与参数 (parameter)数量：** PEFT的一个主要目标是效率。评估时应始终考虑所达到的性能与可训练参数数量的关系。一种方法在标准衡量指标上可能性能略低，但如果效率显著更高，在资源有限的环境中会更受欢迎。
- **对超参数 (hyperparameter)的敏感性：** LoRA等PEFT方法具有特定的超参数（例如，秩$r$，缩放因子$\alpha$）。理想情况下，评估应探讨性能衡量指标对这些设置的敏感程度。
- **任务可迁移性：** 在一个任务上训练的PEFT模块在密切相关的任务上表现如何？评估可迁移性可以提供对所学调整的泛化能力的见解。
- **人工评估：** 对于NLG任务，自动化衡量指标往往未能捕捉到流畅性、连贯性和创造性等方面。人工评估虽然昂贵且耗时，但对于全面评估仍是一个重要组成部分，尤其是在预期存在细微质量差异时。

选择和解读合适的衡量指标对于理解使用PEFT技术固有的权衡非常重要。这有助于做出明智的决定，选择哪种方法和配置最适合特定的任务要求和操作限制。后续章节将进一步探讨基准测试、鲁棒性分析以及评估这些方法相关的计算成本。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR)
  介绍了LoRA，一种常在此章节评估背景下进行比较的参数高效微调方法。
- [SuperGLUE: A Stronger General Language Understanding Evaluation Benchmark](https://arxiv.org/pdf/1905.00537) — Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel Bowman (2019)
  Journal: Advances in Neural Information Processing Systems 32; Publisher: NeurIPS Proceedings; Volume: 32; DOI: [10.48550/arXiv.1905.00537](https://doi.org/10.48550/arXiv.1905.00537)
  介绍了SuperGLUE，一个用于评估自然语言理解模型的广泛使用的基准，涵盖了各种NLU任务和指标。
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040.pdf) — Kishore Papineni, Salim Roukos, Todd Ward, Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting on Association for Computational Linguistics (ACL); Publisher: Association for Computational Linguistics (ACL); Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍BLEU的奠基性论文，BLEU是评估机器翻译文本质量的标准自动指标。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out: Proceedings of the ACL-04 Workshop; Publisher: Association for Computational Linguistics; Pages: 25-34; DOI: [10.3115/1614008.1614022](https://doi.org/10.3115/1614008.1614022)
  介绍了ROUGE，一套广泛用于自动评估摘要和其他文本生成任务的指标。

---

[上一节](../05-%E4%BC%98%E5%8C%96%E3%80%81%E9%83%A8%E7%BD%B2%E4%B8%8E%E5%AE%9E%E9%99%85%E8%80%83%E9%87%8F/08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%E5%A4%9A%E4%B8%AA%20LoRA%20%E9%80%82%E9%85%8D%E5%99%A8%E8%BF%9B%E8%A1%8C%E5%BE%AE%E8%B0%83.md) · [下一节](02-PEFT%20%E4%B8%8E%E5%AE%8C%E5%85%A8%E5%BE%AE%E8%B0%83%E7%9A%84%E5%AF%B9%E6%AF%94%E8%AF%84%E4%BC%B0.md)
