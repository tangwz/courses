# 定量评估：ROUGE、BLEU 和困惑度

来源：[原文](https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment/quantitative-evaluation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管人工判断是模型质量的最终评判标准，但自动化指标提供了一种可扩展且客观的方式来追踪开发过程中的进展。它们让你能够快速比较不同的模型检查点或微调 (fine-tuning)方法，而无需为每一次修改付出手动审查的时间和费用。为此，三种最常用的定量指标是困惑度、ROUGE 和 BLEU。每种指标都提供了一个衡量模型表现的不同角度，它们共同构成了你评估流程的坚实基础。

### 困惑度：衡量预测准确性

困惑度是一种内在指标，它衡量概率模型预测样本的能力。在语言模型的场景下，它量化 (quantization)了模型在遇到测试集中的文本序列时的“惊讶程度”。困惑度分数越低，表明模型越不“惊讶”，意味着它对实际的词语序列赋予了更高的概率。这表明模型的内部概率分布与其测试数据很好地吻合。

困惑度直接来源于模型验证期间计算的交叉熵损失。如果 $H(p,q)$ 是每个词元（token）的平均交叉熵损失，困惑度计算公式如下：


$$
\text{困惑度} = e^{\text{交叉熵}} = 2^{H(p,q)}
$$


例如，给定句子前缀“工程师调试了……”，一个困惑度较低的模型会给“代码”这类合理的续接词赋予高概率。而一个给“天空”这类不太可能出现的词赋予高概率的模型，其困惑度会更高。

**何时使用困惑度：**

- **模型比较：** 它能有效地比较在相同留出数据集上不同模型或超参数 (parameter) (hyperparameter)的表现。困惑度越低通常表示模型拟合得越好。
- **训练监控：** 在训练期间追踪验证集上的困惑度是检测过拟合 (overfitting)的常用方法。如果验证困惑度开始增加，而训练损失持续下降，这表明模型泛化能力正在减弱。

**局限性：**

困惑度衡量模型预测测试数据*语言模式*的能力。它不衡量生成回复的事实准确性、连贯性或整体质量。一个模型可能在新闻文章数据集上获得低困惑度分数，但仍然生成事实不准确的信息。因此，它是衡量模型拟合度的有用指标，但单独使用不足以进行全面评估。

### 基于N-gram的指标：BLEU 和 ROUGE

与着眼于模型内部置信度的困惑度不同，BLEU 和 ROUGE 等指标通过将生成的最终文本与一个或多个人工编写的参考文本进行比较来评估。它们都基于N-gram重叠的原理运作。N-gram是文本中*n*个连续项目（在本例中为单词）的序列。

- **Unigram (1-gram)：** “The”，“cat”，“sat”
- **Bigram (2-gram)：** “The cat”，“cat sat”
- **Trigram (3-gram)：** “The cat sat”

通过比较生成文本与参考文本中的N-gram，这些指标提供了一个能近似反映输出质量的分数。

### ROUGE：摘要评估的召回导向替补

ROUGE（摘要评估的召回导替补）是一组主要用于评估文本摘要等任务的指标。它的核心思想是衡量召回率：即人工编写的参考摘要中有多少N-gram被“召回”或在模型生成的摘要中找到。

最常见的变体包括：

- **ROUGE-N：** 衡量N-gram的重叠度。`ROUGE-1` 衡量unigram重叠度，反映内容重叠，而 `ROUGE-2` 衡量bigram重叠度，能更好地指示流畅度。
- **ROUGE-L：** 衡量最长公共子序列（LCS）。LCS找到生成文本和参考文本之间共享的最长词语序列，但它们不必是连续的块。这使得ROUGE-L善于捕捉句子层面的结构相似性。

较高的ROUGE分数意味着生成文本与参考文本共享大量内容。

> ROUGE-1 unigram在参考文本和模型输出之间的重叠情况。绿色节点表示共享词语，有助于提高召回分数。

Hugging Face的`evaluate`库使得ROUGE的计算变得简单直观。

```python
from evaluate import load

rouge = load('rouge')
predictions = ["the cat was on the mat"]
references = ["the cat sat on the mat"]

results = rouge.compute(predictions=predictions, references=references)
print(results)
# {'rouge1': 0.857..., 'rouge2': 0.666..., 'rougeL': 0.857...}
```

### BLEU：双语评估替补

BLEU（双语评估替补）最初是为了评估机器翻译而创建的。与ROUGE不同，它是一种侧重于准确率的指标。它衡量*生成*文本中有多少N-gram出现在*参考*文本中。

BLEU对标准准确率计算包含两项主要调整：

1. **修改N-gram准确率：** 它将每个候选N-gram的计数限制为其在任何单个参考中出现的最大次数。这可以防止模型通过过度生成某个合理词语（例如，输出“the the the the”）而获得虚高分数。
2. **长度惩罚 (BP)：** 它惩罚比参考文本短的生成文本。非常短的输出可能具有完美的准确率，但却是不充分的翻译或摘要。BP确保模型生成合理长度的输出。

较高的BLEU分数表明生成的文本流畅（N-gram匹配）且长度恰当。它常用于流畅度要求较高的任务，例如翻译和代码生成。

```python
from evaluate import load

bleu = load('bleu')
predictions = ["the cat was on the mat"]
references = [["the cat sat on the mat"]] # 注意：references 是一个列表的列表

results = bleu.compute(predictions=predictions, references=references)
print(results)
# {'bleu': 0.759..., 'precisions': [0.833..., 0.8, 0.6, 0.5], 'brevity_penalty': 1.0, ...}
```

### ROUGE 和 BLEU 的选择

ROUGE 和 BLEU 之间的选择取决于你想要衡量什么。ROUGE 旨在捕捉内容（召回率），而 BLEU 则关注流畅度和忠实度（准确率）。

- 对于**摘要**任务，目标是捕捉原文中的所有要点，通常更倾向于使用**ROUGE**。
- 对于**机器翻译**，输出必须语法正确并准确传达原文含义，**BLEU**是标准方法。

通常，同时报告两者是有益的。一个模型可能通过生成冗长、散漫但包含所有正确关键词的摘要而获得高ROUGE分数，但这可能可读性不佳（BLEU低）。另一个模型可能生成流畅、简洁的句子（BLEU高），但遗漏了部分信息（ROUGE低）。



[交互图表：模型在摘要任务上的表现](https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-5-evaluation-and-deployment/quantitative-evaluation#plot-14lywd4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "模型在摘要任务上的表现"
    },
    "xaxis": {
      "title": "指标"
    },
    "yaxis": {
      "title": "分数",
      "range": [
        0,
        0.8
      ]
    },
    "barmode": "group",
    "legend": {
      "title": {
        "text": "模型"
      }
    },
    "font": {
      "family": "sans-serif"
    },
    "plot_bgcolor": "#e9ecef",
    "paper_bgcolor": "#e9ecef"
  },
  "data": [
    {
      "type": "bar",
      "name": "模型A (侧重召回)",
      "x": [
        "ROUGE-L",
        "BLEU"
      ],
      "y": [
        0.65,
        0.4
      ],
      "marker": {
        "color": "#1c7ed6"
      }
    },
    {
      "type": "bar",
      "name": "模型B (侧重准确率)",
      "x": [
        "ROUGE-L",
        "BLEU"
      ],
      "y": [
        0.45,
        0.6
      ],
      "marker": {
        "color": "#12b886"
      }
    }
  ]
}
```

</details>



> 模型A在ROUGE-L上表现更好，这表明它捕捉了更多的参考内容。模型B在BLEU上表现更强，这说明其输出更流畅和准确，尽管可能遗漏了部分内容。

归根结底，这些定量指标是自动化分析的强大工具。它们提供关于模型表现的快速、可复现和可扩展的信号。然而，它们是人类感知质量的替代指标。高分是一个好迹象，但这不保证模型一定优秀。务必将这些分数与我们接下来讨论的定性、人工参与的评估相结合，以便全面了解模型的能力。

## 参考资料

- [BLEU: A Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu (2002)
  Journal: Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL); Publisher: Association for Computational Linguistics; Pages: 311-318; DOI: [10.3115/1073083.1073135](https://doi.org/10.3115/1073083.1073135)
  介绍BLEU指标的原始研究论文，阐述了其基于修改后的n-gram精度和简短性惩罚的计算方法，用于机器翻译评估。
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — Chin-Yew Lin (2004)
  Journal: Text Summarization Branches Out: Proceedings of the ACL-04 Workshop; Publisher: Association for Computational Linguistics; Pages: 74-81; DOI: [10.3115/1621378.1621389](https://doi.org/10.3115/1621378.1621389)
  介绍ROUGE系列指标的原始出版物，解释了其如何通过衡量面向召回的n-gram重叠来自动评估摘要。
- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  一本关于自然语言处理的综合教材，包括对语言模型、困惑度的定义和计算，以及BLEU和ROUGE等多种评估指标的深入讨论。

---

[上一节](01-%E7%95%8C%E5%AE%9A%E7%94%9F%E6%88%90%E5%9E%8B%E4%BB%BB%E5%8A%A1%E7%9A%84%E6%80%A7%E8%83%BD%E6%8C%87%E6%A0%87.md) · [下一节](03-%E5%AE%9A%E6%80%A7%E8%AF%84%E4%BC%B0%EF%BC%9A%E4%BA%BA%E5%B7%A5%E5%8F%82%E4%B8%8E%E8%AF%84%E4%BC%B0.md)
