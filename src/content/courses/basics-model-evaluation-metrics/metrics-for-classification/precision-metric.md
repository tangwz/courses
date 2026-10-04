---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "precision-metric"
sourceId: 3974
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/precision-metric"
title: "精确率：衡量预测的准确程度"
description: "定义精确率，理解其衡量内容，并学习如何计算。"
order: 6
plots: []
sourceHash: "90a5292d7e550cf4557730a58ce0baaf47241635c080a26c5482163ba160956f"
sourceCorrections: []
---

准确率能让我们对分类模型正确判断的频率有一个总体概念，但在处理类别分布不均或不同类型错误带来的影响差异很大时，它并不能反映全貌。我们需要能提供更具体信息的指标。其中一个指标是**精确率**。

想象一下你构建了一个邮件垃圾邮件过滤器。准确率会告诉你被正确分类的邮件（垃圾邮件被判为垃圾邮件，非垃圾邮件被判为非垃圾邮件）的总百分比。但你可能特别关心这个问题：*在所有被过滤器归入垃圾邮件文件夹的邮件中，有多少是真正的垃圾邮件？* 你肯定不希望你的过滤器将重要邮件错误地标记 (token)为垃圾邮件。这种对*阳性预测*正确性的关注，正是精确率所衡量的。

### 精确率衡量什么

精确率回答了这个问题：**在所有被模型预测为阳性的样本中，实际为阳性的比例是多少？**

它侧重于模型对阳性类别的预测。你可以将其视为衡量预测准确度或质量的指标。高精确率分数意味着当你的模型预测某个样本属于阳性类别时，这个预测很有可能是正确的。

### 精确率公式

为了计算精确率，我们使用混淆矩阵中的值，具体来说是真阳性 ($TP$) 和假阳性 ($FP$):

- **真阳性 ($TP$)**: 被正确预测为阳性的阳性样本数量。
- **假阳性 ($FP$)**: 被错误预测为阳性的阴性样本数量（也称为“I类错误”）。

精确率的公式是：


$$
精确率 = \frac{TP}{TP + FP}
$$


请注意分母 ($TP + FP$) 代表了你的模型*预测*为阳性的样本总数。精确率是正确预测为阳性的样本数 ($TP$) 与被预测为阳性的总数之间的比率。

> 用于计算精确率的组成部分。它仅侧重于模型分类为阳性的样本 ($TP + FP$ )。

### 示例：垃圾邮件过滤器计算

让我们回到我们的垃圾邮件过滤器示例。假设对1000封邮件进行过滤器测试后，我们得到以下混淆矩阵：

|  | 预测：垃圾邮件 | 预测：非垃圾邮件 | 实际总数 |
| --- | --- | --- | --- |
| **实际：垃圾邮件** | TP = 95 | FN = 5 | 100 |
| **实际：非垃圾邮件** | FP = 10 | TN = 890 | 900 |
| **预测总数** | 105 | 895 | 1000 |

为了计算精确率，我们需要 $TP$ 和 $FP$:

- $TP = 95$ (过滤器正确识别了95封垃圾邮件)
- $FP = 10$ (过滤器错误地将10封非垃圾邮件标记 (token)为垃圾邮件)

现在，应用公式：


$$
精确率 = \frac{TP}{TP + FP} = \frac{95}{95 + 10} = \frac{95}{105} \approx 0.905
$$


因此，我们垃圾邮件过滤器的精确率约为0.905，即90.5%。这意味着当过滤器将一封邮件标记为垃圾邮件时，它有大约90.5%的可能是正确的。

### 高精确率何时重要？

当假阳性 ($FP$) 的代价很高时，高精确率就显得特别有用。请看以下情况：

1. **垃圾邮件过滤：** 如前所述，您希望避免将正常邮件标记 (token)为垃圾邮件 ($FP$)。如果重要工作邮件或个人消息被错误地分类为垃圾邮件，可能会带来麻烦。高精确率能够确保被标记为垃圾邮件的邮件确实是垃圾邮件。
2. **医疗诊断（确认严重病情）：** 如果阳性预测会导致昂贵、侵入性或高风险的治疗，您会希望非常确定预测是正确的。假阳性（将健康患者诊断为患有该病）可能会导致不必要的伤害和费用。
3. **搜索引擎结果：** 当您搜索特定内容时，会希望顶部结果高度相关。不相关的结果出现在顶部（在相关性方面属于假阳性）会导致糟糕的用户体验。
4. **欺诈检测（警报）：** 尽管发现欺诈行为很重要，但生成过多虚假警报 ($FP$) 可能会使调查人员不堪重负，并惹恼那些合法交易被标记的客户。

在这些情况下，我们希望最小化假阳性，这意味着最大化精确率。

### 精确率并非全部

精确率为我们提供了关于阳性预测可靠性的有价值信息，但它没有考虑假阴性 ($FN$)，即模型错误地分类为阴性的阳性样本。在我们的垃圾邮件示例中，$FN = 5$，这意味着有5封真正的垃圾邮件漏过过滤器进入了收件箱。如果最小化这些遗漏的阳性样本很重要，我们需要查看另一个指标：召回率。我们将在下一节中讨论召回率。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38936-6](https://doi.org/10.1007/978-3-031-38936-6)
  一本关于统计学习的综合性入门教材，涵盖了精确率等分类指标。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本关于统计学习的权威教材，对模型评估指标进行了深入的阐述。
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) — scikit-learn developers (2024)
  scikit-learn分类指标的官方文档，详细介绍了实际实现和公式。
- [CS229 Machine Learning Course Materials](http://cs229.stanford.edu/materials.html) — Andrew Ng (2018)
  Publisher: Stanford University
  斯坦福大学机器学习课程的教学材料，为评估指标提供了学术基础。
