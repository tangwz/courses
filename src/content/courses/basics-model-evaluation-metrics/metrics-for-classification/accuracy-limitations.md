---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "accuracy-limitations"
sourceId: 3965
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/accuracy-limitations"
title: "准确率何时会产生误导"
description: "理解准确率不是衡量好指标的场景，例如不平衡数据集。"
order: 3
plots: ["plots/3965-0.json"]
sourceHash: "0138a5a95e626fe9b0079e484233d1c3fcdc78f80b794fbca11724ac5eeafcd2"
sourceCorrections: []
---

准确率虽然能快速概括整体正确性（即总预测中正确部分的比例），但有时它可能过于乐观地甚至误导性地展现模型表现。在处理**不平衡数据集**时，这种情况时常发生。

### 不平衡问题

不平衡数据集是指其中一类观测数量远少于其他类别的数据集。可以考虑以下任务：

- 发现罕见疾病（多数患者是健康的）。
- 识别欺诈性信用卡交易（多数交易是合法的）。
- 过滤垃圾邮件（多数邮件可能不是垃圾邮件）。

在这些情况下，一个类别（“多数”类别，如健康患者或合法交易）的数量远超另一个类别（“少数”类别，如患病患者或欺诈交易）。

让我们再次考虑我们的欺诈识别例子。假设我们有1,000笔交易数据。

- 990笔交易是**合法**的（多数类别）。
- 10笔交易是**欺诈**的（少数类别）。



![欺诈示例中的类别分布](plots/3965-0.json)



> 绝大多数交易是合法的，而欺诈性交易很少见。

现在，假设我们构建一个非常简单的（也许是朴素的）分类模型。这个模型不是很复杂；它只是简单地预测**每笔交易都是合法**的。它从不标记 (token)任何交易为欺诈。

这个模型的准确率是多少？让我们计算：

- 对于990笔合法交易，模型正确预测为“合法”。即990个正确预测。
- 对于10笔欺诈交易，模型错误预测为“合法”。即10个错误预测。

正确预测的总数是990。预测总数是1000。

准确率是：


$$
\text{准确率} = \frac{\text{正确预测数量}}{\text{总预测数量}} = \frac{990}{1000} = 0.99
$$


99%的准确率！这听起来很棒，对吗？

### 为什么99%的准确率在此处具有误导性

问题在于：尽管模型达到了99%的准确率，但它完全未能完成其预期任务，即**识别欺诈**。它正确识别了所有合法交易，但却遗漏了*每一笔*欺诈交易。一个未能发现任何欺诈的模型，即使其整体准确率得分很高，也几乎是无用的。

出现这种情况，是因为多数类示例的庞大数量主导了准确率的计算。模型仅通过正确识别最常见的结果就能获得高分。在少量少数类示例上产生的错误几乎不影响总体百分比。

“在许多应用中，尤其涉及不平衡数据时，正确识别少数类别通常是最重要的目标。漏掉一笔欺诈交易可能比错误分类一笔合法交易的代价要大得多。同样，未能发现罕见疾病可能导致严重后果。”

因此，在这种情况下仅依赖准确率可能导致部署的模型在我们最关心的任务上表现不佳。这强调了需要其他评估指标，这些指标能让我们更清楚地了解模型在*每个*类别，特别是少数类别上的表现。接下来我们将讨论的，从混淆矩阵中得出的指标，如精确率和召回率，有助于提供这种更全面的认识。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本书提供了机器学习的实用入门指南，其中一个章节专门讨论分类指标，清晰地解释了准确度在不平衡数据集中的局限性，并介绍了替代的评估指标。
- [Data Mining: Concepts and Techniques](https://www.elsevier.com/books/data-mining-concepts-and-techniques/han/978-0-12-381479-1) — Jiawei Han, Micheline Kamber, and Jian Pei (2011)
  Publisher: Morgan Kaufmann
  这是一本关于数据挖掘的综合性教材，其中详细讨论了分类模型的评估，强调了为什么准确度在倾斜的类别分布中可能不足，以及其他指标的重要性。
