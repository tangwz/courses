---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "understanding-classification-predictions"
sourceId: 3959
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/understanding-classification-predictions"
title: "理解分类预测"
description: "回顾分类模型的输出以及预测是如何产生的。"
order: 1
plots: []
sourceHash: "5d251986063a17e61b516dd71e095bbd9b56bcc58b8d85a4a183f13b91c2a0cc"
sourceCorrections: []
---

在我们探讨准确率或精确率等具体指标之前，我们先弄清楚分类模型会给出什么样的输出，以及我们如何得到最终的预测结果。回顾第一章，分类任务是将输入数据分到预设的类别中。例如，识别一封邮件是“垃圾邮件”还是“非垃圾邮件”，或者将图片归类为“猫”、“狗”或“鸟”。

### 从概率到预测

大多数分类算法不会直接给出硬性的类别标签。相反，它们通常会为每个可能的类别生成一个分数或概率。这个概率表示模型对输入属于该特定类别的置信程度。

以垃圾邮件检测这样一个简单的二分类问题为例。对于一封给定的邮件，模型可能会输出如下结果：

- “垃圾邮件”的概率：0.85
- “非垃圾邮件”的概率：0.15

请注意，这些概率的总和为 1.0 ($0.85 + 0.15 = 1.0$)。这是许多分类模型的常见表现。对于一个多类别问题（例如，手写数字0到9的分类），模型会输出十个概率，每个数字对应一个，这些概率的总和也为1.0。

### 判别阈值

我们如何从这些概率（例如“垃圾邮件”的0.85）得到明确的预测（该邮件*是*垃圾邮件）？我们需要使用一个**判别阈值**。

最常用的默认阈值是0.5。规则很简单：

- 如果正类别（例如“垃圾邮件”）的概率大于该阈值（0.5），则预测为该类别。
- 否则，预测为负类别（例如“非垃圾邮件”）。

分类预测通常通过将概率分数与预设阈值进行比较来确定。例如，如果垃圾邮件的概率是0.85，阈值是0.5，那么因为0.85大于0.5，最终预测是“垃圾邮件”。

如果模型输出P(垃圾邮件) = 0.30（因此P(非垃圾邮件) = 0.70），那么预测将是“非垃圾邮件”，因为 $0.30 \le 0.5$。

尽管0.5是一个标准的起始点，但这个阈值并非一成不变。根据具体目标以及不同类型错误带来的影响（我们很快就会在精确率和召回率部分进行讨论），你可能选择调整这个阈值。例如，如果将一封非垃圾邮件错误地判为垃圾邮件会带来很大问题，你可能需要提高阈值（例如到0.9），以便在判定为“垃圾邮件”前获得更高确信度。

### 真实标签与预测

评估的核心思想是将模型的最终预测与实际的、已知的标签进行比较，这些标签通常称作**真实标签**。

- **真实标签：** 输入数据的正确标签（例如，我们*知道*一封特定的邮件确实是“垃圾邮件”）。这来自于用于评估的已标注数据集。
- **预测：** 模型在应用判别阈值后分配的标签（例如，模型*预测*该邮件是“垃圾邮件”）。

接下来我们要探讨的指标，从准确率开始，都是通过在测试集中的许多数据点上系统地比较这些预测结果与真实标签来计算的。理解分类包含通过阈值将概率转换为离散标签这一步骤，对于正确解读这些指标是必不可少的。现在，我们来看衡量性能的最简单方法：准确率。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  这本广泛使用的教材以易懂的方式介绍了分类，解释了模型如何输出概率以及如何应用决策阈值进行预测。
- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 738
  这本经典而全面的教材严谨地阐述了概率分类模型以及将分数转换为离散类别标签的原理。
- [CS229 Machine Learning Course Notes](http://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Andrew Ng (2022)
  Publisher: Stanford University
  这些广泛引用的机器学习课程讲义清晰地解释了分类概念、模型置信度（概率）以及决策边界在最终预测中的作用。
