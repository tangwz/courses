---
course: "basics-model-evaluation-metrics"
chapter: "preparing-data-evaluation"
lesson: "evaluate-on-unseen-data"
sourceId: 4021
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-4-preparing-data-evaluation/evaluate-on-unseen-data"
title: "为何在新数据上评估模型？"
description: "解释为何在模型训练所用的相同数据上测试模型会产生误导性结果。"
order: 1
plots: ["plots/4021-0.json"]
sourceHash: "d922d720f0b2c790d5f2b742a96aba6eb3ada8235044caf0acde597404f79a44"
sourceCorrections: []
---

设想你正在准备一场重要考试。你手头有一套老师给的练习题。一种学习方法是，你只记住*这些特定*问题的答案。如果期末考试用的是*一模一样的问题*，你很可能会得满分！但如果考试出现*新*问题，考查的是相同的知识点呢？如果你只是死记硬背，你可能会很吃力。你并没有真正掌握这些内容；你只是记住了练习题集。

机器学习 (machine learning)模型也面临类似情况。当我们训练一个模型时，我们会给它看一个数据集（即“练习题”），模型会从这些数据中学习规律。如果接着用模型训练过的*相同数据*来评估它，我们本质上是在问它已经记住答案的问题。模型看起来可能会表现得异常出色，给出令人满意的准确率或低错误率。

然而，这种表现常常是一种假象。模型可能对训练数据学习得*过于*具体，包括其噪声和特殊性，而不是我们真正关注的底层普遍规律。这种现象被称为**过拟合 (overfitting)**。

### 理解过拟合 (overfitting)

过拟合是指模型对训练数据学习得非常好，以至于它捕捉到该数据特有的随机波动或噪声，而不是输入与输出之间真实的底层关系。你可以把它想象成模型将数据点拟合得过于紧密。

- \*\*在训练数据上：\*\*一个过拟合模型通常具有非常高的准确率或非常低的错误率，因为它基本上记住了它看过的例子。
- **在新数据上：**当遇到新的、从未见过的数据时，过拟合模型表现会很差。它无法**泛化**，因为从训练集中记住的特定噪声和规律不适用于新数据。

请看一个简单的视觉比喻：



![模型拟合比较](plots/4021-0.json)



> 蓝点代表我们的数据。一个良好模型（绿线）捕捉了普遍趋势。一个欠拟合 (underfitting)模型（黄虚线）过于简单，错过了趋势。一个过拟合模型（红虚点线）波动过大以至于精确地穿过每个训练点，但它可能无法很好地预测新点。

### 目标：评估泛化能力

衡量机器学习 (machine learning)模型成功的真正标准，不是它在已见过的数据上表现如何，而是它在**新的、未见过的数据**上表现如何。这种在训练时未使用过的数据上表现良好的能力被称为**泛化能力**。

为何泛化能力如此重要？因为构建大多数模型的目的，是在现实中应用它们，基于模型创建时还无法获得的数据进行预测或做出决策。

- 垃圾邮件过滤器需要正确分类它以前从未见过的*新*邮件。
- 医疗诊断模型需要准确地用于*新*患者。
- 股价预测器需要预测*未来*价格。

在训练数据上评估模型，几乎不能告诉你其泛化能力。它衡量的是记忆能力，而不是对未来数据的预测能力。

### 为何在训练数据上测试会产生误导

如果你训练好模型并立即在相同的训练数据上测试它，你通常会得到非常乐观的结果。准确率分数可能接近100%，或者错误率（如回归中的MSE）可能接近于零。这会给人一种虚假的自信。模型可能严重过拟合 (overfitting)，而当模型实际投入使用时，其表现可能会急剧下降。

"因此，为了获得对其在实际情况中表现的真实评估，你*必须*在模型训练过程中从未接触过的数据上评估它。这些未见过的数据作为模型未来将遇到的数据的替代品。这是我们需要分割数据的根本原因，我们接下来会讲解这一点。"

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本广泛使用的教科书，系统地解释了过拟合、泛化和模型选择等概念。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本综合性教科书，从概率角度审视机器学习，涵盖了模型复杂度、泛化以及避免过拟合的方法。
- [Machine Learning Lecture Notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf) — Andrew Ng (2011)
  知名机器学习课程的讲义，清晰解释了偏差、方差、过拟合以及使用未见过数据评估模型的必要性。
