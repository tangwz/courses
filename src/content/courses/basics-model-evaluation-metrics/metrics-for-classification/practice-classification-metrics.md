---
course: "basics-model-evaluation-metrics"
chapter: "metrics-for-classification"
lesson: "practice-classification-metrics"
sourceId: 3985
sourceUrl: "https://apxml.com/zh/courses/basics-model-evaluation-metrics/chapter-2-metrics-for-classification/practice-classification-metrics"
title: "实践：计算分类指标"
description: "通过例子计算准确率、精确率、召回率和F1分数。"
order: 10
plots: ["plots/3985-0.json"]
sourceHash: "fb93d035c52feb4a39e49f3776d97165eca539c07a7b767d1f58d2124ffa204e"
sourceCorrections: []
---

准确率、精确率、召回率和F1分数是评估分类模型的重要指标。计算这些指标对掌握分类模型的表现非常重要。通过一个具体的例子，将展示这些值是如何从正确和错误预测的基本计数中得出的。

### 场景：电子邮件垃圾邮件检测

假设我们构建了一个机器学习 (machine learning)模型，用于将电子邮件分类为“垃圾邮件”（正类别）或“非垃圾邮件”（负类别）。我们用一组模型从未见过的100封电子邮件来测试该模型。运行预测并与实际标签进行比较后，我们得到以下结果：

- **真阳性 (TP)：** 15封邮件被正确识别为垃圾邮件。
- **假阳性 (FP)：** 10封邮件被错误识别为垃圾邮件（它们实际上是“非垃圾邮件”）。
- **真阴性 (TN)：** 70封邮件被正确识别为“非垃圾邮件”。
- **假阴性 (FN)：** 5封邮件被错误识别为“非垃圾邮件”（它们实际上是“垃圾邮件”）。

让我们验证邮件总数：$15 (TP) + 10 (FP) + 70 (TN) + 5 (FN) = 100$ 封邮件。这与我们的测试集大小一致。

### 第1步：构建混淆矩阵

首先，让我们将这些结果整理成之前学过的混淆矩阵格式。请记住，行通常表示实际类别，列表示预测类别。

|  | 预测值：垃圾邮件 | 预测值：非垃圾邮件 | **实际总数** |
| --- | --- | --- | --- |
| **实际值：垃圾邮件** | TP = 15 | FN = 5 | 20 |
| **实际值：非垃圾邮件** | FP = 10 | TN = 70 | 80 |
| **预测总数** | 25 | 75 | 100 |

该矩阵清晰地呈现了模型表现的视觉概览。我们可以看到对角线上正确预测的数量（TP和TN），以及对角线外的错误（FP和FN）。我们还看到实际垃圾邮件总数（20）和非垃圾邮件总数（80），以及模型预测为垃圾邮件（25）和非垃圾邮件（75）的数量。

### 第2步：计算准确率

准确率表示正确预测的总体比例。

公式为：


$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$


代入我们的值：


$$
Accuracy = \frac{15 + 70}{15 + 10 + 70 + 5} = \frac{85}{100} = 0.85
$$


因此，模型的准确率是85%。这意味着它正确分类了100封邮件中的85封。虽然85%听起来不错，但我们知道准确率有时可能会产生误导，尤其当类别不平衡时（这里，“非垃圾邮件”有80个，而“垃圾邮件”有20个）。让我们计算其他指标以获得更全面的情况。

### 第3步：计算精确率

精确率衡量正向预测的准确程度。在模型*预测*为垃圾邮件的所有邮件中，有多少实际上*是*垃圾邮件？

公式为：


$$
Precision = \frac{TP}{TP + FP}
$$


使用混淆矩阵中的值（查看“预测值：垃圾邮件”列）：


$$
Precision = \frac{15}{15 + 10} = \frac{15}{25} = 0.60
$$


精确率为60%。这表明当我们的模型将一封邮件标记 (token)为垃圾邮件时，有60%的几率是正确的。其余40%是假阳性（被错误标记为垃圾邮件的合法邮件）。

### 第4步：计算召回率（敏感度）

召回率（也称敏感度或真阳性率）衡量模型正确识别出多少实际正例。在所有*实际上是*垃圾邮件的邮件中，模型*找到了*多少？

公式为：


$$
Recall = \frac{TP}{TP + FN}
$$


使用混淆矩阵中的值（查看“实际值：垃圾邮件”列）：


$$
Recall = \frac{15}{15 + 5} = \frac{15}{20} = 0.75
$$


召回率为75%。这意味着我们的模型成功识别了测试集中所有实际垃圾邮件的75%。剩余的25%是假阴性（漏过过滤器的垃圾邮件）。

### 第5步：计算F1分数

F1分数提供了一个平衡精确率和召回率的单一指标，它使用了它们的调和平均值。当我们希望衡量同时考虑两种错误（FP和FN）时，这很有用。

公式为：


$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$


使用我们刚刚计算的精确率（0.60）和召回率（0.75）：


$$
F1 = 2 \times \frac{0.60 \times 0.75}{0.60 + 0.75} = 2 \times \frac{0.45}{1.35} = \frac{0.90}{1.35} \approx 0.667
$$


F1分数约为66.7%。这个单一数字综合反映了模型在精确率和召回率方面的表现。

### 结果总结

让我们将这些重要指标可视化：



![垃圾邮件检测模型表现指标](plots/3985-0.json)



> 垃圾邮件检测示例的计算性能指标。

### 解释

通过计算这些指标，我们获得了比只看85%准确率更多的信息：

- **准确率 (85%)：** 整体正确率高。
- **精确率 (60%)：** 当模型判断为“垃圾邮件”时，有10次中有6次是正确的。这意味着10次中有4次，它将合法邮件标记 (token)为垃圾邮件（假阳性）。这可能会让用户感到困扰。
- **召回率 (75%)：** 模型捕获了每4封实际垃圾邮件中的3封。这意味着每4封垃圾邮件中就有1封漏入收件箱（假阴性）。这可能会让用户接触到不必要的垃圾邮件。
- **F1分数 (66.7%)：** 提供了一个平衡的衡量标准，在比较模型时很有用，尤其当精确率和召回率值不同时。

请注意其中的权衡。如果我们调整模型，使其在标记垃圾邮件方面更积极（可能提高召回率），我们也可能增加假阳性，从而降低精确率。反之，使模型更保守以避免标记合法邮件（提高精确率）可能会让更多实际垃圾邮件漏过（降低召回率）。精确率与召回率的相对重要性通常取决于具体的应用。对于垃圾邮件检测，用户可能更能容忍一些垃圾邮件漏过（较低的召回率），而不是重要邮件被标记为垃圾邮件（需要较高的精确率）。

本次实践练习展示了如何通过从基本的TP、FP、TN和FN计数中计算这些标准指标，从而更全面地了解分类模型的行为及其对特定任务的适用性。

## 参考资料

- [Hands-On Machine Learning with Scikit-Learn, Keras, & TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) — Aurélien Géron (2019)
  Publisher: O'Reilly Media; Pages: 84-110
  一本关于机器学习概念和实现的实用指南，其中包含一个关于分类评估指标、混淆矩阵及其计算的章节。
- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) — scikit-learn developers (2023)
  Journal: scikit-learn User Guide
  scikit-learn 的官方文档，提供各种分类指标（包括准确率、精确率、召回率和 F1 分数）的定义和实际使用示例。
- [Introduction to Machine Learning](https://mitpress.mit.edu/books/introduction-machine-learning-fourth-edition) — Ethem Alpaydin (2020)
  Publisher: MIT Press; Pages: 285-309
  本教科书介绍了基本的机器学习算法和评估技术，其中有一章专门讨论分类性能评估。
- [CS229 Lecture Notes: Machine Learning](https://github.com/maxim5/cs229-2018-autumn) — Andrew Ng (2018)
  Journal: Stanford University; Publisher: Stanford University; Pages: 8-10
  来自一门著名机器学习课程的综合讲义，在其学习算法的讨论中涵盖了模型评估和分类指标的各个方面。
