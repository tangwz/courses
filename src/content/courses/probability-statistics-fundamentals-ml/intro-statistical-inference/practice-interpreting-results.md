---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-statistical-inference"
lesson: "practice-interpreting-results"
sourceId: 2473
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference/practice-interpreting-results"
title: "练习：解释统计结果"
description: "练习解释简单统计分析的结果，包括点估计和p值。"
order: 8
plots: ["plots/2473-0.json"]
sourceHash: "4aada1195210771e453b3b36b325438d75dfc49a1254688b63ab8212eef0c153"
sourceCorrections: []
---

统计推断帮助我们从观察样本数据转向对更大的总体进行有根据的陈述。重点不仅仅是得到一个数字（比如平均值或p值），而是理解这个数字告诉了我们什么（以及没有告诉我们什么）。

在本部分，我们将考察一些你可能遇到统计结果的常见情形，并练习正确地解释它们。请记住，目标是在特定情境中理解点估计、置信区间和p值的含义。

### 情形1：估计网站转化率

假设你正在一个电子商务网站工作。你进行了一项A/B测试，其中1000名访问者看到一个新的结算页面设计（B组），而另外1000名访问者看到旧设计（A组）。你想估计新设计的转化率（进行购买的访问者百分比）。

测试后，你发现看到新设计的1000名访问者中有55人进行了购买。你的分析结果如下：

- **转化率的点估计（B组）：** 0.055（或5.5%）
- **转化率的95%置信区间（B组）：** [0.042, 0.068]（或4.2%到6.8%）

**解释：**

1. **点估计：** 值0.055是我们对所有访问者都看到新设计时，*整个总体*的真实转化率的单一最佳估计。这个估计是直接基于我们的样本数据（55次转化/1000名访问者）得出的。
2. **置信区间：** 区间[0.042, 0.068]为新设计的真实转化率提供了一个合理值范围。我们“95%置信”所有潜在访问者的真实转化率介于4.2%和6.8%之间。
   - “95%置信”意味着什么？它指的是用于创建该区间的*方法*。如果我们将此实验重复进行多次，每次都构建一个95%置信区间，我们预计大约95%的这些区间会成功包含真实的总体转化率。这不意味着真实值*有*95%的概率落在此特定区间内；相反，它反映了我们对生成该区间所用过程的信心。
   - 这个区间让我们了解点估计周围的不确定性程度。更宽的区间会表示更多不确定性（可能是由于样本量较小或变异性较大），而更窄的区间则表示更精确的估计。

### 情形2：测试功能变更的影响

假设你的团队在一个流媒体服务上部署了一个新的推荐算法。你想知道与旧算法相比，这项变更是否显著增加了每周每用户观看视频的平均数量。

你在一个月内从两组用户那里收集数据，并进行了一项假设检验。

- **零假设 ($H_0$)：** 新算法*没有*增加每周每用户观看视频的平均数量。（数学上通常写为 $\mu_{new} \le \mu_{old}$ 或 $\mu_{new} - \mu_{old} \le 0$，其中 $\mu$ 代表总体平均值。）
- **备择假设 ($H_1$)：** 新算法*确实*增加了每周每用户观看视频的平均数量。（$\mu_{new} > \mu_{old}$ 或 $\mu_{new} - \mu_{old} > 0$）。
- **显著性水平 ($\alpha$)：** 你设定了 $\alpha = 0.05$ 的阈值。这意味着你愿意接受5%的几率，在实际并非如此的情况下，错误地得出新算法更好的结论（第一类错误）。

统计软件基于样本数据输出以下结果：

- **样本平均视频观看量（旧算法）：** 8.2个视频/周
- **样本平均视频观看量（新算法）：** 8.9个视频/周
- **P值：** 0.028

这是样本平均值的简单可视化：



![每周观看视频的样本平均值](plots/2473-0.json)



> 样本平均值显示新算法有更高的参与度，但p值告诉我们这种差异是否具有统计学意义。

**解释：**

1. **比较p值与$\alpha$：** 计算出的p值（0.028）小于我们选定的显著性水平（$\alpha = 0.05$）。
2. **决策：** 因为 $p < \alpha$，我们**拒绝零假设 ($H_0$)**。
3. **结论：** 我们得出结论，有统计学意义的证据表明，与旧算法相比，新的推荐算法*增加了*每周每用户观看视频的平均数量。如果新算法真的不比旧算法好，观察到的差异（样本中的8.9个视频对比8.2个视频）不太可能仅仅是由于随机机会造成的。

**如果p值为0.15会怎样？**

如果p值是0.15（大于$\alpha = 0.05$），我们的解释会改变：

1. **比较p值与$\alpha$：** $0.15 > 0.05$。
2. **决策：** 我们**未能拒绝零假设 ($H_0$)**。
3. **结论：** 我们得出结论，没有足够的统计学意义证据表明新算法增加了平均视频观看量。尽管样本平均值更高（8.9对比8.2），p值为0.15表明，即使真实平均值相同（或者新算法甚至更差），这样的差异也可能由于随机抽样变异合理地发生。我们没有证明这两种算法*同样*有效，只是我们缺乏强有力的证据来声称新算法*更好*。

### 情形3：比较机器学习 (machine learning)模型表现

You've trained two different classification models (Model A and Model B) to predict customer churn. You test both models on the same hold-out test dataset and obtain their accuracy scores. You want to know if the difference in accuracy is statistically significant.

你训练了两个不同的分类模型（模型A和模型B）来预测客户流失。你在相同的留出测试数据集上测试这两个模型，并获得它们的准确率分数。你想知道准确率的差异是否具有统计学意义。

- **零假设 ($H_0$)：** 两个模型在底层数据分布上具有相同的真实准确率。（准确率$_A$ = 准确率$_B$）
- **备择假设 ($H_1$)：** 模型具有不同的真实准确率。（准确率$_A \ne$ 准确率$_B$）
- **显著性水平 ($\alpha$)：** 你选择 $\alpha = 0.05$。

你使用了一个适当的统计检验（例如麦克尼玛检验，它适用于在同一数据集上比较分类器），并得到：

- **模型A准确率（在测试集上）：** 88%
- **模型B准确率（在测试集上）：** 90%
- **P值：** 0.21

**解释：**

1. **比较p值与$\alpha$：** p值（0.21）大于显著性水平（$\alpha = 0.05$）。
2. **决策：** 我们**未能拒绝零假设 ($H_0$)**。
3. **结论：** 尽管模型B在*这个特定的测试集上*达到了比模型A更高的准确率（90%对比88%），但这种差异在0.05水平上*不*具有统计学意义。没有足够强有力的证据来得出结论，认为模型B会在来自相同分布的新的、未见过的数据上持续表现优于模型A。观察到的2%的差异很可能是由于测试集中包含的特定数据点（随机机会）造成的。在实践中，如果模型B有其他优势（例如更快或更简单），你可能仍然选择它，但你不会仅仅根据这个结果就声称更高的准确率。

这些例子说明了如何解释点估计、置信区间和p值等统计输出，使我们能够从数据中得出更谨慎和知情的结论，这在机器学习和数据分析中做出决策或评估模型时是必不可少的。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R (Second Edition)](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer
  涵盖统计推断、点估计、置信区间和假设检验，并结合机器学习应用。
- [Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing](https://www.cambridge.org/core/books/trustworthy-online-controlled-experiments/9781108724265) — Ron Kohavi, Diane Tang, Ya Xu (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108653985](https://doi.org/10.1017/9781108653985)
  一份关于设计、运行和分析A/B测试的实用指南，包含结果解读和潜在问题。
- [The ASA Statement on P-Values: Context, Process, and Purpose](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108) — Ronald L. Wasserstein, Nicole A. Lazar (2016)
  Journal: The American Statistician; Publisher: Taylor & Francis; Volume: 70; Pages: 129-133; DOI: [10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108)
  提供了美国统计协会关于统计推断中p值正确解读及常见误解的权威指导。
- [All of Statistics: A Concise Course in Statistical Inference](https://www.springer.com/book/978-0-387-40272-7) — Larry Wasserman (2004)
  Publisher: Springer; DOI: [10.1007/978-0-387-21764-2](https://doi.org/10.1007/978-0-387-21764-2)
  对统计推断进行了严谨而易懂的讨论，涵盖估计、置信区间和假设检验。
