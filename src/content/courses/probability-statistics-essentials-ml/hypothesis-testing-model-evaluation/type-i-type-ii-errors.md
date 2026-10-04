---
course: "probability-statistics-essentials-ml"
chapter: "hypothesis-testing-model-evaluation"
lesson: "type-i-type-ii-errors"
sourceId: 1431
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/type-i-type-ii-errors"
title: "理解第一类和第二类错误"
description: "定义并区分假设检验中的第一类（alpha）和第二类（beta）错误。"
order: 2
plots: []
sourceHash: "dafae9cc74ce8489574308432b068a817bc101cdc1a4a0c4f3edfd2057e78205"
sourceCorrections: []
---

统计检验中的假设包括虚无假设（$H_0$），它代表默认状态或无效应，以及备择假设（$H_1$），它代表您试图寻找证据的状况。当使用样本数据在这两者之间做出判断时，实际上是基于不完整信息（样本而非整个总体）进行判断。自然地，这意味着有时可能会做出错误的决定。统计检验中存在两种具体的错误方式，被称为第一类错误和第二类错误。掌握这些是正确解读检验结果的基础。

### 第一类错误：误报

当您在**虚无假设（$H_0$）实际为真时却拒绝它**，就会发生第一类错误。您可以将其视为“假阳性”或误报。

- **概率：** 犯第一类错误的概率用希腊字母阿尔法 ($\alpha$) 表示。
- **显著性水平：** 这个值，$\alpha$，也被称为检验的**显著性水平**。您作为研究人员或分析师，在进行检验*之前*选择这个值。$\alpha$ 的常见选择是 0.05 (5%) 或 0.01 (1%)。
- **含义：** $\alpha$ 为 0.05 意味着您愿意接受 5% 的机会错误地拒绝真实的虚无假设。如果 $H_0$ 确实为真，并且您多次重复实验，您会期望大约 5% 的时间会仅仅由于随机抽样变异而错误地拒绝它。

**例子：** 假设 $H_0$ 是“新网站设计*不*会提高转化率”，而 $H_1$ 是“新设计*会*提高转化率”。第一类错误意味着在新设计实际上没有改进甚至更糟的情况下，却得出新设计*更好*的结论（拒绝 $H_0$），样本中观察到的差异仅仅是由于偶然性。结果可能是浪费资源去实施一个无效的设计。

### 第二类错误：漏报

当您在**虚无假设（$H_0$）实际为假时却未能拒绝它**（这意味着备择假设 $H_1$ 为真），就会发生第二类错误。这就像“假阴性”或未能发现真实存在的效应。

- **概率：** 犯第二类错误的概率用希腊字母贝塔 ($\beta$) 表示。
- **功效：** *正确*拒绝虚假 $H_0$ 的概率称为检验的**功效**，它等于 $1 - \beta$。功效体现了检验发现真实效应的能力。
- **影响 $\beta$ 的因素：** 与 $\alpha$ 不同，$\beta$ 通常不直接设定。它取决于几个因素，包括所选的 $\alpha$、样本大小 ($n$)、假设值与实际总体值之间的真实差异（效应大小）以及数据的变异性。

**例子：** 沿用相同的网站设计场景（$H_0$：无增长，$H_1$：有增长）。第二类错误意味着在新设计实际上*确实*提高了转化率的情况下，却得出新设计*没有*更好（未能拒绝 $H_0$）的结论。这里的后果是错失了实施有益改进的机会。

### 不可避免的权衡

在进行固定样本大小的假设检验时，$\alpha$ 和 $\beta$ 之间存在固有的权衡。

- 如果您将拒绝 $H_0$ 的标准设置得非常严格（例如，选择一个非常小的 $\alpha$，如 0.001），您会降低犯第一类错误的机会。然而，这使得总体上更难拒绝 $H_0$，从而增加了未能发现真实效应的机会（增加了 $\beta$，降低了功效）。
- 相反，如果您让拒绝 $H_0$ 变得更容易（例如，选择一个更大的 $\alpha$，如 0.10），您会增加检验发现真实效应的功效（降低 $\beta$），但同时也会增加犯第一类错误的风险。

这种关系在下方图表中呈现：

> 假设检验中的决策结果，说明了第一类（$\alpha$）和第二类（$\beta$）错误。

显著性水平 $\alpha$ 的选择通常取决于在特定情境下犯每种错误的相对后果。如果第一类错误的代价很高（例如，批准一个有缺陷的医疗设备），您可能会选择一个非常小的 $\alpha$。如果第二类错误更令人担忧（例如，错失一种可能有效果的治疗），您可能会接受略高的 $\alpha$ 来增加检验的功效（$1 - \beta$）。

在机器学习 (machine learning)中，这直接体现在模型评估和特征选择上：

- **模型比较：** 模型 B 是否*明显*优于模型 A？第一类错误意味着在 B 并不更好时声称它更好。第二类错误意味着在 B 确实存在优越性时未能识别。
- **特征重要性：** 特征 X 对目标变量是否有重要影响？第一类错误意味着将一个无用的特征视为重要。第二类错误意味着忽视一个真正有用的特征。

在接下来的章节中，我们使用 Python 库计算 P 值并进行 t 检验和卡方检验等特定检验之前，理解这个框架十分必要。这些错误量化 (quantization)了基于统计证据做出决策所关联的风险。

## 参考资料

- [Probability & Statistics for Engineers & Scientists](https://www.pearson.com/us/higher-education/program/Walpole-Probability-Statistics-for-Engineers-and-Scientists-10th-Edition/PGM33019.html) — Ronald E. Walpole, Raymond H. Myers, Sharon L. Myers, Keying Ye (2017)
  Publisher: Pearson
  一本经典的概率论与统计学教材，详细解释了假设检验、I型和II型错误及其实际意义。
- [An Introduction to Statistical Learning: with Applications in Python](https://doi.org/10.1007/978-3-031-38747-0) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  理解机器学习统计基础的必备书籍，从应用角度涵盖了假设检验、模型评估以及相关的错误。
- [Hypothesis Testing: Errors, Power, p-values (from 18.05 Introduction to Probability and Statistics)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/readings/mit18_05s14_reading18.pdf) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  麻省理工学院开放课程讲义，清晰易懂地解释了I型和II型错误、显著性水平以及统计功效。
