---
course: "probability-statistics-essentials-ml"
chapter: "inferential-statistics-sampling-estimation"
lesson: "confidence-intervals-explained"
sourceId: 1427
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-4-inferential-statistics-sampling-estimation/confidence-intervals-explained"
title: "置信区间说明"
description: "了解参数估计中置信区间的定义与解读。"
order: 5
plots: ["plots/1427-0.json"]
sourceHash: "d54d606583adb60a957e9e207d769c6f2edc8581e0746ce580257e1da74b6de4"
sourceCorrections: []
---

点估计，例如样本均值$\bar{x}$，为总体参数 (parameter)（如总体均值$\mu$）提供一个单一的最佳估计值。然而，这个单一值未能传达估计值周围的不确定性。例如，我们的$\bar{x}$与真实$\mu$有多接近？这是一个非常精确的估计，还是真实值可能存在显著差异？置信区间通过量化 (quantization)参数的合理值范围来解决这个问题。

### 什么是置信区间？

置信区间（CI）为未知总体参数 (parameter)提供了一个*范围*的可能值，这些值根据我们的样本数据计算得出。它不是一个单一数字，而是给出了一个下限和上限。例如，我们可能不说用户的估计平均身高是170厘米（点估计），而是说我们有95%的信心认为真实平均身高介于167厘米和173厘米之间。

把它想象成下注。点估计就像赌一个特定结果。置信区间就像赌结果会落在某个范围内，这会给你更多保证，尽管不如单一点估计精确。

置信区间的一般结构是：

$\text{点估计} \pm \text{误差范围}$

点估计是我们最佳的单一估计值（例如样本均值$\bar{x}$）。误差范围量化 (quantization)了该估计值的不确定性。它取决于我们对区间的信心程度以及数据中的变异性，并按样本大小进行调整。

### 置信水平

置信区间中的“置信”部分以百分比表示，通常为90%、95%或99%。这个百分比称为**置信水平**。准确理解这个水平的含义十分重要。

95%的置信水平**不**意味着我们从单一样本计算出的*特定区间*包含*真实总体参数 (parameter)*的概率为95%。这是一个常见的误解。

相反，置信水平指的是用于构建区间的**方法的可靠性**。如果我们从相同总体中抽取许多独立的随机样本，并使用相同步骤为每个样本构建置信区间，我们预期大约95%的这些区间会包含真实、未知的总体参数。另外5%的区间则纯粹由于偶然未能包含真实值。

想象一下向靶子（真实参数）射箭。每个箭的落点都是来自不同样本的点估计。置信区间就像围绕每支箭落点画一个圆。95%的置信水平意味着我们画圆的方法足够好，以至于我们画出的圆有95%会包含靶心（真实参数）。我们不知道我们为*我们的*样本画的*特定*圆是否真的包含靶心，但我们使用的方法长期来看有95%的时间都有效。



![模拟的95%置信区间 (真实均值 = 50)](plots/1427-0.json)



> 根据20个来自相同总体的不同样本计算得出的20个95%置信区间。虚线红线表示真实总体均值。大多数区间（蓝线）包含真实均值，但有一个（样本8，标红）不包含，这说明了95%置信水平的含义。

### 影响置信区间宽度的因素

误差范围，以及置信区间的宽度，受三个主要因素影响：

1. **置信水平：** 如果您想要更高的置信度（例如99%而不是95%），您需要允许更大的不确定性。这需要更大的误差范围，从而导致更宽的区间。为了更确定您的区间包含真实值，您需要撒一张更宽的网。
2. **样本大小($n$)：** 随着样本大小的增加，我们的估计变得更精确。中心极限定理告诉我们，均值的标准误差($\sigma/\sqrt{n}$)随着$n$的增大而减小。更小的标准误差导致更小的误差范围，从而得到一个更窄、更精确的置信区间。更多数据带来更多确定性。
3. **总体中的变异性（标准差$\sigma$）：** 如果总体中的数据点自然分散（标准差高），从中抽取的样本也倾向于有更大的变异性。这会增加标准误差并导致更宽的置信区间。反之，如果总体值紧密聚集（标准差低），您可以在相同的样本大小和置信水平下获得一个更窄的区间。我们通常使用样本标准差$s$来估计总体标准差$\sigma$。

### 重新审视解读

值得重申正确解读：当您计算出一个单一的95%置信区间，例如[167厘米, 173厘米]时，该表述是：

“我们有95%的信心认为，我们用于从样本生成此区间的*方法*，会生成95%的时间包含真实总体均值的区间。”

在实际操作中，我们通常会略微简化为：“我们有95%的信心认为真实总体均值介于167厘米和173厘米之间。”虽然这在长期频率解读上略微不那么精确，但它传达了实际意义：该区间代表了参数 (parameter)的一系列可能值，这些值基于观测到的样本，并且对该程序有特定置信水平。

置信区间是统计推断中的一个基本工具。它们提供了一种方式来量化 (quantization)我们对更大总体进行论断时，与样本估计相关的不确定性。这在机器学习 (machine learning)中必不可少，例如评估性能指标（如准确率、错误率）的可靠性，或了解估计模型参数中的不确定性。在下一部分，我们将看计算总体均值置信区间的具体公式和步骤。

## 参考资料

- [All of Statistics: A Concise Course in Statistical Inference](https://doi.org/10.1007/978-0-387-21736-9) — Larry Wasserman (2004)
  Publisher: Springer; Pages: 442; DOI: [10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
  对统计推断进行了严谨而易懂的阐述，包括置信区间的理论基础。
- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-9e-devore/9781305251809/) — Jay L. Devore (2015)
  Publisher: Cengage Learning
  全面介绍了概率和统计，对置信区间有清晰的解释和示例。
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://learning.oreilly.com/library/view/practical-statistics-for/9781492072942/) — Peter Bruce, Andrew Bruce, Peter Gedeck (2020)
  Publisher: O'Reilly Media
  将置信区间等统计概念与数据科学和机器学习应用联系起来。
- [Introduction to Probability and Statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff and Jonathan Bloom (2014)
  Journal: MIT OpenCourseWare; Publisher: MIT OpenCourseWare
  麻省理工开放课程资料，对概率和统计推断（包括置信区间）进行了基本阐述。
