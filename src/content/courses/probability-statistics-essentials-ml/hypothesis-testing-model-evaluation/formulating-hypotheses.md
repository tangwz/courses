---
course: "probability-statistics-essentials-ml"
chapter: "hypothesis-testing-model-evaluation"
lesson: "formulating-hypotheses"
sourceId: 1430
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/formulating-hypotheses"
title: "制定零假设与备择假设"
description: "学习如何设定零假设 (H0) 和备择假设 (H1) 用于统计检验。"
order: 1
plots: []
sourceHash: "3f0635061876c8c53f537c6ec9720946a87665ae764869e5c94716ba67b79dd1"
sourceCorrections: []
---

假设检验始于将我们想回答的问题转化为两个相互对立的陈述：零假设和备择假设。可以将其视为设定一场统计学的“法庭审理”。零假设代表最初的假定或现状（比如“在被证明有罪之前是无辜的”），而备择假设代表我们试图寻找证据来支持的主张或影响。

### 零假设 ($H_0$)

**零假设**，记作 $H_0$，是指没有影响、没有差异或没有关系的陈述。它通常代表默认的信念状态或一个我们用来检验证据的基准线。这是我们暂时假定为真，并试图寻找*反对*证据的假设。

从数学上看，零假设通常涉及等式（$=$）或在其陈述中包含等式（$\leq$ 或 $\geq$）。

**机器学习 (machine learning)中的常见例子：**

- **模型比较：** 假设我们开发了一种新的分类算法，并想比较其平均准确率（$μ_{new}$）与现有算法（$μ_{old}$）的平均准确率。零假设可能是准确率没有差异：
  $H_0: μ_{new} = μ_{old}$
  或者，如果我们只关注新模型是否*不差于*旧模型，它可以是：
  $H_0: μ_{new} \geq μ_{old}$
- **特征重要性：** 我们想检验某个特定特征是否与目标变量有线性关系。零假设会陈述相关系数（$\rho$）为零：
  $H_0: \rho = 0$
- **A/B测试：** 一家公司测试一个新的网站设计，看它是否能提升用户转化率（$p$）。零假设可能陈述新设计（$p_{new}$）不优于旧设计（$p_{old}$）：
  $H_0: p_{new} \leq p_{old}$

假设检验的目的不是*证明* $H_0$ 为真，而是确定我们的样本数据中是否有足够的统计证据来*拒绝*它，从而支持另一种解释。

### 备择假设 ($H_1$ 或 $H_a$)

**备择假设**，记作 $H_1$（有时也记作 $H_a$），是与零假设相矛盾的陈述。它代表我们实际上有兴趣检测或证明的结果。这是需要证据才能被接受的主张。

备择假设通常涉及不等式（$\neq$，$<$，或 $>$）。它必须与零假设互斥，这意味着 $H_0$ 和 $H_1$ 不能同时为真。理想情况下，它们应涵盖所检验参数 (parameter)的所有可能结果。

**与上述 $H_0$ 对应的例子：**

- **模型比较：**
  - 如果我们想知道准确率是否只是*不同*：
    $H_1: μ_{new} \neq μ_{old}$ （这是一个**双尾**检验，因为我们寻找任何方向上的差异）。
  - 如果我们专门假设新模型*更好*：
    $H_1: μ_{new} > μ_{old}$ （这是一个**单尾**或**方向性**检验）。
- **特征重要性：**
  - 如果我们想知道是否存在*任何*线性关系（正向或负向）：
    $H_1: \rho \neq 0$ （双尾检验）。
- **A/B测试：**
  - 如果公司希望新设计能*提升*转化率：
    $H_1: p_{new} > p_{old}$ （单尾检验）。

单尾和双尾备择假设的选择完全取决于研究问题。你是对检测任何差异感兴趣，还是只对特定方向的差异感兴趣？

### 假设制定：好的做法

1. **在数据收集前确定：** 在收集或分析数据*之前*制定 $H_0$ 和 $H_1$ 是十分必要的。这可以防止数据影响假设，保持客观性。
2. **关注总体参数 (parameter)：** 假设是对总体参数（例如总体均值 $μ$、总体比例 $p$ 或相关系数 $\rho$）的陈述，而不是对样本统计量（例如样本均值 $\bar{x}$ 或样本比例 $\hat{p}$）的陈述。我们使用样本统计量来对这些总体参数进行推断。
3. **确保互斥和（通常）穷尽：** $H_0$ 和 $H_1$ 不应重叠，理想情况下，它们应涵盖参数值的所有可能性。例如，如果 $H_0: μ = 10$，则双尾 $H_1$ 是 $μ \neq 10$。如果 $H_0: μ \leq 10$，则对应的 $H_1$ 必须是 $μ > 10$。

明确定义零假设和备择假设是假设检验过程中的第一个根本步骤。它为收集证据和对所调查的主张做出统计决策奠定了基础。一旦假设确定，接下来的步骤包括选择显著性水平、收集数据、计算检验统计量，并将其与临界值进行比较或使用p值，这些内容我们将在下文介绍。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-3-031-03893-0](https://doi.org/10.1007/978-3-031-03893-0)
  涵盖统计学习方法及其基础统计推断原则，有助于理解模型评估中假设构建。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  关于统计学习的经典资料，提供了数据分析和机器学习中常应用的统计推断的严谨处理。
- [Introduction to Probability and Statistics, Spring 2014](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Journal: MIT OpenCourseWare; Publisher: Massachusetts Institute of Technology
  提供讲义和材料，阐明假设检验的基本概念，对初学者有益。
