---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-statistical-inference"
lesson: "null-alternative-hypotheses"
sourceId: 2469
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference/null-alternative-hypotheses"
title: "零假设与备择假设"
description: "学习如何建立零假设（H0）和备择假设（H1）。"
order: 5
plots: []
sourceHash: "2e0f7279c847623e3a05b87c7200d72dc6062e75a8b9a43cab4c4a3d323de6cb"
sourceCorrections: []
---

正如我们之前讨论的，假设检验提供了一种基于数据做出决策的系统方法。可以将其看作一个岔路口：根据我们样本中的证据，我们是坚持一个默认假定，还是有足够的理由相信另一种主张是真实的？这一过程的起始步骤是明确阐述这两种对立的可能性：零假设和备择假设。

### 零假设 ($H_0$)

**零假设**，通常表示为 $H_0$，代表默认立场或现状。它通常表述为“无影响”、“无差异”或“无改变”。这是我们在查看样本数据中的证据之前，最初持有的假定。在许多研究情形中，零假设实际上是研究人员试图驳斥的陈述。

从数学上看，零假设几乎总是包含某种形式的等式关系（$=$, $\le$, 或 $\ge$）。

以下是一些零假设的例子：

- **网站设计：** 一家公司测试一个新的网站布局。零假设可能是新布局的用户平均停留时间与旧布局相同。如果 $\mu_{new}$ 是新布局的平均时间，$\mu_{old}$ 是旧布局的平均时间，我们可以写成：
  $H_0: \mu_{new} = \mu_{old}$
  或者等效地：
  $H_0: \mu_{new} - \mu_{old} = 0$
- **算法表现：** 我们开发了一种新的机器学习 (machine learning)算法来预测房价。零假设可能是我们新算法的平均预测误差不优于（即，大于或等于）现有算法的误差。如果 $E_{new}$ 是新算法的平均误差，$E_{old}$ 是旧算法的平均误差，这可以写成：
  $H_0: E_{new} \ge E_{old}$
- **制造质量：** 一家工厂生产螺栓，其平均直径应为 10 毫米。质量控制的零假设可以是生产过程符合此规范：
  $H_0: \mu = 10 \text{毫米}$

零假设是我们用来比较样本证据的基准。

### 备择假设 ($H_1$ 或 $H_a$)

**备择假设**，表示为 $H_1$ 或有时为 $H_a$，是与零假设相矛盾的陈述。它代表了我们可能怀疑或希望为真，而不是零假设的情况。它通常是研究人员试图寻求证据支持的主张。

备择假设通常涉及不等号（$\ne$, $<$, 或 $>$）。它必须与零假设互斥，这意味着 $H_0$ 和 $H_1$ 不能同时为真。

让我们看看与上述例子对应的备择假设：

- **网站设计：** 如果零假设是 $H_0: \mu_{new} = \mu_{old}$，那么备择假设可能是平均时间有所不同（更高或更低）：
  $H_1: \mu_{new} \ne \mu_{old}$
  或者，如果我们特别假设*新*设计会*增加*用户参与度，备择假设将表明该方向：
  $H_1: \mu_{new} > \mu_{old}$
- **算法表现：** 如果零假设是 $H_0: E_{new} \ge E_{old}$，备择假设（我们希望证明的）是新算法的平均误差*更低*：
  $H_1: E_{new} < E_{old}$
- **制造质量：** 如果零假设是 $H_0: \mu = 10 \text{毫米}$，备择假设可能是该过程*不*符合规范（平均直径与 10 毫米不同）：
  $H_1: \mu \ne 10 \text{毫米}$

### 单尾检验与双尾检验

请注意，备择假设（$H_1$）的形式决定了检验的性质：

- **双尾检验：** 当 $H_1$ 使用“不等于”符号（$\ne$）时，我们关注检测*任一*方向的差异（例如，$\mu_{new}$ 可能大于*或*小于 $\mu_{old}$）。这称为双尾检验。
- **单尾检验：** 当 $H_1$ 使用“大于”（$>$）或“小于”（$<$）符号时，我们关注检测*仅一个特定方向*的差异。这称为单尾检验（具体而言，对于 $>$ 是右尾检验，对于 $<$ 是左尾检验）。

单尾检验和双尾检验的选择完全取决于您在查看数据*之前*试图回答的问题。您是只关心新网站设计表现是否*更好*，还是也想知道它是否表现*更差*？如果您关心检测特定方向的差异，请使用单尾检验。如果您关心检测*任何*差异，请使用双尾检验。

### 为决策奠定基础

建立零假设（$H_0$）和备择假设（$H_1$）是假设检验的必要起始步骤。这两个陈述明确了我们正在评估的具体主张以及我们正在考量的潜在备选项。它们为后续步骤奠定了基础，在这些步骤中，我们将利用样本数据和概率规则来决定我们是否有足够的证据拒绝零假设（$H_0$），转而支持备择假设（$H_1$）。我们将在接下来的部分中了解如何做出该决策，这通常会使用一个称为 p 值的工具。

## 参考资料

- [Applied Statistics and Probability for Engineers](https://www.wiley.com/en-us/Applied+Statistics+and+Probability+for+Engineers%2C+7th+Edition-p-9781119400327) — Douglas C. Montgomery and George C. Runger (2018)
  Publisher: John Wiley & Sons
  一本应用统计学标准教材，对假设检验的解释和示例，包括零假设和备择假设的构建。
- [All of Statistics: A Concise Course in Statistical Inference](https://doi.org/10.1007/978-0-387-21736-9) — Larry Wasserman (2004)
  Publisher: Springer; Pages: 442; DOI: [10.1007/978-0-387-21736-9](https://doi.org/10.1007/978-0-387-21736-9)
  一本关于统计推断的精炼且理论性强的教材，对假设检验概念提供严谨的处理。
- [Introduction to Probability and Statistics (18.05)](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/) — Jeremy Orloff, Jonathan Bloom (2014)
  Publisher: MIT OpenCourseWare
  大学水平的课程材料，介绍概率和统计推断，其中有专门讨论假设检验的部分。
