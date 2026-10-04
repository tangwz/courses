---
course: "probability-statistics-essentials-ml"
chapter: "hypothesis-testing-model-evaluation"
lesson: "p-values-explained"
sourceId: 1432
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/p-values-explained"
title: "P值说明"
description: "理解假设检验中P值的定义、计算与解释。"
order: 3
plots: ["plots/1432-0.json"]
sourceHash: "e35b41863bf14d495abfe3b588635c162bd3c1f7c7eef0eedda756a34ca3978f"
sourceCorrections: []
---

假设检验涉及通过建立原假设（$H_0$）和备择假设（$H_1$），并根据样本数据做出关于总体的决策。在做出这些决策时，可能出现两类错误：第一类错误（拒绝了真实的原假设$H_0$）和第二类错误（未能拒绝错误的原假设$H_0$）。一个主要目标是确定数据中的证据是否足够有力来拒绝$H_0$。正是P值提供了进行这项决策的标准化度量。

### 什么是P值？

P值是一个概率，衡量反对原假设证据的强度。正式定义为：

> **P值**是指在假定原假设（$H_0$）正确的情况下，获得至少与实际观察结果一样极端（或更极端）的检验结果的概率。

可以这样理解：你进行实验或收集数据，计算出一个检验统计量（例如t分数或卡方值，我们稍后会提到），然后问自己：“如果原假设确实是真的（例如，新模型*不*比旧模型好），仅仅由于随机性，看到如此极端或更极端的结果的可能性有多大？”这个可能性就是P值。

### P值的解读

其解释取决于P值的大小：

- **小的P值（例如，$\le 0.05$）：** 表明在原假设为真时，你的观测数据很不可能出现。这提供了有力证据，反对原假设。它提示可能存在某种值得注意的情况，并且观测到的效应可能并非仅仅由随机变动引起。
- **大的P值（例如，$> 0.05$）：** 表明在原假设为真时，你的观测数据是相当可能的。这提供了微弱证据，反对原假设。这不表示$H_0$是真的，只说明你的具体检验未能提供足够的证据来拒绝它。

### 判断规则：P值与Alpha（$\alpha$）的比较

为了做出正式判断，我们将P值与一个预设的临界值进行比较，这个临界值称为**显著性水平**，表示为$\alpha$（alpha）。这个$\alpha$是我们愿意容忍的第一类错误的概率。$\alpha$的常见选择有0.05（5%）、0.01（1%）或0.10（10%）。

判断规则很简单：

1. **如果$p \le \alpha$**：拒绝原假设（$H_0$）。我们判定存在统计上显著的证据支持备择假设（$H_1$）。
2. **如果$p > \alpha$**：未能拒绝原假设（$H_0$）。我们判定没有足够的统计显著证据来支持备择假设（$H_1$）。

**重要提示：** 请注意，我们说的是“未能拒绝$H_0$”，而不是“接受$H_0$”。假设检验旨在查看是否有足够的证据来*反驳*原假设，而不是证明它是真的。大的P值仅意味着我们的检验不够灵敏，或者在当前数据下确实没有可探测到的效应。

### P值如何计算

计算P值需要将你计算出的检验统计量（根据你的样本数据得出）与该统计量在*假设原假设为真*时的已知概率分布进行比较。

例如，在t检验中，你计算出一个t分数。P值就是t分布曲线中比你计算的t分数更极端区域的面积。



![P值（双尾检验）](plots/1432-0.json)



> 阴影区域代表P值（在双尾检验中分为两部分）。它是在假设$H_0$为真时，观测到与计算出的统计量（垂直虚线，统计量观测值）一样极端或更极端的检验统计量的概率。

好消息是，你很少需要手动计算这些面积。统计软件和像Python的`scipy.stats`这样的库会为你处理这些计算。你的任务是理解输入（你的数据、你的假设），并正确解读输出（P值）。

### 避免常见问题

P值很有用，但它也常被误解。请记住以下几点：

- **P值不是原假设为真的概率。** 它是在*假设*$H_0$为真时计算的。它告知你的数据与该假设的吻合程度。
- **P值不是备择假设为真的概率。**
- **统计显著性（$p \le \alpha$）不自动意味着实际重要性。** 即使实际效应量（例如，模型准确率的差异）微不足道且实际不相关，非常小的P值也可能源于非常大的数据集。应始终结合效应量来考量P值。
- **大的P值（$p > \alpha$）不证明原假设是真的。** 它只意味着你的当前数据和检验缺乏足够的证据来拒绝它。

### 机器学习 (machine learning)背景下的P值

在机器学习中，P值会在以下情况出现：

- **模型比较：** 检验新模型的性能指标（例如，准确率、F1分数）是否统计上显著优于旧模型。$H_0$可能是“这些模型性能相同。”
- **特征选择：** 检验预测变量与回归模型中的目标变量是否存在统计显著关系。$H_0$可能是“此特征的系数为零”（表示它没有线性效应）。这在线性模型的解读中很常见。
- **A/B测试：** 确定系统变更（例如，网站布局、推荐算法）是否导致用户行为（例如，点击率）发生统计显著变化。$H_0$可能是“该变更没有影响。”

了解P值能让你从仅仅观察差异，转向做出有统计依据的判断，判断这些差异是真实存在的，还是仅仅由随机性引起的。在接下来的章节中，我们将介绍产生这些P值的具体检验（例如t检验和卡方检验）。

## 参考资料

- [Practical Statistics for Data Scientists: 50+ Essential Concepts Using R and Python](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  为数据科学背景下的假设检验和P值解读提供实用指导。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer
  提供统计学概念的基础理解，包括与机器学习模型评估相关的假设检验。
- [The ASA Statement on P-Values: Context, Process, and Purpose](http://dx.doi.org/10.1080/00031305.2016.1154108) — Ronald L. Wasserstein, Nicole A. Lazar (2016)
  Journal: The American Statistician; Publisher: Taylor & Francis on behalf of the American Statistical Association; Volume: 70; Pages: 129-133; DOI: [10.1080/00031305.2016.1154108](https://doi.org/10.1080/00031305.2016.1154108)
  一份权威出版物，阐述了P值在科学研究中的定义、使用和常见误解。
- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-10e-devore/9781305251809PF/) — Jay L. Devore (2015)
  Publisher: Cengage Learning
  一本全面的教科书，涵盖了假设检验、检验统计量和P值计算的理论基础。
