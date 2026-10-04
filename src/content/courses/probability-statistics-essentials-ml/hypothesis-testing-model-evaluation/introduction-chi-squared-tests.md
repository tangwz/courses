---
course: "probability-statistics-essentials-ml"
chapter: "hypothesis-testing-model-evaluation"
lesson: "introduction-chi-squared-tests"
sourceId: 1434
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/introduction-chi-squared-tests"
title: "卡方检验介绍"
description: "了解卡方检验在分类数据分析中的应用（拟合优度检验、独立性检验）。"
order: 5
plots: []
sourceHash: "23a1dc11492e54b2330b6765c50b5c42e9e4fc6422cdb440622896f2fd7734ea"
sourceCorrections: []
---

尽管t检验是比较连续数据均值的优秀工具，但我们遇到的许多数据，尤其是在机器学习 (machine learning)的分类任务中，是分类数据。我们如何检验关于频率的假设或检查不同类别之间的关系呢？卡方（$\chi^2$）检验提供了一种统计方法来处理这些情况。它们通过比较样本数据中不同类别的*观察*计数与在特定零假设为真时我们*预期*会看到的计数来运作。

### 理解卡方统计量

任何卡方检验的核心是其$\chi^2$统计量本身。它衡量并总结了每个类别中观察频率($O_i$)与零假设下的预期频率($E_i$)之间的差异。计算遵循以下一般形式：

$\chi^2 = \sum_{\text{所有类别 } i} \frac{(O_i - E_i)^2}{E_i}$

直观上，如果观察计数与预期计数非常接近，那么差异$(O_i - E_i)$将很小，从而得到一个小的$\chi^2$值。这表明数据与零假设吻合良好。反之，观察计数与预期计数之间的大差异会导致大的$\chi^2$值，这提供了*反对*零假设的证据。

### 常见的卡方检验

两种主要类型的卡方检验与数据分析和机器学习 (machine learning)应用尤为相关：

1. **卡方拟合优度检验：** 当您有一个分类变量，并想确定其观察频率分布是否与特定的理论或假设分布存在显著差异时，会使用此检验。

   - **零假设 ($H_0$)**：观察频率与基于假设分布的预期频率相符。
   - **备择假设 ($H_1$)**：观察频率*不*与预期频率相符。
   - **例子**：假设一位网站所有者假设用户流量在工作日（周一至周五，每天20%）均匀分布。他们收集了一个月的每日访问数据。拟合优度检验将比较每个工作日访问量的*观察*比例与*预期*比例（20%）。一个显著的结果（大的$\chi^2$）将表明流量*不*是均匀分布的。
2. **卡方独立性检验：** 当您有两个分类变量并想确定它们之间是否存在统计上显著的关联或关系时，会使用此检验。它有助于回答问题：“这两个变量是独立的，还是一个变量的类别取决于另一个变量的类别？”此检验的数据通常以列联表的形式呈现。

   - **零假设 ($H_0$)**：这两个变量是独立的（它们之间没有关联）。
   - **备择假设 ($H_1$)**：这两个变量是相关的（它们之间存在关联）。
   - **例子**：考虑分析客户数据，看“订阅计划”（例如，基本型、高级型、专业型）的选择是否与所使用的“设备类型”（例如，移动设备、桌面设备）相关。列联表将显示每种组合的计数（例如，使用基本计划的移动设备用户数量）。该检验会计算每个单元格的预期计数，假设计划选择和设备类型是独立的。将这些预期计数与观察计数进行比较，即可得出$\chi^2$统计量。如果检验结果显著，则表明订阅计划的选择*确实*与所使用的设备类型相关。

> 进行卡方检验的流程是比较观察计数与预期计数（从零假设推导而来），以计算$\chi^2$统计量，进而得出P值和统计决策。

### 自由度与结果解读

卡方统计量遵循一种特定的概率分布，即卡方分布。与t分布类似，其形态取决于自由度($df$)。不同检验的$df$计算方式略有不同：

- **拟合优度检验**：$df = k - 1$，其中$k$是变量的类别数量。
- **独立性检验**：$df = (\text{行数} - 1) \times (\text{列数} - 1)$，基于列联表的维度。

知道$\chi^2$统计量和$df$后，我们可以找到与检验结果相关的P值。结果的解读与其他假设检验保持一致：P值表示在*零假设实际为真*的情况下，从我们的数据中获得一个与计算出的$\chi^2$值一样极端或更极端的$\chi^2$值的概率。一个小的P值（通常小于预定的显著性水平$\alpha$，例如0.05）会使我们拒绝零假设。

### 假设与机器学习 (machine learning)中的应用

为使卡方检验得出可靠结果，通常应满足以下条件：

- 数据必须以分类变量的频率或计数形式呈现。
- 构成计数的各个观测值应相互独立。
- 每个类别（或列联表中的单元格）的预期频率不应太小。一个常见指导原则是，大多数（例如，>80%）预期频率应为5或更大，并且没有一个应小于1。

在机器学习中，卡方检验常用于：

- **特征选择**：独立性检验常用于评估分类输入特征与分类目标变量之间的关系（例如，在分类问题中）。与目标显著关联的特征通常被认为对模型更具意义。
- **模型诊断**：拟合优度检验可能可用于比较分类模型预测类别的分布与数据集中实际分布，尽管其他指标通常更受青睐。
- **数据查看**：在探索性数据分析（EDA）阶段理解分类数据中的关系。

卡方检验将我们的假设检验能力扩展到分类数据，为评估分布和关联提供了有价值的方法。Python库如SciPy包含相关函数（例如，用于拟合优度检验的`scipy.stats.chisquare`，用于独立性检验的`scipy.stats.chi2_contingency`），使得执行这些检验在计算上变得简单，我们将在后续章节中看到。

## 参考资料

- [Introduction to the Practice of Statistics](https://www.vitalsource.com/products/introduction-to-the-practice-of-statistics-david-s-moore-george-p-mccabe-bruce-a-craig-v9781319383671) — David S. Moore, George P. McCabe, Bruce A. Craig (2021)
  Publisher: W. H. Freeman and Company
  一本广泛使用的教材，对统计方法进行了基础而清晰的阐述，包括对拟合优度检验和独立性检验的卡方检验、其假设以及如何解释结果的详细说明。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani (2013)
  Publisher: Springer
  本书对机器学习的统计方法进行了广泛介绍，为理解假设检验及其在分类问题特征选择等领域的应用提供了必要的统计背景。
- [\`scipy.stats.chi2_contingency\` and \`scipy.stats.chisquare\` Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html) — SciPy Developers (2023)
  SciPy 函数的官方文档，提供了在 Python 中执行独立性卡方检验和拟合优度卡方检验的实际实现方法，详细说明了其参数和在数据分析中的用法。
