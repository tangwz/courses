---
course: "probability-statistics-fundamentals-ml"
chapter: "intro-statistical-inference"
lesson: "inference-ml-evaluation"
sourceId: 2472
sourceUrl: "https://apxml.com/zh/courses/probability-statistics-fundamentals-ml/chapter-5-intro-statistical-inference/inference-ml-evaluation"
title: "统计推断与机器学习评估的联系"
description: "了解统计推断的思路如何关联到评估机器学习模型的性能和重要性。"
order: 7
plots: ["plots/2472-0.json"]
sourceHash: "05a20a0a202bfde3346fb808d0048a9c0a3cba6d4a9bb91ed2bca6f294e4caad"
sourceCorrections: []
---

您已经了解了统计推断如何帮助我们基于较小样本对大量总体进行合理的推断。我们研究了如何估计特定值（点估计）、理解这些值可能所在的范围（置信区间），以及正式检验关于总体的断言（假设检验）。这与评估机器学习 (machine learning)模型有何关联？事实证明，关联非常直接。

当我们训练机器学习模型时，通常会在一个称为测试集的独立数据集上评估其性能。这个测试集就像我们的样本。我们计算的性能指标，例如准确率、精确率或均方误差，本质上是一个**点估计**。它是我们基于测试集样本，对模型在*所有可能未见过的数据*（总体）上表现如何的最佳估计。

“就像任何样本统计量一样，这个性能指标也存在不确定性。如果我们使用不同的测试集（另一个样本），我们可能会得到略有不同的性能得分。这就是**置信区间**变得有用之处。与其仅仅报告“模型达到了92%的准确率”，我们可以计算一个置信区间，例如说明：“我们有95%的信心认为模型在未见过数据上的真实准确率在89%到95%之间。”这能更清晰地描绘模型的预期表现以及我们估计的可靠性。较窄的区间表示更精确的估计，通常源于更大的测试集。”

**假设检验**在比较模型或评估变化时起着重要作用。假设您开发了两个模型，模型A和模型B，并且想知道模型B是否确实优于模型A。

- 模型A在测试集上获得85%的准确率。
- 模型B在测试集上获得87%的准确率。

模型B是否确实更好，还是这2%的差异仅仅是因为恰好落在我们测试集中的特定数据点（即随机机会）？假设检验提供了一个回答此问题的框架：

1. **提出假设：**

   - **零假设 ($H_0$):** 模型A和模型B在性能上没有实际差异。它们的真实准确率相等 ($accuracy_A = accuracy_B$)。观察到的差异是由于抽样变异性造成的。
   - **备择假设 ($H_1$):** 性能存在实际差异。模型B的真实准确率高于模型A的 ($accuracy_B > accuracy_A$)。（注意：我们也可以检验 $accuracy_B \neq accuracy_A$）。
2. **检验假设：** 我们将使用统计检验（具体检验取决于指标和数据）来计算基于观察到的性能差异和样本量（测试集大小）的**p值**。
3. **解读p值：**

   - 一个**小的p值**（通常小于0.05）提供了反对零假设的证据。它表明如果模型真实性能相同，则观察到2%（或更大）的性能差异是不太可能的。我们可能会得出结论，模型B*统计学上显著地*优于模型A。
   - 一个**大的p值**（大于或等于0.05）意味着我们没有足够的证据拒绝零假设。观察到的2%差异完全可能是由于随机机会造成。我们不能自信地声称模型B基于此检验更优。

这个框架有助于避免我们过度解读可能只是噪声的微小性能提升。它鼓励对模型比较采取更严谨的方法。

考虑这个使用置信区间比较两个模型估计性能的可视化：



![模型性能与置信区间比较](plots/2472-0.json)



> 条形图显示了模型A（85%）和模型B（87%）的点估计（测试集上的平均准确率）。误差条表示95%置信区间。请注意，这些区间显著重叠，表明差异可能不具有统计显著性。假设检验将提供一个正式的p值来量化 (quantization)这一点。

在比较模型时，假设检验的思路有时出现在某些模型*内部*。例如，在线性回归中，统计检验常用于判断输入特征与输出变量之间是否存在统计学上的显著关联（即其系数是否显著不为零）。

总而言之，统计推断提供了以下工具，用于：

- 了解测试集上的性能指标是估计值，而非真实情况。
- 使用置信区间量化这些估计的不确定性。
- 使用假设检验正式检验模型间观察到的性能差异是统计显著的还是可能由偶然造成。

应用这些观点有助于您在评估和比较机器学习模型时做出更明智、更可靠的决策，超越简单的点估计比较，转向理解结果的意义和确定性。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  这本统计学习的经典参考书涵盖了理论基础和实用算法。其关于模型评估和选择的章节对于理解机器学习模型评估中的不确定性至关重要。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2013)
  Publisher: Springer
  这本统计学习的入门书籍更易于理解，为理解统计推断如何应用于机器学习模型评估提供了坚实的基础，涵盖了测试集和性能指标等概念。
- [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms](https://doi.org/10.1162/089976698300017197) — Thomas G. Dietterich (1998)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 10; Pages: 1895-1923; DOI: [10.1162/089976698300017197](https://doi.org/10.1162/089976698300017197)
  这篇基础性论文介绍了用于比较两种监督分类算法性能的近似统计检验方法，例如 McNemar 检验和配对 t 检验，直接满足了机器学习评估中严格假设检验的需求。
- [Probability and Statistics for Engineering and the Sciences](https://www.cengage.com/c/probability-and-statistics-for-engineering-and-the-sciences-10e-devore/9780357539156/) — Jay L. Devore (2021)
  Publisher: Cengage Learning
  这本综合性教科书全面介绍了概率论和统计推断，包括点估计、置信区间和假设检验的详细解释，为所讨论的统计概念提供了坚实的基础。
