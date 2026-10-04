# SHAP 值的特性

来源：[原文](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-properties)

[返回章节目录](README.md) · [返回课程目录](../README.md)

SHAP（SHapley 加性解释）值源自合作博弈论中的Shapley值，提供了一种系统性方法，将模型预测的输出归因于其输入特征。但是，是什么让SHAP值成为一种特别有吸引力的方法呢？它们的优势在于其独有的、在一众加性特征归因方法中满足的一系列理想数学特性。理解这些特性有助于阐明为什么SHAP提供一致且可靠的解释。

SHAP解释遵循从Shapley值继承而来的三个主要特性：

### 1. 局部准确性（或加性）

此特性确保对单个预测提供的解释能准确反映模型对该实例的输出。具体而言，给定预测的所有输入特征（$M$）的SHAP值（$\phi_i$）之和，等于模型对该实例的预测（$f(x)$）与整个数据集或背景数据集上的平均预测（$E[f(X)]$，也称为基准值或期望值）之间的差值。

在数学上，这表达为：


$$
\sum_{i=1}^{M} \phi_i = f(x) - E[f(X)]
$$


- $f(x)$ 是模型对输入实例 $x$ 的预测值。
- $E[f(X)]$ 是模型输出在背景数据集上的期望值（通常是回归的平均预测值或分类的平均概率值）。当没有特征信息时，这作为基准预测。
- $\phi_i$ 是特征 $i$ 的SHAP值，表示其对预测值偏离基准值的贡献。

**意义：** 局部准确性确保特征贡献的总和精确地解释了特定预测与基准线之间的差异。它提供了在局部层面预测值的完整归因。如果一个特征使预测值更高，它将获得正的SHAP值；如果使其更低，它将获得负的SHAP值。这些影响的总和与模型偏离平均值的程度精确匹配。

### 2. 缺失性

缺失性特性表明，对预测确实没有影响的特征应被赋予零SHAP值。如果特征 $x_i$ 在其值不影响模型对特定实例 $x$ 的输出（可能是因为它未包含在正在评估的特征子集中，或者其值使得模型忽略它）的意义上是“缺失”的，那么其归因重要性应为零。

符号上，如果特征 $i$ 没有边际贡献，则：


$$
\phi_i = 0
$$


**意义：** 这与直觉一致。对预测结果没有贡献的特征，不应在解释中获得任何功劳或责任。这避免了归因方法将虚假重要性赋予不相关输入。

### 3. 一致性（或单调性）

一致性可以说是区分SHAP与其他方法的最重要特性。它表明，如果模型发生变化，使某个特征的边际贡献增加或保持不变（无论是否存在其他特征），则分配给该特征的SHAP值也应增加或保持不变。如果特征的影响变得严格更大，其值绝不应减少。

让我们考虑两个模型，$f$ 和 $f'$。如果对于特定特征 $i$ 和其他特征的所有可能子集 $S$，特征 $i$ 在模型 $f'$ 中的贡献大于或等于其在模型 $f$ 中的贡献：


$$
f'(x_{S \cup \{i\}}) - f'(x_S) \ge f(x_{S \cup \{i\}}) - f(x_S)
$$


那么，一致性特性确保特征 $i$ 的SHAP值也将反映这种增加的重要性：


$$
\phi_i(f', x) \ge \phi_i(f, x)
$$


**意义：** 一致性确保特征重要性度量与模型实际使用特征的方式相符。如果一个特征在模型逻辑中变得统一地更重要，其SHAP值不会出现反常的下降。这提供了一个保证，即SHAP值能忠实地呈现特征影响，使得解释比可能违反此特性的方法更可靠、更值得信赖。例如，从基于树的模型（如增益或置换重要性）中得出的标准特征重要性度量并不总是满足一致性。

这三个特性根植于博弈论，为SHAP提供了坚实的理论根基。它们确保解释在局部层面是准确的（局部准确性），忽略不相关特征（缺失性），并可靠地反映特征贡献的变化（一致性）。这些特性使SHAP成为一个用于解释复杂机器学习 (machine learning)模型的强大且被广泛采用的框架。在确立了这些特性之后，我们现在可以考虑计算SHAP值的实用方法。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) — Scott M. Lundberg and Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765–4774; DOI: [10.5555/3295222.3295325](https://doi.org/10.5555/3295222.3295325)
  介绍了SHAP并正式定义了其源自Shapley值的性质。
- [A Value for n-person Games](https://www.jstor.org/stable/2253866) — Lloyd S. Shapley (1953)
  Journal: Contributions to the Theory of Games; Publisher: Princeton University Press; Volume: II; Pages: 307-317
  介绍了Shapley值的基本概念，SHAP便是基于此。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  提供SHAP值及其性质的易懂解释，并将其置于可解释机器学习的更广泛背景中。

---

[上一节](02-SHAP%20%E5%80%BC%EF%BC%9AShapley%20%E5%80%BC%E4%B8%8E%E6%A8%A1%E5%9E%8B%E7%89%B9%E5%BE%81%E7%9A%84%E5%85%B3%E8%81%94.md) · [下一节](04-KernelSHAP%EF%BC%9A%E4%B8%80%E7%A7%8D%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%97%A0%E5%85%B3%E7%9A%84%E6%96%B9%E6%B3%95.md)
