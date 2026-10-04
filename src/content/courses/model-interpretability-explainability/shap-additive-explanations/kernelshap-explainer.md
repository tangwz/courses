---
course: "model-interpretability-explainability"
chapter: "shap-additive-explanations"
lesson: "kernelshap-explainer"
sourceId: 4078
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/kernelshap-explainer"
title: "KernelSHAP：一种与模型无关的方法"
description: "了解 KernelSHAP，一种估算任何模型 SHAP 值的方法。"
order: 4
plots: []
sourceHash: "41da3881e8c2737a5cd1c5f80a87bc2d81a5bdffb5a83c85b53b40d882a5fe10"
sourceCorrections: []
---

Shapley 值建立了公平特征归因的理论基础。然而，精确计算它们通常需要在所有可能的特征子集上评估模型（对于 $M$ 个特征，需要进行 $2^M$ 次评估），这对于大多数模型来说在计算上是不可行的。为此，近似方法应运而生，其中 KernelSHAP 是一种广泛使用的、与模型无关的方法，用于估算这些值。

可以将 KernelSHAP 理解为在 LIME 中看到的想法与 Shapley 值的理论严谨性之间架起了一座桥梁。与 LIME 类似，KernelSHAP 通过拟合 (overfitting)一个局部替代模型来解释预测。然而，KernelSHAP 并非使用任意扰动，而是采用了一种直接源自 Shapley 值性质的特定采样策略和加权方案。这使其与模型无关——它不需要了解复杂模型的内部机制，只需要知道其输入和输出表现即可。

### KernelSHAP 的工作原理

KernelSHAP 估算每个特征对特定实例 $x$ 预测的贡献。以下是该过程的分解：

1. **采样联盟**: 为了估算某个特征的影响，我们需要评估模型在有该特征存在时与“不存在”时的输出。KernelSHAP 通过生成样本（通常称为“联盟”）来实现这一点。一个联盟代表了原始特征中存在的某个子集。
2. **表示特征缺失**: 对于一个期望所有特征都存在的模型，我们如何模拟某个特征“缺失”呢？KernelSHAP 使用一个**背景数据集**。为了创建一个表示特定联盟（特征子集）的扰动样本，联盟中*存在*的特征会保留我们想要解释的实例 $x$ 中的值。联盟中*缺失*的特征则从背景数据集中取样值。这个背景数据集理想情况下代表了特征值的预期分布，使我们能够通过用典型值替换来近似移除特征的影响。
3. **模型评估**: 使用原始黑盒 (black box)模型对每个生成的扰动样本进行预测。
4. **加权线性回归**: KernelSHAP 随后根据这些预测拟合一个加权线性回归模型。该线性模型的输入是二进制指示符（$z'$），表示每个原始特征是否在扰动样本中存在（$z'_i=1$）或缺失（$z'_i=0$）。目标变量是原始模型对该扰动样本所做的预测。
   模型形式为：
   $g(z') = \phi_0 + \sum_{i=1}^{M} \phi_i z'_i$
   这里，$g(z')$ 近似原始模型对 $z'$ 所表示的联盟的预测，$M$ 是特征的数量，$\phi_0$ 是基准值（背景数据集上的平均预测，即 $E[f(X)]$），而系数 $\phi_i$ 是每个特征 $i$ 的估算 SHAP 值。
5. **SHAP 核权重 (weight)**: 主要之处在于加权。样本在回归中并非等权重。KernelSHAP 使用特定的权重，称为 SHAP 核，这些权重源自 Shapley 值定义。这些权重对特征很少（例如，只有一个特征存在）或几乎所有特征都存在的联盟（例如，只有一个特征缺失）赋予更多重要性。这种加权方案确保所得系数（$\phi_i$）通过此估算过程尽可能地满足 Shapley 值的理想属性，例如局部准确性和一致性。

> KernelSHAP 工作流程的简化视图，强调了如何使用实例、背景数据、模型评估和加权线性回归来估算 SHAP 值。

### 实际考量

- **模型无关性**: KernelSHAP 的主要优势在于它能够与任何机器学习 (machine learning)模型配合使用，无论其内部结构如何。只要你能提供输入并获得输出，就可以使用 KernelSHAP。
- **背景数据集的选择**: 背景数据集的选择非常重要。它定义了“缺失”特征的基准。一个常见的方法是使用训练数据的代表性样本或摘要版本（例如，使用 k-means 聚类中心）来减少计算量。SHAP 库通常默认使用零值或均值/众数，但为了获得更有意义的结果，通常建议提供数据样本。
- **计算成本**: 生成样本、运行模型预测和拟合加权回归可能计算量很大，特别是对于预测时间较长、数据维度较高或使用大型背景数据集的模型。准确估算所需的样本数量会随着特征数量的增加而变化。
- **近似质量**: 请记住，KernelSHAP 提供的是真实 Shapley 值的*估算值*。准确性取决于所使用的样本数量和背景数据集的适用性。对于基于树的模型，像 TreeSHAP（接下来会讨论）这样的专门算法可以更高效地计算精确的 SHAP 值。

KernelSHAP 提供了一种强大且有理论依据的方法，可以将 Shapley 值的原则应用于几乎任何机器学习模型。它作为 SHAP 框架中的一个基本工具，即使对于模型内部机制无法访问的复杂黑盒 (black box)系统，也能实现解释。然而，它的计算需求意味着对于像树这样的特定模型类型，通常更倾向于使用更高效的专门方法。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774; DOI: [10.5555/3295222.3295328](https://doi.org/10.5555/3295222.3295328)
  介绍SHAP、KernelSHAP和TreeSHAP的原始研究，基于Shapley值统一了模型解释方法。
- [SHAP (SHapley Additive exPlanations) Documentation](https://shap.readthedocs.io/en/latest/) — Scott Lundberg and SHAP Contributors (2024)
  提供SHAP（包括KernelSHAP）的实践指导和API参考，包含代码示例。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/shap.html) — Christoph Molnar (2024)
  Publisher: Self-published
  详细易懂地解释了KernelSHAP及其在模型可解释性中的背景，基于原始论文。
