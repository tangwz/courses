---
course: "model-interpretability-explainability"
chapter: "comparing-applying-techniques"
lesson: "lime-vs-shap-conceptual"
sourceId: 4092
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-4-comparing-applying-techniques/lime-vs-shap-conceptual"
title: "LIME 与 SHAP：区别"
description: "比较 LIME 和 SHAP 的基本方法和假设。"
order: 1
plots: []
sourceHash: "441ef885e8caaba2715e6f2b89f261ef9c0b57f02eea0950bfbb8f12c4cf2e4a"
sourceCorrections: []
---

LIME 和 SHAP 是两种广泛使用的可解释性技术。尽管两者都旨在帮助理解黑盒 (black box)模型预测，尤其是在局部层面，但它们的基本原理和机制有显著不同。理解它们的基本区别对于为您的可解释性需求选择合适的工具以及正确理解它们提供的解释非常重要。

LIME（Local Interpretable Model-agnostic Explanations，局部可解释模型无关解释）基于**局部近似**的原则运行。它的主要思想很直观：即使复杂模型的决策边界在全球范围是高度非线性的，但在您想要解释的特定实例的紧邻区域，它通常可以由一个更简单、可解释的模型（如线性模型或决策树）合理地近似。

为此，LIME 的运行方式如下：

1. **扰动：** 它获取目标实例，并通过稍微修改其特征值来生成许多变体或扰动。对于表格数据，这可能涉及基于特征分布进行抽样；对于文本，则可能意味着删除词语。
2. **预测：** 它从原始黑盒模型获取这些扰动实例的预测。
3. **加权：** 它根据这些扰动实例与原始实例的接近程度，为它们分配权重 (weight)。在特征空间中更近的点被赋予更高的重要性。
4. **代理模型训练：** 它在这个加权的扰动实例集及其对应的黑盒预测上，训练一个简单、可解释的模型（“代理模型”）。
5. **解释：** 原始实例预测的解释从这个*局部代理模型*得出。例如，如果代理模型是线性模型，则该线性模型的特征系数作为解释，表示每个特征的估计局部重要性。

LIME 的主要优点在于其模型无关性以及直观的方法。它将原始模型完全视为一个黑盒，只要求能够从中获取预测。然而，解释质量很大程度上取决于局部代理模型在特定邻域中对黑盒模型行为的捕捉程度。扰动策略和代理模型类型的选择会影响最终的解释。

另一方面，SHAP（SHapley Additive exPlanations，Shapley 加性解释）的基础是**合作博弈论**，特别是 Shapley 值。想象一个游戏，其中特征“合作”产生预测。Shapley 值提供了一种独特、理论上合理的方式来将“收益”（模型对特定实例的预测与所有实例的基准或平均预测之间的差值）公平地分配给“玩家”（特征）。

SHAP 框架将这一思想应用于模型解释：

1. **基准：** 它确立一个基准值，通常是训练数据集上的平均预测。
2. **联盟：** 它考虑所有可能的特征子集（联盟）。对于每个联盟，它计算当只有这些特征存在（或已知）而其他特征不存在（或未知，通常通过边缘化或抽样处理）时模型的预测。
3. **边际贡献：** 它计算将某个特定特征添加到其尚未属于的每个可能联盟时的边际贡献。
4. **Shapley 值 (SHAP 值)：** 特征的 SHAP 值是它在所有可能联盟中的边际贡献的加权平均值。该值表示该特征将预测从基准推开的贡献。
5. **可加性：** 一个重要的属性是，所有特征的 SHAP 值之和等于实例预测与基准预测之间的差值：
   
   $$
   f(x) - E[f(X)] = \sum_{i=1}^{M} \phi_i
   $$
   
   其中 $f(x)$ 是模型对实例 $x$ 的预测，$E[f(X)]$ 是预期（基准）预测，$M$ 是特征数量，而 $\phi_i$ 是特征 $i$ 的 SHAP 值。

计算精确的 Shapley 值在计算上要求很高。因此，SHAP 采用多种近似技术：

- **KernelSHAP：** 一种模型无关的方法，使用加权线性回归（类似于 LIME 的抽样和加权，但使用符合 Shapley 的特定加权）来估计 SHAP 值。
- **TreeSHAP：** 一种高效、针对特定模型、为基于树的模型（如决策树、随机森林、XGBoost）量身定制的算法，计算精确的 SHAP 值，速度比 KernelSHAP 快得多。
- 针对深度学习 (deep learning)模型（DeepSHAP）和特定模型类型还有其他变体。

SHAP 的优点在于其源自 Shapley 值的坚实理论依据，保证了局部准确性（特征贡献的总和等于预测差值）和一致性（如果模型变得*更*依赖某个特征，该特征的重要性不应降低）等属性。

以下是主要区别的总结：

| 特点 | LIME | SHAP |
| --- | --- | --- |
| **核心思想** | 局部代理模型近似 | 博弈论（Shapley 值）特征贡献分配 |
| **理论基础** | 启发式（局部保真度） | 扎实（Shapley 属性：局部准确性、一致性） |
| **解释目标** | 通过简单代理模型解释局部模型行为 | 解释每个特征如何促使预测偏离基准 |
| **输出** | 特征权重/重要性 *针对代理模型* | 加性特征归因（SHAP 值） *针对原始模型* |
| **模型无关性** | 始终模型无关 | 模型无关（KernelSHAP）和高效的特定模型版本（TreeSHAP） |
| **一致性** | 不保证；可能随扰动/代理模型选择而异 | 由 Shapley 值属性保证 |

简而言之，LIME 提出问题：“如果我用一个简单模型*在局部*近似复杂模型，那么这个简单模型能告诉我什么关于特征重要性的信息？” SHAP 提出问题：“如何根据特征的贡献，将此特定预测与平均预测之间的差值公平地分配给所有特征？” 这种基本方法上的区别导致了它们在属性、优点和不足方面的差异，这些将在下一节中进行讨论。

## 参考资料

- ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) — Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Pages: 1135-1144; DOI: [10.48550/arXiv.1602.04938](https://doi.org/10.48550/arXiv.1602.04938)
  介绍LIME（局部可解释模型无关解释）的原始研究论文，阐述其局部近似方法和模型无关特性。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg and Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774
  SHAP（Shapley加性解释）的奠基性论文，提出其博弈论基础和各种近似Shapley值的方法，如KernelSHAP和TreeSHAP。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  一本在线书籍，提供各种可解释性技术的全面概述和比较，包含LIME和SHAP的专门章节。
