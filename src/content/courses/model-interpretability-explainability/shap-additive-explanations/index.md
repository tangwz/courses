---
course: "model-interpretability-explainability"
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations"
sourceId: 810
chapter: "shap-additive-explanations"
title: "SHapley 可加性解释 (SHAP)"
order: 3
description: "了解 SHAP 值在模型可解释性方面的理论背景和实际使用。"
hasQuiz: true
---

在考察了使用 LIME 的局部代理模型后，本章将介绍 SHapley 可加性解释 (SHAP)，这是一种基于合作博弈论原理来理解模型预测的不同方法。

你将了解到 Shapley 值的理论依据，以及 SHAP 框架如何调整这些值，为每个特征对特定预测的贡献分配一个重要性值。我们将讨论使 SHAP 值成为特征重要性的一致且准确衡量标准的重要性质。

本章涵盖计算 SHAP 值的不同方法：

*   **KernelSHAP**：一种模型无关的方法，适用于任何机器学习模型。
*   **TreeSHAP**：一种高效方法，专门为决策树、随机森林和梯度提升机等基于树的模型设计。

此外，你将学习如何使用 SHAP 的 Python 库实现 SHAP，并解释常见的可视化图表，如力图、摘要图和依赖图，以便了解个别预测（局部解释）和整体模型行为（全局解释）。实际示例将引导你生成和理解这些解释。
