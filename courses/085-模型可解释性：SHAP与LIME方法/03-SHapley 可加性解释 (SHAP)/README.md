# 第 3 章：SHapley 可加性解释 (SHAP)

来源：[原章节](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations)

[返回课程目录](../README.md)

在考察了使用 LIME 的局部代理模型后，本章将介绍 SHapley 可加性解释 (SHAP)，这是一种基于合作博弈论原理来理解模型预测的不同方法。

你将了解到 Shapley 值的理论依据，以及 SHAP 框架如何调整这些值，为每个特征对特定预测的贡献分配一个重要性值。我们将讨论使 SHAP 值成为特征重要性的一致且准确衡量标准的重要性质。

本章涵盖计算 SHAP 值的不同方法：

*   **KernelSHAP**：一种模型无关的方法，适用于任何机器学习模型。
*   **TreeSHAP**：一种高效方法，专门为决策树、随机森林和梯度提升机等基于树的模型设计。

此外，你将学习如何使用 SHAP 的 Python 库实现 SHAP，并解释常见的可视化图表，如力图、摘要图和依赖图，以便了解个别预测（局部解释）和整体模型行为（全局解释）。实际示例将引导你生成和理解这些解释。

## 小节

- 1. [Shapley 值概述](01-Shapley%20%E5%80%BC%E6%A6%82%E8%BF%B0.md)
- 2. [SHAP 值：Shapley 值与模型特征的关联](02-SHAP%20%E5%80%BC%EF%BC%9AShapley%20%E5%80%BC%E4%B8%8E%E6%A8%A1%E5%9E%8B%E7%89%B9%E5%BE%81%E7%9A%84%E5%85%B3%E8%81%94.md)
- 3. [SHAP 值的特性](03-SHAP%20%E5%80%BC%E7%9A%84%E7%89%B9%E6%80%A7.md)
- 4. [KernelSHAP：一种与模型无关的方法](04-KernelSHAP%EF%BC%9A%E4%B8%80%E7%A7%8D%E4%B8%8E%E6%A8%A1%E5%9E%8B%E6%97%A0%E5%85%B3%E7%9A%84%E6%96%B9%E6%B3%95.md)
- 5. [TreeSHAP：针对树模型优化](05-TreeSHAP%EF%BC%9A%E9%92%88%E5%AF%B9%E6%A0%91%E6%A8%A1%E5%9E%8B%E4%BC%98%E5%8C%96.md)
- 6. [解读 SHAP 图：力图](06-%E8%A7%A3%E8%AF%BB%20SHAP%20%E5%9B%BE%EF%BC%9A%E5%8A%9B%E5%9B%BE.md)
- 7. [解读SHAP图：概览图与依赖图](07-%E8%A7%A3%E8%AF%BBSHAP%E5%9B%BE%EF%BC%9A%E6%A6%82%E8%A7%88%E5%9B%BE%E4%B8%8E%E4%BE%9D%E8%B5%96%E5%9B%BE.md)
- 8. [SHAP 在 Python 中的实现](08-SHAP%20%E5%9C%A8%20Python%20%E4%B8%AD%E7%9A%84%E5%AE%9E%E7%8E%B0.md)
- 9. [SHAP值计算：动手实践](09-SHAP%E5%80%BC%E8%AE%A1%E7%AE%97%EF%BC%9A%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5.md)

章节测验：[在线测验](https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/quiz)
