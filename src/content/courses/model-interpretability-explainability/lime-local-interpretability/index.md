---
course: "model-interpretability-explainability"
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-2-lime-local-interpretability"
sourceId: 808
chapter: "lime-local-interpretability"
title: "局部可解释模型无关解释 (LIME)"
order: 2
description: "学习 LIME 的原理及其在说明机器学习模型单个预测结果上的运用。"
hasQuiz: true
---

了解机器学习模型的整体行为表现是有帮助的，但很多时候我们需要弄清楚为什么会做出某个具体的预测。对于单个样本，我们如何说明复杂、不透明模型的输出呢？本章将介绍局部可解释模型无关解释 (LIME)，这是一种专门为此目的而设计的方法。

LIME 的工作方式是，通过采用一个更简单、可解释的模型，在您想要解释的预测附近局部地模拟复杂模型。由于它将原始模型视为黑箱，因此几乎可以应用于任何分类器或回归器。

在本章中，您将学到：

*   LIME 的核心思路及其生成解释的方式。
*   LIME 的运作机制，包括数据扰动和局部替代模型的使用。
*   如何应用 LIME 来说明在表格数据和文本数据上训练的模型的预测结果。
*   理解 LIME 生成的特征重要性分数和可视化结果的方法。
*   使用 LIME Python 库进行实际操作。
*   使用 LIME 时需要注意的重要考量和局限性。

最后，我们将通过一个动手练习结束本章，您将在其中应用 LIME 为预训练模型生成并分析解释。
