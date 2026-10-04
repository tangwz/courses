---
course: "model-interpretability-explainability"
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-1-foundations-interpretability"
sourceId: 806
chapter: "foundations-interpretability"
title: "模型可解释性的基本原理"
order: 1
description: "了解机器学习中模型可解释性的重要性、定义及分类。"
hasQuiz: true
---

机器学习模型，尤其是复杂的模型，经常表现为“黑箱”。它们的内部运作方式并非一目了然，然而，理解给定特定输入 ($X$) 后，它们 *为何* 会产生特定输出 ($y$) 变得日益重要，这对于建立信任、调试错误和保证公平性都很有帮助。本章建立了进行模型解释所需的基本思想。

我们将首先讨论解释模型预测的动因。接着，我们将阐明可解释性与可说明性之间的区别。之后，我们将考察不同解释方法的分类，考量它们的特点，例如它们是内置的（内禀）还是事后应用的（事后），以及它们是依赖于特定模型类型（模型特定）还是可用于任何模型（模型无关）。我们还将区分理解整体模型逻辑（全局解释）与解释单个预测（局部解释）。最后，我们将提及一些在解释模型时遇到的一些常见的难题。
