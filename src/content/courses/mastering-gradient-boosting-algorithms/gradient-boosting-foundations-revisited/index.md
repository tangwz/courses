---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-1-gradient-boosting-foundations-revisited"
sourceId: 526
chapter: "gradient-boosting-foundations-revisited"
title: "梯度提升算法核心原理回顾"
order: 1
description: "回顾集成方法、决策树的核心思想，以及梯度提升的根本原理。"
hasQuiz: false
---

本章回顾构成进阶梯度提升算法根基的根本原理。要理解XGBoost、LightGBM和CatBoost等具体实现方式的运作机制和优化方法，我们首先需要清晰地了解它们的由来以及演变而来的核心思想。

我们将首先简要回顾集成方法，着重说明提升法与袋装法等技术有何不同。接着，我们将审视决策树，它们是提升框架中常用的基本学习器，并讨论使其适用的特性。之后，我们将重新审视加性建模方法，其中模型是按顺序构建的，每个新模型都试图纠正前一个模型的错误。

理解这些模型如何学习涉及到优化，因此我们将简要提及梯度下降在此背景下的基本原理。最后，我们将介绍经典梯度提升机（GBM）算法本身，为后续章节分析其更进阶的演变版本奠定基础。本次回顾确保每个人在进入更复杂的内容之前都具备必要的背景知识。
