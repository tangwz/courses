---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost"
sourceId: 1314
chapter: "advanced-gradient-boosting-lightgbm-catboost"
title: "高级梯度提升：LightGBM与CatBoost"
order: 5
description: "了解LightGBM和CatBoost。学习它们如何处理类别数据、提升训练速度，并与XGBoost的性能进行比较。"
hasQuiz: false
---

XGBoost相较于标准梯度提升算法有了显著提升，但也有其他专业库随之出现，旨在解决特定的性能瓶颈。本章将介绍两个著名的框架——LightGBM和CatBoost，它们在训练速度和处理类别数据方面提供了独特的优化。

我们首先会介绍LightGBM以及它在大规模数据集上加速训练的方法，例如基于梯度的单侧采样（GOSS）和独占特征捆绑（EFB）。接着，我们会介绍CatBoost及其处理类别特征的复杂内部机制，其中包含有助于避免目标数据泄露的有序提升策略。本章最后将对XGBoost、LightGBM和CatBoost的性能特点进行直接比较，随后会提供一个实际练习，让您使用这些库实现模型。
