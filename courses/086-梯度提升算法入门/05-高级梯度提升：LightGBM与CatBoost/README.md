# 第 5 章：高级梯度提升：LightGBM与CatBoost

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-5-advanced-gradient-boosting-lightgbm-catboost)

[返回课程目录](../README.md)

XGBoost相较于标准梯度提升算法有了显著提升，但也有其他专业库随之出现，旨在解决特定的性能瓶颈。本章将介绍两个著名的框架——LightGBM和CatBoost，它们在训练速度和处理类别数据方面提供了独特的优化。

我们首先会介绍LightGBM以及它在大规模数据集上加速训练的方法，例如基于梯度的单侧采样（GOSS）和独占特征捆绑（EFB）。接着，我们会介绍CatBoost及其处理类别特征的复杂内部机制，其中包含有助于避免目标数据泄露的有序提升策略。本章最后将对XGBoost、LightGBM和CatBoost的性能特点进行直接比较，随后会提供一个实际练习，让您使用这些库实现模型。

## 小节

- 1. [LightGBM 介绍：基于梯度的单边采样](01-LightGBM%20%E4%BB%8B%E7%BB%8D%EF%BC%9A%E5%9F%BA%E4%BA%8E%E6%A2%AF%E5%BA%A6%E7%9A%84%E5%8D%95%E8%BE%B9%E9%87%87%E6%A0%B7.md)
- 2. [LightGBM 的独占特征捆绑](02-LightGBM%20%E7%9A%84%E7%8B%AC%E5%8D%A0%E7%89%B9%E5%BE%81%E6%8D%86%E7%BB%91.md)
- 3. [CatBoost 简介：处理类别特征](03-CatBoost%20%E7%AE%80%E4%BB%8B%EF%BC%9A%E5%A4%84%E7%90%86%E7%B1%BB%E5%88%AB%E7%89%B9%E5%BE%81.md)
- 4. [CatBoost的有序提升和对称树](04-CatBoost%E7%9A%84%E6%9C%89%E5%BA%8F%E6%8F%90%E5%8D%87%E5%92%8C%E5%AF%B9%E7%A7%B0%E6%A0%91.md)
- 5. [性能比较：XGBoost、LightGBM 与 CatBoost](05-%E6%80%A7%E8%83%BD%E6%AF%94%E8%BE%83%EF%BC%9AXGBoost%E3%80%81LightGBM%20%E4%B8%8E%20CatBoost.md)
- 6. [动手实践：实现 LightGBM 和 CatBoost](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%20LightGBM%20%E5%92%8C%20CatBoost.md)
