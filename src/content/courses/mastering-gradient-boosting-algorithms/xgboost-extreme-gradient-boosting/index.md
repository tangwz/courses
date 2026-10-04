---
course: "mastering-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-4-xgboost-extreme-gradient-boosting"
sourceId: 535
chapter: "xgboost-extreme-gradient-boosting"
title: "XGBoost：极限梯度提升"
order: 4
description: "对XGBoost进行详细分析，涵盖其正则化目标、分枝寻找算法和系统优化。"
hasQuiz: false
---

本章介绍XGBoost（极限梯度提升），直接基于前面讨论的梯度提升基本思想和正则化方法。XGBoost是一种常用且有效的梯度提升实现方式，以多项重要改进为特点，旨在提升性能和准确性。

我们将分析使XGBoost高效的核心构成部分：
*   **正则化学习目标**：了解XGBoost如何将$L_1$和$L_2$惩罚项直接纳入目标函数中，以控制模型复杂度。
*   **分枝寻找算法**：学习寻找分枝的精确贪心算法，以及处理大规模数据集的近似算法。
*   **稀疏性处理**：了解其内置机制，以高效处理缺失值。
*   **系统优化**：回顾并行处理和缓存优化等可提高训练速度的方法。

在本章结束时，您将掌握XGBoost背后的技术细节，并准备好使用其Python库进行实现，为实际应用配置其主要参数。
