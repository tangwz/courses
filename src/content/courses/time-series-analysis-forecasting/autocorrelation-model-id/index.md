---
course: "time-series-analysis-forecasting"
sourceUrl: "https://apxml.com/zh/courses/time-series-analysis-forecasting/chapter-3-autocorrelation-model-id"
sourceId: 719
chapter: "autocorrelation-model-id"
title: "自相关性与模型识别"
order: 3
description: "了解自相关函数 (ACF) 和偏自相关函数 (PACF) 及其在识别潜在时间序列模型中的用途。"
hasQuiz: true
---

在讨论了时间序列的组成部分和其平稳性后，接下来主要关注理解数据的内部相关结构。$t$ 时刻的一个值与之前时刻（例如 $t-1$、$t-2$ 等）的值有什么关系呢？回答这个问题对于选择合适的预测模型非常重要。

本章介绍分析这种时间依赖关系的主要工具：

*   **自相关函数 (ACF):** 学习它如何衡量时间序列与其滞后版本之间的相关性。
*   **偏自相关函数 (PACF):** 理解它在排除中间滞后值的影响后，如何衡量时间序列与某个滞后版本之间的相关性。
*   **绘图与解读:** 查看如何使用 Python 库生成 ACF 和 PACF 图，并学习那些提示特定模型类型（如 AR 或 MA 模型）的常见模式。

到本章结束时，你将能够计算并解读 ACF 和 PACF 图，以帮助识别适用于你的平稳时间序列数据的潜在候选模型。
