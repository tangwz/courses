---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-6-gated-recurrent-units-gru"
sourceId: 613
chapter: "gated-recurrent-units-gru"
title: "门控循环单元 (GRUs)"
order: 6
description: "了解门控循环单元 (GRUs)，这是一种更精简、参数更少的 LSTM 替代方案。"
hasQuiz: true
---

虽然长短期记忆 (LSTM) 网络提供了一种使用多个门控来获取长期依赖关系的有效方法，但门控循环单元 (GRUs) 提供了一种相关且通常更简单的替代方案。

本章介绍 GRU 架构。我们将查看它的组成部分，特别是**更新门** ($z_t$) 和**重置门** ($r_t$)，并了解它们如何共同作用以控制信息流动并更新隐藏状态。我们将了解候选隐藏状态是如何计算并与先前的隐藏状态结合的。

我们还将直接比较 GRU 和 LSTM，讨论它们的结构差异、相对计算效率，并为特定序列建模任务选择合适的门控单元提供实用指导。
