---
course: "monitoring-managing-ml-models-production"
sourceUrl: "https://apxml.com/zh/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection"
sourceId: 700
chapter: "advanced-drift-detection"
title: "高级漂移检测技术"
order: 2
description: "学习生产机器学习系统中检测数据漂移和概念漂移的精密方法。"
hasQuiz: false
---

在为机器学习系统监控打下基础之后，我们现在关注一种重要的故障模式：漂移。用历史数据训练的模型在遇到随时间变化的新数据时，可能会失去效用。对单个特征进行简单的统计检查，往往不足以识别生产环境中不易察觉或复杂的变动。

本章介绍更精密的检测这些变化的技术。您将学到：

*   基本漂移检测方法的局限性。
*   检测**多变量数据漂移**的方法，包括考虑特征之间的关系。
*   **序列分析**技术，用于可能更快地检测到分布变化。
*   专门用于识别**概念漂移**的策略，即输入和目标变量之间的关系发生变化的情况。
*   使用**对抗性验证**来量化分布差异。
*   监控非结构化数据（如**嵌入**）中漂移的方法。
*   如何设计和实现针对特定需求定制的**自定义漂移检测逻辑**。

我们将了解这些方法背后的理论，并提供实践指导，最后通过一个动手实践练习来实现多变量漂移检测机制。
