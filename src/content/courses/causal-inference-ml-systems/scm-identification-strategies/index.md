---
course: "causal-inference-ml-systems"
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-1-scm-identification-strategies"
sourceId: 844
chapter: "scm-identification-strategies"
title: "结构因果模型与识别策略"
order: 1
description: "回顾结构因果模型 (SCMs) 以及用于因果效应的不限于常规准则的进阶识别技术。"
hasQuiz: false
---

为了估计因果效应并构建因果感知机器学习系统，需要一种精确的语言来表示因果假设并推导出识别策略。本章将提供这些内容。

我们首先重新回顾结构因果模型 (SCMs)，澄清支撑大部分因果推断的表示法和假设。你将使用进阶的图形表示，包括复杂的有向无环图 (DAGs) 和替代模型，来描绘复杂的因果关系。

本章的核心内容集中在因果识别上：判断一个因果效应，例如干预 $P(Y | do(X=x))$ 的平均效应，是否可以从观测数据中计算出来。你将学习为此应用 Pearl'的 do-calculus 的形式规则，并研究在标准后门或前门准则不足时处理这些情况的识别策略。我们还将处理因果图中存在的循环和反馈回路等复杂情况。

最后，你将学习敏感性分析技术，以评估识别假设的违反情况可能如何影响你的结论。本章最后包含实践练习，将这些识别原则应用于有难度的因果问题。
