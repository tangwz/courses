---
course: "getting-started-with-gradient-boosting-algorithms"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations"
sourceId: 1310
chapter: "ensemble-learning-and-boosting-foundations"
title: "集成学习与提升的基本原理"
order: 1
description: "了解集成机器学习的原理。学习Bagging和Boosting的区别，以及AdaBoost算法的运作方式。"
hasQuiz: false
---

要构建有效的梯度提升模型，最好先了解集成学习的一般原理。单个预测模型，比如决策树，容易出现高方差或高偏差。通过结合多个模型的预测结果，我们通常能得到更准确、泛化能力更好的结果。本章将介绍结合模型的方法。

您将首先学习什么是集成方法以及它如何运作。接下来我们将比较两种常见策略：Bagging，其中模型是独立并行构建的；以及Boosting，其中模型是按顺序构建的，每个新模型都试图纠正之前模型所犯的错误。这将引出对提升原理的介绍，并查看AdaBoost算法，该算法直接为我们之后将学习的梯度提升方法做铺垫。

最后，我们将定义充当这些集成模型组成部分的“弱学习器”，并回顾这些结合方法如何影响偏差-方差权衡。这里介绍的内容为理解下一章中梯度提升机的运作方式提供了必要的背景知识。
