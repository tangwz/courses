---
course: "julia-for-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-3-julia-supervised-learning-algorithms"
sourceId: 1206
chapter: "julia-supervised-learning-algorithms"
title: "实现监督学习算法"
order: 3
description: "在Julia中使用MLJ.jl实现和评估监督学习模型，如线性回归、决策树和支持向量机。"
hasQuiz: false
---

监督学习是指构建模型，这些模型从配有相应正确输出的输入数据中进行学习。本章将着重介绍如何使用Julia编程语言实际实现这些算法。您将大量使用MLJ.jl，它是Julia中一个主要框架，旨在为多种机器学习工作提供统一的接口。

您将学习如何：
*   熟悉MLJ.jl生态系统，进行模型构建和评估。
*   实现和训练线性模型，例如用于预测连续值的线性回归和用于分类任务的逻辑回归。
*   构建决策树模型并理解像随机森林这样的集成方法。
*   使用支持向量机（SVM）解决分类问题。
*   使用准确率、精确率、召回率和 $F_1$ 分数等标准评估指标来衡量模型性能。
*   运用交叉验证技术进行更可靠的模型评估，并在MLJ.jl中进行超参数调优以优化您的模型。

到本章结束时，您将能够在Julia中选择、实现、训练和评估几种常见的监督学习模型。
