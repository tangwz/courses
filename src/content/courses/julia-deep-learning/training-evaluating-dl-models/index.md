---
course: "julia-deep-learning"
sourceUrl: "https://apxml.com/zh/courses/julia-deep-learning/chapter-4-training-evaluating-dl-models"
sourceId: 1213
chapter: "training-evaluating-dl-models"
title: "训练与评估深度学习模型"
order: 4
description: "掌握使用 Flux.jl 在 Julia 中训练深度学习模型的过程，包括评估、正则化和超参数调整。"
hasQuiz: false
---

在学习了如何使用 Flux.jl 构建各种神经网络结构之后，下一步是让这些模型发挥作用。本章着重讲解训练深度学习模型的实际操作，并系统地评估它们的表现。

您将学习实现并理解模型训练循环的组成部分，从分批处理数据和迭代周期，到计算损失和使用优化器更新模型参数。我们将讲解如何使用回调函数来有效监控训练过程，并查看准确率、$F_1$-score（用于分类）和均方误差（$MSE$）（用于回归）等常见衡量标准，以评估模型的效用。

此外，本章将引导您掌握提高模型泛化能力的方法，例如包括 dropout 和权重衰减（$L_2$ 正则化）在内的正则化方法。您还将讨论超参数调整的策略，以找到最佳的模型配置。最后，我们将讨论可视化训练进度和调试训练过程中遇到的常见问题的方法，最后通过一个实践练习来巩固这些知识点。
