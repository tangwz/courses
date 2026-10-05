# 第 1 章：集成学习与提升的基本原理

来源：[原章节](https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-1-ensemble-learning-and-boosting-foundations)

[返回课程目录](../README.md)

要构建有效的梯度提升模型，最好先了解集成学习的一般原理。单个预测模型，比如决策树，容易出现高方差或高偏差。通过结合多个模型的预测结果，我们通常能得到更准确、泛化能力更好的结果。本章将介绍结合模型的方法。

您将首先学习什么是集成方法以及它如何运作。接下来我们将比较两种常见策略：Bagging，其中模型是独立并行构建的；以及Boosting，其中模型是按顺序构建的，每个新模型都试图纠正之前模型所犯的错误。这将引出对提升原理的介绍，并查看AdaBoost算法，该算法直接为我们之后将学习的梯度提升方法做铺垫。

最后，我们将定义充当这些集成模型组成部分的“弱学习器”，并回顾这些结合方法如何影响偏差-方差权衡。这里介绍的内容为理解下一章中梯度提升机的运作方式提供了必要的背景知识。

## 小节

- 1. [什么是集成方法？](01-%E4%BB%80%E4%B9%88%E6%98%AF%E9%9B%86%E6%88%90%E6%96%B9%E6%B3%95%EF%BC%9F.md)
- 2. [Bagging 与 Boosting](02-Bagging%20%E4%B8%8E%20Boosting.md)
- 3. [提升（Boosting）原理介绍](03-%E6%8F%90%E5%8D%87%EF%BC%88Boosting%EF%BC%89%E5%8E%9F%E7%90%86%E4%BB%8B%E7%BB%8D.md)
- 4. [AdaBoost算法：梯度提升算法的前身](04-AdaBoost%E7%AE%97%E6%B3%95%EF%BC%9A%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%E7%AE%97%E6%B3%95%E7%9A%84%E5%89%8D%E8%BA%AB.md)
- 5. [了解弱学习器](05-%E4%BA%86%E8%A7%A3%E5%BC%B1%E5%AD%A6%E4%B9%A0%E5%99%A8.md)
- 6. [集成学习中的偏差-方差权衡](06-%E9%9B%86%E6%88%90%E5%AD%A6%E4%B9%A0%E4%B8%AD%E7%9A%84%E5%81%8F%E5%B7%AE-%E6%96%B9%E5%B7%AE%E6%9D%83%E8%A1%A1.md)
