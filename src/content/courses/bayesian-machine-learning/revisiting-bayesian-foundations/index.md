---
course: "bayesian-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-1-revisiting-bayesian-foundations"
sourceId: 732
chapter: "revisiting-bayesian-foundations"
title: "重审贝叶斯根本"
order: 1
description: "从进阶视角温习贝叶斯核心理念，例如贝叶斯定理、先验、后验和似然。"
hasQuiz: false
---

在构建复杂的贝叶斯模型之前，我们必须巩固对基本原理的理解。本章将重新审视贝叶斯推断的根本，确保我们具备处理后续主题所需的、共通的进阶视角。尽管您可能已经接触过贝叶斯理念，但我们将重新审视它们，考虑到复杂机器学习应用的需要。

我们首先回顾概率建模原则，并分析贝叶斯定理，特别是其在高维环境中的应用和影响。回忆该定理：
$$
P(\theta | \mathcal{D}) = \frac{P(\mathcal{D} | \theta) P(\theta)}{P(\mathcal{D})}
$$
其中，$P(\theta | \mathcal{D})$ 表示给定数据 $\mathcal{D}$ 时参数 $\theta$ 的后验概率，$P(\mathcal{D} | \theta)$ 是似然，$P(\theta)$ 是先验概率，而 $P(\mathcal{D})$ 是模型证据或边际似然。

我们将讨论先验选择的考量及其影响，并比较主观和客观的方法。贝叶斯推断与信息理论之间的关联性将被审视，通过熵和Kullback-Leibler (KL)散度等度量进行，通常写为 $D_{KL}(P||Q)$。此外，我们还会识别应用贝叶斯方法时面临的常见计算难题，例如难以计算的积分，并介绍检查和评估这些模型充分性的进阶技术。本次回顾为后续介绍的进阶推断算法和模型类型做好准备。
