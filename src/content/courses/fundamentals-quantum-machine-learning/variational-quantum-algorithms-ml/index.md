---
course: "fundamentals-quantum-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-4-variational-quantum-algorithms-ml"
sourceId: 377
chapter: "variational-quantum-algorithms-ml"
title: "变分量子算法在机器学习中的应用"
order: 4
description: "学习用于机器学习的变分量子算法（VQA），包括PQC设计、成本函数、梯度、优化以及贫瘠高原现象。"
hasQuiz: false
---

本章介绍变分量子算法（VQA），这是一种混合量子-经典框架，通常被认为适合近期量子处理器使用。我们将从这些方法所依据的变分原理开始，并接着讲解为机器学习任务构建VQA的实际要素。

您将学到：

*   设计高效的参数化量子电路（PQC），也称为ansätze。
*   基于量子测量结果建立成本函数，并针对分类或回归等机器学习问题进行调整。
*   计算量子电路期望值的梯度，侧重于参数移位规则等技术。
*   应用高级经典优化算法（例如Adam、SPSA）和量子感知方法，例如量子自然梯度（$QNG$），来训练VQA。
*   理解由贫瘠高原现象带来的训练挑战，以及减轻其影响的策略。

本章包含变分量子分类器（VQC）的实际操作实现，结合了PQC设计、成本函数定义和优化等思想。
