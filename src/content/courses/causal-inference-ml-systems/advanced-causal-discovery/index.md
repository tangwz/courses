---
course: "causal-inference-ml-systems"
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-2-advanced-causal-discovery"
sourceId: 848
chapter: "advanced-causal-discovery"
title: "高级因果发现方法"
order: 2
description: "实现和评估从不同数据类型中发现因果结构的高级算法。"
hasQuiz: false
---

在第一章学习到的结构因果模型（SCMs）和识别方法的基础上，本章主要介绍直接从数据中学习因果结构的方法。从观测数据中推断出正确的因果图（通常为有向无环图，即DAG），会带来重大挑战。

在本章中，你将研究已有和现代的因果发现算法。我们将涵盖：

*   **基于约束的方法：** 诸如PC和FCI算法（及其扩展），这些方法使用条件独立性的统计检验，例如 $X \perp Y | Z$，以推断结构属性。
*   **基于评分的方法：** 诸如GES和LiNGAM，这些方法寻找使定义的评分标准最大化的图结构。
*   **使用多样化数据：** 整合干预数据和处理异构数据集的方法，以提高发现的准确性。
*   **实际考量：** 评估发现算法性能，处理潜在变量的存在，以及使用Python库实现这些方法。

完成本章后，你将掌握在实际场景中应用和评估不同因果发现算法所需的知识。
