---
course: "intro-to-multimodal-ai"
sourceUrl: "https://apxml.com/zh/courses/intro-to-multimodal-ai/chapter-3-techniques-integrating-modalities"
sourceId: 1161
chapter: "techniques-integrating-modalities"
title: "模态整合方法"
order: 3
description: "了解结合来自不同模态数据的主要技术，包括合并策略和表示学习。"
hasQuiz: true
---

在介绍了文本、图像和音频等单一数据模态如何表示以供AI处理之后，本章将讨论组合这些多样化信息的方法。我们将审视使AI系统能够结合这些不同来源数据的主要技术。

本章内容包括：
*   **多模态合并策略**：概述早期、中期和晚期合并。您将了解到数据流如何以及在哪个阶段进行合并。例如，早期合并可能涉及连接特征向量 $v_{\text{image}}$ 和 $v_{\text{text}}$：
    $$ v_{\text{fused}} = \text{concat}(v_{\text{image}}, v_{\text{text}}) $$
*   **共享表示**：将来自不同模态的信息投影到共同特征空间的方法。目标通常是找到一个空间 $S$，在此空间中，表示 $s_i \in S$ 在不同模态之间具有可比性。
*   **协调表示**：侧重于学习不同模态空间之间的映射或关联，而不是必须创建一个单一共享空间的方法。
*   **基本架构**：常用神经网络架构简介。
*   **注意力机制**：简要了解注意力，这项技术允许模型选择性地衡量来自不同模态的输入数据不同部分的重要性。

在本章结束时，您将对多模态系统如何整合来自独立渠道的信息有扎实的理解，从而为解释和决策提供更全面的依据。
