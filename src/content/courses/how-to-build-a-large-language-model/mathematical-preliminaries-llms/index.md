---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-2-mathematical-preliminaries-llms"
sourceId: 1080
chapter: "mathematical-preliminaries-llms"
title: "大型语言模型的数学预备知识"
order: 2
description: "回顾理解大型语言模型架构和训练所需的线性代数、微积分和概率内容。"
hasQuiz: false
---

构建大型语言模型需要扎实掌握其背后的数学原理。尽管深度学习框架处理许多底层计算，但理解这些数学对于设计高效架构、理解训练过程、调试诸如梯度爆炸 ($ \nabla L \to \infty $) 等问题以及实现自定义组件都很有帮助。

本章将回顾构成本课程所讨论技术支撑的主要数学内容。我们将回顾：

*   **线性代数：** 重点在于向量、矩阵和运算，它们对神经网络中数据表示和转换非常重要。
*   **微积分：** 涵盖导数、梯度（例如 $ \nabla f(x) $）和链式法则，这些对于理解反向传播和梯度下降等优化算法是不可或缺的：
    $$ \theta_{t+1} = \theta_t - \eta \nabla_{\theta} J(\theta) $$
*   **概率与统计：** 讨论诸如概率分布、熵和采样等内容，它们在语言建模和评估中扮演重要角色。
*   **数值稳定性：** 简要提及在实现深度学习模型时遇到的实际问题。

我们还将确立后续章节中使用的数学符号，以确保清晰。本次回顾旨在重温这些知识，为全面理解大型语言模型构建和训练的技术细节提供必要的背景。
