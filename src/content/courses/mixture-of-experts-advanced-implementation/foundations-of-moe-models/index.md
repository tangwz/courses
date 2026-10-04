---
course: "mixture-of-experts-advanced-implementation"
sourceUrl: "https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-1-foundations-of-moe-models"
sourceId: 1232
chapter: "foundations-of-moe-models"
title: "专家混合模型的核心原理"
order: 1
description: "回顾稀疏门控MoE模型的核心架构，包括门控网络、专家专业化和负载均衡损失。"
hasQuiz: false
---

传统上，扩展深度学习模型需要让每个组成部分都变大，这会大大增加计算成本。专家混合 (MoE) 模型提出了一种不同的方法。它们通过引入一组被称为“专家”的专门化子网络来提升模型容量，并动态地将每个输入标记路由到这些专家中的一小部分。这使得参数数量可以大幅增加，同时单次前向传播所需的计算量不会按比例增加。

本章确立了这些稀疏架构的核心原理。我们将首先分析其主要组成部分：学习如何路由输入的**门控网络**，以及发展出专门功能的**专家网络**。然后，我们将分析MoE层的完整数学公式。对于输入 $x$，其输出 $y(x)$ 是专家输出的稀疏加权和：

$$
y(x) = \sum_{i=1}^{N} g(x)_i E_i(x)
$$

在此公式中，$g(x)_i$ 是专家 $i$ 的门控值，$E_i(x)$ 是该专家的输出。你将学会如何实现稀疏性，通常做法是确保对于任何给定输入，大多数 $g(x)_i$ 值都为零。

训练这些模型有一些独特的考量。我们将介绍用于负载均衡的辅助损失函数，这些函数促使标记在所有专家之间均匀分布。这是防止常见训练不稳定现象（如专家崩溃，即少数专家被过度使用而其他专家没有得到训练）的必要方法。最后，本章将通过从头构建一个基本的MoE层来运用这些思想。
