---
course: "bayesian-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/bayesian-machine-learning/chapter-3-variational-inference-techniques"
sourceId: 738
chapter: "variational-inference-techniques"
title: "变分推断方法"
order: 3
description: "了解近似后验分布的高级变分推断方法，包括SVI和BBVI。"
hasQuiz: false
---

计算后验分布 $p(\mathbf{z}|\mathbf{x})$ 通常具有很大的计算难度，原因在于证据 $p(\mathbf{x})$ 需要进行难以求解的积分。尽管之前讨论过的马尔可夫链蒙特卡洛方法提供了一种从后验分布采样的途径，但这些方法可能需要大量的计算资源，尤其是在处理大型数据集或复杂模型时。

本章将介绍变分推断（VI），这是一类用于近似后验分布的备选方法。变分推断并非通过采样，而是将贝叶斯推断重新构造为一个优化问题。目标是从一个易于处理的分布族中找到一个分布 $q(\mathbf{z})$，使其与真实的后验分布 $p(\mathbf{z}|\mathbf{x})$ 最为接近。这通常通过最大化模型证据的一个下界来实现，该下界被称为证据下界（ELBO）：
$$ \mathcal{L}(q) = \mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z})] - \mathbb{E}_{q(\mathbf{z})} [\log q(\mathbf{z})] $$

我们将从将推断表述为优化问题开始，并推导出ELBO。您将了解到常见的平均场近似以及坐标上升变分推断（CAVI）算法。随后，我们将介绍使用随机变分推断（SVI）将推断扩展到大型数据集的方法，以及如何使用黑盒变分推断（BBVI）来处理具有复杂梯度的模型。我们还将考察表达能力更强的变分族，并分析变分推断与马尔可夫链蒙特卡洛方法在速度、准确性和可扩展性方面的优缺点。通过使用常用库进行实践操作，将有助于您更好地掌握这些内容。
