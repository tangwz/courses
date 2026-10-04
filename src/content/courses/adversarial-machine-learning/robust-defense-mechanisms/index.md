---
course: "adversarial-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/adversarial-machine-learning/chapter-5-robust-defense-mechanisms"
sourceId: 823
chapter: "robust-defense-mechanisms"
title: "强大的防御机制"
order: 5
description: "涵盖高级防御方法：对抗训练、可证明防御、输入转换和梯度遮蔽。"
hasQuiz: false
---

在详细阐述了攻击者如何通过规避、投毒和推断攻击等方式攻击机器学习系统的弱点后，我们现在将重点转向构建能够抵御此类操作的模型。本章将介绍提升模型安全性的实用技术。

您将学习对抗训练，这是一种重要方法，它将对抗样本（例如通过投影梯度下降（PGD）生成的样本）直接融入训练过程。这通常被表述为一个最小最大优化问题，如下所示：$$ \min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in S} L(\theta, x+\delta, y) \right] $$。我们将研究可证明防御，例如随机平滑，它们在定义的扰动半径 $r$ 内为模型的抵御能力提供数学保证。此外，您还将分析输入转换技术，了解混淆梯度可能带来的虚假安全感问题，并研究专门设计用于对抗数据投毒和后门威胁的策略。实践环节将指导您实现对抗训练等主要防御机制。
