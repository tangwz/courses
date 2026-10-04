---
course: "introduction-autoencoders-feature-learning"
sourceUrl: "https://apxml.com/zh/courses/introduction-autoencoders-feature-learning/chapter-4-autoencoders-learning-features"
sourceId: 1150
chapter: "autoencoders-learning-features"
title: "自编码器与特征学习"
order: 4
description: "了解自编码器如何进行特征学习和降维，从数据中提取有意义的表示形式。"
hasQuiz: true
---

在确定了自编码器的架构及其训练方法后，我们现在将重点放在一项主要应用上：它们如何从数据中学习。将输入数据压缩到瓶颈层形成一个更小、更密集的表示，然后对其进行重建，这个过程迫使网络识别并保存输入数据中最显著的特征。

本章将考察这种能力。我们将考察自编码器如何作为特征学习工具，自动从原始数据中识别并提取有用属性，而无需针对这些特定特征进行显式编程。你将理解瓶颈层中的数据如何被视为一组学习到的特征。我们还将介绍这如何直接实现降维，这是一种通过减少变量数量来简化数据集的方法，同时力求保留主要信息。讨论内容将包括对这些自动学习到的特征与传统手动特征工程所得特征的比较，以及观察或显示自编码器所学特征的方式。
