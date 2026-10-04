---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities"
sourceId: 1102
chapter: "identifying-mitigating-training-instabilities"
title: "识别与应对训练不稳定现象"
order: 24
description: "诊断并应对常见的训练不稳定情况，如损失飙升、发散以及数值溢出/下溢。"
hasQuiz: false
---

训练大型语言模型时间一长，常会遇到一些意料之外的问题。原本看似稳定的训练过程可能突然发散，产生`NaN`（非数值）损失等无效输出，或者损失函数突然急剧升高。这些情况会阻碍训练进展，并消耗大量计算资源。

本章主要讲解如何应对这些训练挑战。你将学会通过监测损失值和梯度范数($||\nabla L||$)等重要指标，来识别常见的不稳定迹象。我们将会介绍诊断此类问题根源的方法，比如损失飙升和数值精度错误，特别是在使用混合精度格式（$FP16$或$BF16$）时。我们还会回顾稳定化方法，包括梯度裁剪和适当的学习率调整，并考虑模型架构选择如何影响训练的稳定表现。学完本章，你将能更好地预判、诊断并处理大规模训练中出现的不稳定情况。
