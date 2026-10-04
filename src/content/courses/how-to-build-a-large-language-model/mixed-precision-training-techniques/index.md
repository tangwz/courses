---
course: "how-to-build-a-large-language-model"
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-20-mixed-precision-training-techniques"
sourceId: 1098
chapter: "mixed-precision-training-techniques"
title: "混合精度训练方法"
order: 20
description: "运用$FP16$和$BF16$混合精度训练来加快训练速度、减少内存占用，同时保持训练的稳定性。"
hasQuiz: false
---

训练大型语言模型对现有硬件造成较大压力，这主要是因为进行标准32位浮点($FP32$)运算时会产生大量的内存占用和计算开销。混合精度训练提供了一种缓解这些制约的方法。通过有策略地运用低精度数值格式，特别是16位浮点($FP16$)和bfloat16 ($BF16$)，我们能够大幅减少内存占用，并且通常能加快计算速度，尤其是在支持NVIDIA Tensor Cores等专用硬件上。

本章将详细介绍大型语言模型混合精度训练的应用方法。我们将考察$FP16$和$BF16$格式的特点，并讨论与$FP32$相比的权衡取舍。我们将讨论使用低精度时遇到的常见稳定性问题，特别是$FP16$的有限范围，并介绍_损失缩放_等方法，以在训练过程中保持数值稳定性。$BF16$格式的作用和好处也将予以讨论。最后，我们将考察如何运用主流深度学习框架中内置的自动混合精度（AMP）支持，以简化实现过程。
