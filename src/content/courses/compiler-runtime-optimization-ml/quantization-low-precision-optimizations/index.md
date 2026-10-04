---
course: "compiler-runtime-optimization-ml"
sourceUrl: "https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations"
sourceId: 884
chapter: "quantization-low-precision-optimizations"
title: "量化与低精度优化"
order: 8
description: "实现模型量化、低精度内核生成和混合精度计算的编译器技术。"
hasQuiz: false
---

运行机器学习模型通常需要大量的计算资源和内存带宽。降低模型权重和激活值的数值精度，可以实现更低的延迟、更小的内存占用和更低的功耗。这种性能提升可能会以牺牲一定精度为代价，因此需要通过专门的方法进行细致的管理。

本章主要关注编译器和运行时策略，以便使用低精度算术（主要是8位整数$INT8$和新兴的低精度浮点格式如$FP8$）来实现和优化模型。我们将考察量化原理，包括映射方案和校准，如何在编译器中间表示（IRs）中体现。您将了解支持量化感知训练（QAT）和训练后量化（PTQ）的编译器流程，包括量化和反量化操作的插入与优化。我们将介绍如何生成借助硬件特定低精度指令的优化内核，并讨论有效处理混合精度计算的方法。目的是理解编译器如何使得低精度技术得以有效应用，从而实现高效的模型部署。
