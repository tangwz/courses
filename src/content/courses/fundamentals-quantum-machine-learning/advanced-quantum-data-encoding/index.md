---
course: "fundamentals-quantum-machine-learning"
sourceUrl: "https://apxml.com/zh/courses/fundamentals-quantum-machine-learning/chapter-2-advanced-quantum-data-encoding"
sourceId: 375
chapter: "advanced-quantum-data-encoding"
title: "高级量子数据编码与特征映射"
order: 2
description: "学习高级量子数据编码方法以及QML中量子特征映射的属性。"
hasQuiz: false
---

要使用量子计算机进行机器学习，经典数据首先需要以量子形式表示。此过程被称为量子数据编码，对量子机器学习（QML）来说非常重要。我们将经典数据点（例如向量 $x$）转换为量子态（通常表示为存在于希尔伯特空间中的 $|\phi(x)\rangle$）的方式，很大程度上影响着后续量子算法的能力和表现。

本章主要讲解将经典信息编码到量子系统中的方法及其作用。我们将分析量子特征映射的数学结构，这些结构定义了经典到量子的数据转换。您将了解到各种编码策略，包括高阶多项式映射和旨在增强模型表达能力的数据重上传技术。我们还将介绍如何分析这些映射的属性，例如它们产生纠缠的能力以及它们与核方法的几何关系。此外，我们会应对编码高维数据的实际难题，并提供使用标准量子计算库实现自定义特征映射的实践指导。在本章结束时，您将牢固掌握高级数据编码技术及其理论依据，让您能够为特定的QML任务设计和选择合适的特征映射。
