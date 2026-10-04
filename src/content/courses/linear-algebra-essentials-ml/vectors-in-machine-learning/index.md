---
course: "linear-algebra-essentials-ml"
sourceUrl: "https://apxml.com/zh/courses/linear-algebra-essentials-ml/chapter-1-vectors-in-machine-learning"
sourceId: 460
chapter: "vectors-in-machine-learning"
title: "机器学习中的向量"
order: 1
description: "了解向量、向量运算、点积、范数及其在机器学习数据表示中的应用。"
hasQuiz: true
---

本质上，机器学习处理数据。向量提供了一种表示单个数据点或特征的基本数学结构。无论是处理图像中的像素值、文档中的词频，还是传感器测量值，将这些信息表示为向量，使我们能够一致地应用数学运算。

本章将介绍向量及其在机器学习场景中的相关性。你将学习到：

*   如何使用向量表示数据点和特征。
*   基本的向量运算：加法、减法和标量乘法，以及它们的几何意义。
*   计算向量的幅值（长度），使用不同的范数，例如 $L_1$（曼哈顿）范数和 $L_2$（欧几里得）范数：
    $$||v||_1 = \sum_i |v_i|$$
    $$||v||_2 = \sqrt{\sum_i v_i^2}$$
*   使用点积来理解向量相似性和投影。
*   测量特征空间中向量之间的距离。
*   在Python中高效地执行这些运算，使用NumPy库。

在本章结束时，你将能够熟练地使用NumPy操作向量，并理解这些运算与机器学习任务中数据处理的关系。
