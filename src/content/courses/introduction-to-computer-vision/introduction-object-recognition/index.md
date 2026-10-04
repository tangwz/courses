---
course: "introduction-to-computer-vision"
sourceUrl: "https://apxml.com/zh/courses/introduction-to-computer-vision/chapter-5-introduction-object-recognition"
sourceId: 439
chapter: "introduction-object-recognition"
title: "物体识别入门"
order: 5
description: "掌握物体识别的基本思想，了解模板匹配，并理解常见难题。"
hasQuiz: true
---

在明确了图像如何进行数字化表示以及如何提取边缘和角点等基本特征之后，我们现在将重点转向理解图像*内部*的内容。本章将介绍物体识别这项基本任务：即在数字图像中识别特定物体或模式。

You will learn:
*   物体识别系统的基本目标和工作原理。
*   一种称为模板匹配的直接方法，包括它如何在较大图像中搜索已知模式来工作。
*   使用常用工具进行模板匹配的实际操作。
*   简单模板匹配在处理外观变化（例如尺度或方向改变）时的固有局限性。
*   一般物体识别任务中遇到的常见难题概览。
*   简要提及机器学习方法如何常用于应对这些难题，为更深入的学习做准备。

我们将从模板匹配开始，作为识别物体的一个直观的初步方法，并通过实现一个简单例子来加深理解。
