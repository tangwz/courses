---
course: "rnns-and-sequence-modeling"
sourceUrl: "https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-1-introduction-sequential-data"
sourceId: 605
chapter: "introduction-sequential-data"
title: "序列数据简介"
order: 1
description: "了解序列数据类型（文本、时间序列）以及为什么序列建模任务需要循环神经网络等专门模型。"
hasQuiz: true
---

我们遇到的大部分数据都具有固有的顺序。比如句子中的单词、随时间变化的股票价格，或者设备传感器读数。这种顺序有作用的信息，被称为序列数据。传统机器学习模型通常独立处理输入，当顺序中包含有价值的信息时，这种做法就不合适了。

本章为理解和处理序列数据做好准备。我们将会了解：

*   什么是序列数据，并给出文本和时间序列等具体例子。
*   定义序列的特点，比如时间依赖性和可变长度。
*   为什么传统前馈网络在序列处理任务中表现不足。
*   将原始序列数据转换为机器学习模型能理解的数值格式的方法。
*   使用序列模型处理的常见问题概览，包括预测、分类和生成。

在本章结束时，你将清晰地了解序列数据的本质以及为什么需要专门的处理方法，为后续章节中学习循环神经网络做好铺垫。
