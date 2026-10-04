---
course: "basics-ml-deployment"
sourceUrl: "https://apxml.com/zh/courses/basics-ml-deployment/chapter-2-preparing-model-deployment"
sourceId: 795
chapter: "preparing-model-deployment"
title: "为模型部署做准备"
order: 2
description: "学习保存训练好的模型及其相关组件的方法，以便之后加载并用于预测。"
hasQuiz: true
---

训练完机器学习模型后，当务之急是保存其学到的状态。不保存的话，模型只存在于内存中，一旦训练脚本结束就会丢失。本章将讲述如何让模型持久保存，并为在不同环境或稍后使用做好准备。

你将了解模型序列化是什么，它是将训练好的模型对象转换为可存储（如文件）的格式，以便之后加载回内存的过程。我们将介绍为此目的的两个常用Python库：`pickle` 和 `joblib`，说明它们的用法以及何时更适合使用其中一个而非另一个。

此外，准备模型不仅仅是保存算法的权重。我们将讨论追踪和管理模型所依赖的软件库（及其版本）的必要性。你还将学习如何保存数据预处理步骤（如缩放器或编码器），这些步骤必须在新输入数据进行预测前保持一致地应用。本章包含实际示例，引导你完成简单模型的保存和加载。
