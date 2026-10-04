---
course: "getting-started-with-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-6-saving-loading-models"
sourceId: 508
chapter: "saving-loading-models"
title: "保存和加载模型"
order: 6
description: "学习保存和加载TensorFlow模型权重及整个模型架构的不同方法。"
hasQuiz: true
---

在投入时间和计算资源训练机器学习模型后，您会希望保存其学习到的状态。保存模型可以让您稍后停止并恢复训练，无需重新训练即可用于对新数据进行预测，或者与他人分享。本章侧重于TensorFlow和Keras中模型保存的实用方法。

您将学习如何：
*   在训练期间使用检查点定期保存模型权重。
*   仅保存模型学习到的参数（权重）。
*   保存完整的模型，包括其架构、权重和优化器配置。
*   了解标准的TensorFlow `SavedModel` 格式，该格式适用于部署。
*   加载以前保存的权重或整个模型，用于推断或继续训练。
*   了解TensorFlow Hub以获取预训练模型组件。

我们将介绍可用的不同方法，并讨论何时使用每种方法，以确保您能有效管理训练好的模型。
