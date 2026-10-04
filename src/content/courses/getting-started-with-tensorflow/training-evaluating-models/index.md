---
course: "getting-started-with-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-tensorflow/chapter-4-training-evaluating-models"
sourceId: 500
chapter: "training-evaluating-models"
title: "模型的训练与评估"
order: 4
description: "使用TensorFlow内置功能和回调函数编译、训练并评估您的Keras模型。"
hasQuiz: true
---

既然您已经使用Keras API构建了模型，本章将讲解训练模型并评估其性能的重要步骤。首先，我们将通过`model.compile()`配置学习过程，您将选择合适的损失函数（如回归任务的均方误差 $MSE$ 或分类任务的交叉熵），选择指导学习的优化器（如Adam或$SGD$），并定义用于观察进度的指标（如准确率）。

接着，您将学习如何使用`model.fit()`将数据喂入模型并执行训练循环，理解批次（epochs）和批处理大小（batch size）等参数。随后，我们将介绍使用`model.evaluate()`评估训练好的模型在测试数据上的效果，并使用`model.predict()`对新输入生成预测。最后，我们将介绍Keras回调函数，包括用于保存训练进度的`ModelCheckpoint`，用于防止过拟合的`EarlyStopping`，以及整合TensorBoard以展示训练指标和模型图。本章最后将通过结合这些技术的实践练习进行总结。
