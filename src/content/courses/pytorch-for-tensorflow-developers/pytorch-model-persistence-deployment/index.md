---
course: "pytorch-for-tensorflow-developers"
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment"
sourceId: 1047
chapter: "pytorch-model-persistence-deployment"
title: "保存、加载和部署模型"
order: 5
description: "学习PyTorch模型持久化（保存/加载）以及部署介绍，并与TensorFlow做法进行比较。"
hasQuiz: true
---

成功训练模型后，接下来的步骤是保存您的成果并准备将其用于应用程序。本章侧重于PyTorch框架中这些必要的训练后操作，同时与您可能在TensorFlow中了解到的做法进行比较。

您将学习：
*   管理模型持久化，使用PyTorch的`state_dict`，并与TensorFlow的SavedModel和HDF5格式进行对比。
*   区分保存整个模型与仅保存其参数（$权重$和$偏差$），并了解检查点策略。
*   使用TorchScript为模型部署做准备，它有助于PyTorch模型的序列化和优化。
*   了解ONNX（开放神经网络交换），以实现不同框架间的模型互操作性。
*   简要介绍TorchServe，用于PyTorch模型服务。

完成本章后，您将能够有效地保存、加载、检查和准备您的PyTorch模型以应对各种部署场景，并将您现有的TensorFlow知识调整到PyTorch生态系统中。
