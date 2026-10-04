---
course: "advanced-tensorflow"
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies"
sourceId: 648
chapter: "distributed-training-strategies"
title: "使用分布式策略扩展训练"
order: 3
description: "学习如何使用TensorFlow的分布式策略，在多个GPU和节点上扩展模型训练。"
hasQuiz: false
---

随着机器学习模型规模增大，数据集也变得越来越大，在单个设备（CPU或GPU）上进行训练常常变得不切实际，因为训练时间过长或内存限制。将训练扩展到多个设备或机器上常常是必要的，以便高效处理这些要求高的工作负载。

本章将介绍如何分发TensorFlow训练任务的方法。您将了解分布式机器学习的核心原理以及TensorFlow的`tf.distribute.Strategy` API，这个API能简化此过程。我们会讲解针对不同硬件配置的特定策略：

*   `MirroredStrategy` 用于在单台机器上的多个GPU上进行训练。
*   `MultiWorkerMirroredStrategy` 用于在多台机器间进行同步训练。
*   `ParameterServerStrategy`背后的思想，适用于异步方法。
*   `TPUStrategy` 用于使用Google的张量处理单元（TPU）。

此外，您还将学习管理数据并行性的技术和调试分布式训练配置的方法。完成本章的学习将使您具备能力，来加速大型模型的训练，使用TensorFlow的分布式功能。
