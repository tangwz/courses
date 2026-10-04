# 第 3 章：使用分布式策略扩展训练

来源：[原章节](https://apxml.com/zh/courses/advanced-tensorflow/chapter-3-distributed-training-strategies)

[返回课程目录](../README.md)

随着机器学习模型规模增大，数据集也变得越来越大，在单个设备（CPU或GPU）上进行训练常常变得不切实际，因为训练时间过长或内存限制。将训练扩展到多个设备或机器上常常是必要的，以便高效处理这些要求高的工作负载。

本章将介绍如何分发TensorFlow训练任务的方法。您将了解分布式机器学习的核心原理以及TensorFlow的`tf.distribute.Strategy` API，这个API能简化此过程。我们会讲解针对不同硬件配置的特定策略：

*   `MirroredStrategy` 用于在单台机器上的多个GPU上进行训练。
*   `MultiWorkerMirroredStrategy` 用于在多台机器间进行同步训练。
*   `ParameterServerStrategy`背后的思想，适用于异步方法。
*   `TPUStrategy` 用于使用Google的张量处理单元（TPU）。

此外，您还将学习管理数据并行性的技术和调试分布式训练配置的方法。完成本章的学习将使您具备能力，来加速大型模型的训练，使用TensorFlow的分布式功能。

## 小节

- 1. [分布式机器学习的基本原理](01-%E5%88%86%E5%B8%83%E5%BC%8F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
- 2. [\`tf.distribute.Strategy\` 概述](02-%60tf.distribute.Strategy%60%20%E6%A6%82%E8%BF%B0.md)
- 3. [用于单机多 GPU 训练的 MirroredStrategy](03-%E7%94%A8%E4%BA%8E%E5%8D%95%E6%9C%BA%E5%A4%9A%20GPU%20%E8%AE%AD%E7%BB%83%E7%9A%84%20MirroredStrategy.md)
- 4. [用于多节点训练的 MultiWorkerMirroredStrategy](04-%E7%94%A8%E4%BA%8E%E5%A4%9A%E8%8A%82%E7%82%B9%E8%AE%AD%E7%BB%83%E7%9A%84%20MultiWorkerMirroredStrategy.md)
- 5. [ParameterServerStrategy 基本思想](05-ParameterServerStrategy%20%E5%9F%BA%E6%9C%AC%E6%80%9D%E6%83%B3.md)
- 6. [用于在 TPU 上训练的 TPUStrategy](06-%E7%94%A8%E4%BA%8E%E5%9C%A8%20TPU%20%E4%B8%8A%E8%AE%AD%E7%BB%83%E7%9A%84%20TPUStrategy.md)
- 7. [有效处理数据并行](07-%E6%9C%89%E6%95%88%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C.md)
- 8. [调试分布式训练任务](08-%E8%B0%83%E8%AF%95%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E4%BB%BB%E5%8A%A1.md)
- 9. [实践：实现分布式训练](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%AE%9E%E7%8E%B0%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83.md)
