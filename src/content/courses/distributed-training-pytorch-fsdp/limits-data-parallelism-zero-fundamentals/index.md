---
course: "distributed-training-pytorch-fsdp"
sourceUrl: "https://apxml.com/zh/courses/distributed-training-pytorch-fsdp/chapter-1-limits-data-parallelism-zero-fundamentals"
sourceId: 1372
chapter: "limits-data-parallelism-zero-fundamentals"
title: "数据并行的局限性与 ZeRO 基本原理"
order: 1
description: "分析分布式数据并行的内存约束，并学习 FSDP 中使用的 ZeRO 分片算法的理论原理。"
hasQuiz: false
---

训练深度学习模型时，单个 GPU 的内存容量是严格的物理限制。分布式数据并行 (DDP) 长期以来是跨多个设备扩展训练的标准方法。然而，DDP 的工作方式是在每个 GPU 上复制完整的模型状态。尽管这对较小的架构有效，但这种冗余对大型语言模型而言会成为瓶颈。如果模型的参数、梯度和优化器状态的总和超出了单个设备的显存，DDP 便无法工作，无论您的集群中有多少个 GPU。

本章将介绍从模型复制到模型分片的架构转变。我们将分析与训练相关的具体内存开销，细分 32 位优化器状态与 16 位权重和梯度的消耗。对于一个参数量为 $\Psi$ 的模型，我们将说明在 Adam 优化器进行混合精度训练时，内存占用通常会扩展到约 $$16\Psi$$ 字节，而非假定的 $$2\Psi$$ 或 $$4\Psi$$。

您将学习 Zero Redundancy Optimizer (ZeRO) 的理论与实践原理，它为 PyTorch FSDP 提供支持。内容将阐述 ZeRO 如何在可用设备间划分数据以消除冗余。我们将介绍 ZeRO 优化的三个阶段：

*   **阶段 1：** 分片优化器状态。
*   **阶段 2：** 分片梯度。
*   **阶段 3：** 分片模型参数。

通过掌握这些机制，您将能够计算您的硬件所需的准确内存，并配置必要的通信原语。本章以一个实践练习结束，您将在其中使用 FSDP 包装一个标准的 PyTorch 模块，为课程后续介绍的高级优化建立基线。
