---
course: "deploying-diffusion-models-scale"
chapter: "building-scalable-inference-apis"
lesson: "request-batching-techniques"
sourceId: 5439
sourceUrl: "https://apxml.com/zh/courses/deploying-diffusion-models-scale/chapter-4-building-scalable-inference-apis/request-batching-techniques"
title: "请求批处理技术"
description: "通过批处理传入的推理请求，提高GPU利用率。"
order: 3
plots: ["plots/5439-0.json"]
sourceHash: "8ea57d447efc2de1ffa651344114261b22f59bd0cfea350d5aaf5de0373dd5f7"
sourceCorrections: []
---

扩散模型推理 (inference)，特别是迭代去噪过程，大量使用GPU资源。单个推理请求，即使是中等大小的图像，也可能无法完全饱和现代GPU的计算能力。逐一处理请求会导致资源显著未充分利用，GPU在请求之间花费大量时间空闲，或者为少量数据执行内存传输。请求批处理直接解决了这种低效率问题。

基本想法很简单：系统不是在请求到达时立即将单个推理请求发送给模型工作器，而是将多个请求组合成一个“批次”。然后，扩散模型在GPU上同时处理这个批次。由于神经网络 (neural network)中的矩阵乘法等操作通常在更大的张量（代表批次）上执行效率更高，因此处理N个请求的批次通常比处理单个请求N次所需的时间明显更短。

### 动态批处理实现

对于实时推理 (inference)API，**动态批处理**是最常见和有效的策略。与批次大小固定的静态批处理不同，动态批处理根据传入的请求负载进行调整。其工作方式通常如下：

1. **缓冲：** 传入的推理请求不直接发送给模型工作器。相反，它们被暂时保存在短期缓冲区或队列中。
2. **批次形成：** 系统会等待一小段时间（可配置的超时时间），或者直到缓冲区中累积了特定数量的请求（最大批次大小）为止。
3. **批次处理：** 一旦超时时间到期或达到最大批次大小（以先发生的为准），所收集的请求就会形成一个单一的批次张量。然后，这个批次张量被发送到模型推理引擎（在GPU上运行）进行处理。
4. **结果拆分：** 模型为整个批次生成输出后，系统会解包结果，并将其映射回各自原始的单个请求。
5. **响应交付：** 最后，单个结果被发送回各自的客户端。

> 显示请求通过动态批处理系统的流程图。请求被缓冲、分组、在GPU上一起处理，然后结果被拆分。

### 权衡：吞吐量 (throughput)与延迟

批处理的主要好处是显著提高了**吞吐量**。通过让GPU更忙碌，并更有效地发挥并行处理能力，系统可以在单位时间内处理更高数量的请求。

然而，这可能以增加单个请求的**延迟**为代价。即使GPU当前处于空闲状态，请求也可能必须在缓冲区中等待批次填满或超时时间到期。与立即处理请求相比，这种额外的等待时间增加了端到端延迟。



![批处理对吞吐量和延迟的影响](plots/5439-0.json)



> 实施批处理时经常观察到的性能特征。吞吐量通常随批次大小的增加而增加（达到一定程度），而平均延迟也倾向于因缓冲区中的等待时间而增加。调整涉及找到最佳平衡。

### 实施考虑

- **超时时间调整：** 批处理超时时间是一个重要参数 (parameter)。短的超时时间可以最小化增加的延迟，但在流量低时可能导致批次较小、效率较低。长的超时时间可以最大化批次大小和吞吐量 (throughput)，但可能导致用户无法接受的延迟。最佳值通常取决于特定模型、硬件和预期负载模式，需要经验性调整。
- **最大批次大小：** 此限制通常由可用的GPU内存决定。更大的批次会消耗更多内存。设置过高可能导致内存不足（OOM）错误。
- **处理异构请求：** 批次内的请求可能具有不同的参数（例如，提示、负面提示、引导比例、推理 (inference)步数、随机种子）。推理服务器和模型代码必须设计成能够处理这种情况。提示通常可以直接进行批处理（如果使用Transformer，则需要填充）。同一批次中不同的步数或采样器可能更复杂或不可能实现，可能需要将参数不兼容的请求在单独的批次中或按顺序处理。
- **框架支持：** 许多服务框架（如NVIDIA Triton Inference Server、TorchServe、TensorFlow Serving）提供对动态批处理的内置支持，自动处理缓冲、超时逻辑和批次形成。使用这些框架可以大大地简化实施。
- **反批处理逻辑：** 确保拆分批处理结果并将其路由回正确的原始请求的机制是可靠的。这通常由批处理中间件或框架处理。

请求批处理是一种标准且非常有效的技术，用于提高基于GPU的推理的吞吐量和成本效率，特别是对于扩散模型图像生成等计算密集型任务。需要仔细调整批处理参数，以平衡吞吐量的提升与请求延迟潜在增加之间的关系。

## 参考资料

- [Dynamic Batching in Triton Inference Server](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batching) — NVIDIA Corporation (2023)
  官方文档，解释了流行推理服务框架中动态批处理的配置和行为。
- [Designing Machine Learning Systems: An Iterative Process for Production-Ready AI Applications](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen (2022)
  Publisher: O'Reilly Media
  一本关于机器学习系统设计的书籍，提供了推理服务优化及其实际影响的背景知识。
- [Clipper: A Low-Latency Online Prediction Serving System](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw) — Daniel Crankshaw, Xin Wang, Giulio Zhou, Michael J. Franklin, Joseph E. Gonzalez, Ion Stoica (2017)
  Journal: 14th USENIX Symposium on Networked Systems Design and Implementation (NSDI '17); Publisher: USENIX Association; Pages: 613-627
  一篇关于在线预测服务系统的基础论文，通过请求聚合等技术解决了延迟和吞吐量等挑战。
