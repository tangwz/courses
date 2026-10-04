# 第 2 章：第二章：分布式模型训练的工程化

来源：[原章节](https://apxml.com/zh/courses/advanced-ai-infrastructure-design-optimization/chapter-2-engineering-distributed-model-training)

[返回课程目录](../README.md)

当今模型的规模，尤其是那些参数量达到数十亿（$N > 10^9$）的模型，常常超出单个GPU的内存容量。训练这些模型需要将工作负载分配到加速器集群上。本章提供构建高效分布式训练系统的工程原理和实用方法。

我们将从审视主要的分布式策略开始。您将学习数据并行机制，其中模型被复制，数据被分片，以及梯度同步中涉及的通信模式。对于单个设备无法容纳的模型，我们将介绍模型并行和流水线并行，这涉及将模型的层或操作划分到多个加速器上。

接着，重点将转向使用生产级框架进行实现。我们将使用Horovod，因为它对数据并行采取直接方法，然后转向微软的DeepSpeed，以实现先进的内存优化技术，例如零冗余优化器（ZeRO）。

最后，我们将讨论大规模训练的运行实际情况。您将学习如何通过高效的检查点机制来设计容错能力，这是长时间运行任务的必要组成部分。本章以实践实验室环节结束，在此环节中，您将配置并运行一个使用PyTorch的完全分片数据并行（FSDP）的Transformer模型分布式训练任务。

## 小节

- 1. [数据并行：同步与异步更新](01-%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%EF%BC%9A%E5%90%8C%E6%AD%A5%E4%B8%8E%E5%BC%82%E6%AD%A5%E6%9B%B4%E6%96%B0.md)
- 2. [大型模型的模型与流水线并行](02-%E5%A4%A7%E5%9E%8B%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%A8%A1%E5%9E%8B%E4%B8%8E%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C.md)
- 3. [使用 Horovod 进行训练](03-%E4%BD%BF%E7%94%A8%20Horovod%20%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83.md)
- 4. [使用 Microsoft DeepSpeed 实现 ZeRO 和卸载](04-%E4%BD%BF%E7%94%A8%20Microsoft%20DeepSpeed%20%E5%AE%9E%E7%8E%B0%20ZeRO%20%E5%92%8C%E5%8D%B8%E8%BD%BD.md)
- 5. [长时间运行任务中的容错与检查点技术](05-%E9%95%BF%E6%97%B6%E9%97%B4%E8%BF%90%E8%A1%8C%E4%BB%BB%E5%8A%A1%E4%B8%AD%E7%9A%84%E5%AE%B9%E9%94%99%E4%B8%8E%E6%A3%80%E6%9F%A5%E7%82%B9%E6%8A%80%E6%9C%AF.md)
- 6. [动手实践：使用 PyTorch FSDP 进行分布式训练](06-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E4%BD%BF%E7%94%A8%20PyTorch%20FSDP%20%E8%BF%9B%E8%A1%8C%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83.md)
