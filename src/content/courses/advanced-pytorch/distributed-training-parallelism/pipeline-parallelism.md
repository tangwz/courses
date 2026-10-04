---
course: "advanced-pytorch"
chapter: "distributed-training-parallelism"
lesson: "pipeline-parallelism"
sourceId: 2768
sourceUrl: "https://apxml.com/zh/courses/advanced-pytorch/chapter-5-distributed-training-parallelism/pipeline-parallelism"
title: "流水线并行实现"
description: "将模型层按顺序分配到多个设备上，以平衡计算并减少每个设备的内存使用。"
order: 4
plots: []
sourceHash: "9928830cc93330c13e74def3f65d150101b352f0b0b3cccb6262c8d8a623b7f4"
sourceCorrections: []
---

当神经网络 (neural network)变得过大以至于单个设备无法容纳其任何一层，或者需要以不同方式重叠计算和通信时，流水线并行提供了一种替代的扩展策略。不同于复制整个模型或拆分单个层的方法，流水线并行将*模型本身*按顺序分配到多个设备上。每个设备或设备组都成为流水线中的一个“阶段”，负责运行模型层的一个子集。

### 流水线并行的机制

设想一个由多个连续层或模块组成的模型。在流水线并行中，你将连续的模块分配给不同的设备。例如，一个在四个GPU上运行的四层模型：

1. **GPU 0 (阶段 0):** 运行第 1 层。
2. **GPU 1 (阶段 1):** 运行第 2 层。
3. **GPU 2 (阶段 2):** 运行第 3 层。
4. **GPU 3 (阶段 3):** 运行第 4 层并计算损失。

输入数据进入第一阶段（GPU 0）。处理后，输出激活被发送到第二阶段（GPU 1）。这会一直持续，直到最后阶段计算出输出和损失。随后，梯度以相反的顺序反向流经流水线。GPU 3 计算第 4 层的梯度，并将第 3 层输出的梯度发回给 GPU 2，然后 GPU 2 计算第 3 层的梯度并将其发回给 GPU 1，依此类推，直到梯度到达第一阶段。

### 流水线气泡问题

此过程的简单实现效率低下。思考时间线：当阶段 1 处理第一个数据批次时，阶段 0 处于空闲状态，等待下一个批次。类似地，当阶段 2 处理时，阶段 0 和 1 处于空闲状态（假设只有一个批次流过）。在反向传播 (backpropagation)期间，各阶段会再次处于空闲状态，因为它们在等待来自后续阶段的梯度。这种空闲时间，被称为“流水线气泡”，大幅降低了硬件利用率。

> 单批次简单流水线运行示意图，显示了在时间步长（T1-T8）内的正向（Fwd）和反向（Bwd）传播过程中，GPU上存在大量空闲时间（气泡）。

### 通过微批次处理减少气泡

解决流水线气泡问题的标准方法是**微批次处理**。我们不是一次性将整个小批次数据送入流水线，而是将其分成更小的块，称为微批次。流水线并发处理这些微批次。

一旦阶段 0 完成处理第一个微批次并将其激活发送到阶段 1，阶段 0 就可以立即开始处理第二个微批次。这使得多个微批次可以在流水线中同时“飞行”，重叠各阶段的计算，并大幅减少空闲时间。微批次的数量（$m$）是一个超参数 (parameter) (hyperparameter)；更大的 $m$ 通常会带来更好的利用率，但会增加通信开销，并可能因存储每个微批次的中间激活和梯度而增加内存使用量。

> 使用微批次处理（m0-m3）的流水线运行示意图。不同微批次的正向（F）和反向（B）传播在各阶段（GPU）之间重叠，与简单方法相比，减少了空闲时间。初始填充和最终排空阶段仍存在一些气泡。

### 在PyTorch中实现流水线并行

PyTorch提供了实现流水线并行的工具，尽管它通常比DDP需要更多手动设置。其主要构成部分包括：

1. **模型划分：** 将你的 `nn.Module` 手动拆分为连续的 `nn.Sequential` 模块，每个阶段一个。将每个模块放置在其指定设备上。
2. **通信：** 使用 `torch.distributed.send` 和 `torch.distributed.recv` 操作，在相邻阶段之间正向传输激活和反向传输梯度。请记住这些操作是阻塞的。
3. **微批次调度：** 实现一个循环，遍历微批次，管理每个微批次在各阶段的正向和反向传播 (backpropagation)。这需要仔细同步。
4. **梯度同步：** 每个微批次在反向传播期间计算的梯度需要累积。优化器步骤通常仅在小批次中的所有微批次都完成其正向和反向传播后才执行。

我们来概述一个带有微批次处理的简单两阶段流水线（GPU 0 和 GPU 1）的流程：

```python
import torch
import torch.nn as nn
import torch.distributed as dist

# 假设分布式环境已初始化（rank 0 在 GPU 0，rank 1 在 GPU 1）
# 假设模型已拆分为 stage0 和 stage1，并放置在各自的设备上

def run_pipeline_step(stage0, stage1, micro_batches_data, micro_batches_labels, loss_fn, optimizer):

    num_micro_batches = len(micro_batches_data)
    activations_storage = [None] * num_micro_batches # 存储用于反向传播的激活
    gradients_storage = [None] * num_micro_batches # 存储用于反向传播的梯度

    current_rank = dist.get_rank()
    world_size = dist.get_world_size() # 在此示例中假设 world_size = 2

    # --- 正向传播 ---
    for i in range(num_micro_batches):
        micro_batch = micro_batches_data[i]

        if current_rank == 0: # 第一阶段
            # 计算阶段 0 的激活
            activations = stage0(micro_batch.to(current_rank))

            # 将激活发送到下一阶段（rank 1）
            dist.send(activations.cpu(), dst=1, tag=i) # 发送 CPU 张量以避免 GPU 同步问题
            activations_storage[i] = activations # 存储用于反向传播

        elif current_rank == 1: # 最后阶段
            # 从上一阶段（rank 0）接收激活
            received_activations = torch.empty_like(some_prototype_tensor_shape, device='cpu') # 需要形状信息
            dist.recv(received_activations, src=0, tag=i)
            received_activations = received_activations.to(current_rank)
            received_activations.requires_grad_() # 重要：为接收到的张量启用梯度

            # 计算阶段 1 的激活（最终输出）
            outputs = stage1(received_activations)

            # 计算损失
            labels = micro_batches_labels[i].to(current_rank)
            loss = loss_fn(outputs, labels)

            # 存储反向传播所需信息
            activations_storage[i] = received_activations # 此阶段的输入
            gradients_storage[i] = loss # 存储损失以供稍后启动反向传播

    # --- 反向传播 ---
    # 为确保正确性，对微批次进行反向迭代（GPipe 调度）
    for i in range(num_micro_batches - 1, -1, -1):
        if current_rank == 1: # 最后阶段
            loss = gradients_storage[i]
            input_activation = activations_storage[i]

            # 为此微批次的损失启动反向传播
            # 局部计算阶段 1 参数的梯度
            # 如果不是此阶段输入的最后一个微批次，则需要保留计算图
            retain_graph_flag = (i != 0) 
            loss.backward(retain_graph=retain_graph_flag) 

            # 将输入激活的梯度发回上一阶段（rank 0）
            grad_to_send = input_activation.grad.cpu() 
            dist.send(grad_to_send, dst=0, tag=i)

        elif current_rank == 0: # 第一阶段
            # 从下一阶段（rank 1）接收梯度
            grad_received = torch.empty_like(some_prototype_grad_shape, device='cpu') # 需要形状信息
            dist.recv(grad_received, src=1, tag=i)
            grad_received = grad_received.to(current_rank)

            # 使用接收到的梯度继续反向传播
            output_activation = activations_storage[i]
            # 局部计算阶段 0 参数的梯度
            output_activation.backward(gradient=grad_received)

    # --- 优化器步骤 ---
    # 处理完所有微批次后，更新权重
    optimizer.step()
    optimizer.zero_grad()

# --- 注意 ---
# 1. 这是一个简化示例（GPipe 风格的调度）。
# 2. 需要机制来确定接收缓冲区中的张量形状。
# 3. 错误处理、正确的设备放置和同步很关键。
# 4. 存在更高级的调度（例如，交错式）。
# 5. 像 `torch.distributed.pipeline`（实验性）这样的库旨在简化此过程。
```

这种手动实现突出了其复杂性：显式通信调用，管理每个微批次的中间激活及其梯度，以及阶段之间细致的同步。

PyTorch 还有一个实验性的 `torch.distributed.pipeline.sync.Pipe` 模块，旨在抽象化部分这种复杂性，根据拆分为阶段的模型定义，自动处理微批次处理、通信和梯度传播。然而，使用基本操作理解手动过程，能帮助更好了解其内部机制。

### 考量与权衡

- **负载均衡：** 每个阶段的计算成本应大致相等，以避免阶段成为瓶颈。这通常需要仔细的模型划分。
- **通信开销：** 在设备之间传输激活和梯度会引入开销。如果张量尺寸较大或互连带宽较低，则这种开销会更加明显。微批次处理由于更频繁、更小的传输而增加了这种开销。
- **内存：** 虽然*每个设备*的峰值内存有所减少（因为每个设备只保存模型的一部分），但存储多个正在处理的微批次的激活的需求，与简单的流水线相比，可能增加整体内存需求。
- **复杂性：** 正确实现流水线并行，尤其是在微批次处理和优化调度方面，比使用 DDP 复杂得多。

流水线并行对于超大型模型最有用，即使张量并行也不足以应对，或者需要对设备运行和内存进行精细控制时。它通常与数据并行结合使用（例如，在每个流水线阶段内运行 DDP），以实现进一步的扩展。

## 参考资料

- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/pdf/1905.01329.pdf) — Yanping Huang, Youlong Cheng, Dehao Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Thang Luong, Yonghui Wu, Zhifeng Chen (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: NeurIPS Foundation; Pages: 1083-1094; DOI: [10.48550/arXiv.1905.01329](https://doi.org/10.48550/arXiv.1905.01329)
  介绍了用于流水线并行的GPipe算法，详细说明了如何使用微批处理来减少空闲时间。
- [Distributed communication package - \`torch.distributed\`](https://pytorch.org/docs/stable/distributed.html) — PyTorch Contributors (2025)
  Publisher: PyTorch
  PyTorch分布式通信原语的官方文档，是手动实现流水线并行的基础。
- [DeepSpeed: Large-Scale Distributed Training with Pipeline Parallelism](https://arxiv.org/pdf/2006.02708.pdf) — Samyam Rajbhandari, Cong Li, Zhun Liu, Olatunji Ruwase, Justin K. Romberg, Yuxiong He (2020)
  Journal: arXiv preprint arXiv:2006.02708; DOI: [10.48550/arXiv.2006.02708](https://doi.org/10.48550/arXiv.2006.02708)
  展示了DeepSpeed的流水线并行功能，包括用于高效训练大型模型的各种调度策略。
