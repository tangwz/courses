# 互连技术 (NVLink, InfiniBand)

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training/interconnect-technologies-nvlink-infiniband)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练大型语言模型，无论是在单个服务器内的GPU之间还是在集群中多个服务器之间的GPU上，都会引入一个主要依赖：通信。需要交换的大量数据，例如数据并行期间的梯度或模型并行期间的激活和权重 (weight)，很容易成为瓶颈，导致昂贵的计算单元停滞。像典型千兆以太网这样的标准网络接口无法满足这些需求。因此，高性能互连技术是任何LLM训练基础设施的主要组成部分。这些技术侧重于提供两个主要特性：高带宽（数据传输速率）和低延迟（数据传输启动延迟）。

### 节点内通信: NVLink

在配备多个GPU的单个服务器中，主要通信路径通常是PCIe总线。虽然现代PCIe代（如PCIe 4.0或5.0）提供可观的带宽，但它是一个共享资源，GPU之间的通信常常必须通过CPU的内存控制器，这会增加延迟。为克服这一问题，NVIDIA开发了NVLink，这是一种直接连接GPU的专有的高速点对点互连技术。

NVLink允许同一服务器内的GPU直接从各自的高带宽内存（HBM）交换数据，无需通过PCIe总线将数据流经CPU或主系统RAM。与单独使用PCIe相比，这带来了显著更高的带宽和更低的延迟。

> GPU通信路径对比。NVLink在GPU之间提供直接、高带宽连接，绕过通常用于GPU-CPU和间接GPU-GPU通信的较慢、共享的PCIe总线。

每代NVLink都增加了可用带宽。例如，NVLink 3.0（用于A100 GPU）提供每个GPU高达600 GB/s的双向带宽，而NVLink 4.0（用于H100 GPU）则将其提升至900 GB/s。将此与典型的PCIe 5.0 x16插槽进行比较，后者提供128 GB/s的双向带宽。这种显著差异对于模型并行技术（张量和流水线并行）尤其有影响，因为在这些技术中，大型中间激活或权重 (weight)段必须在处理同一模型层不同部分或完全不同层的GPU之间频繁交换。当在单个节点内的GPU之间执行时，它还显著加速了集体通信操作，例如数据并行中常用于同步梯度的`AllReduce`。

### 节点间通信: InfiniBand和高速以太网 (RoCE)

当将LLM训练扩展到单个服务器之外时，连接节点的高性能网络架构变得必要。这种架构需要处理分布在多台机器上的数百或数千个GPU之间的通信。这一领域的主要竞争者是InfiniBand和高速以太网，它们通常使用RDMA。

#### InfiniBand (IB)

InfiniBand是一种高性能计算网络标准，从零开始设计，旨在提供低延迟和高带宽。它作为一种交换式架构运行，类似于以太网，但具有针对HPC和AI工作负载优化的一些主要差异：

1. **RDMA（远程直接内存访问）：** 这是其核心功能。RDMA允许一台服务器中的网络适配器直接访问另一台服务器的内存（包括GPU内存），而无需涉及任一端的操作系统或CPU。这显著降低了延迟并释放了CPU周期。
2. **基于信用的流控制：** InfiniBand使用基于硬件的机制来阻止网络拥塞和数据包丢失，这对于RDMA和集体通信操作的性能很重要。
3. **高带宽：** InfiniBand标准持续发展，提供如HDR（每端口200 Gbps）和NDR（每端口400 Gbps）等速度，未来标准承诺提供更高的速率。

InfiniBand历来是大型AI训练集群的首选网络方案，因为它具有持续的低延迟和成熟的RDMA实现。

#### 带RoCE的高速以太网

以太网普遍存在，其进步已将其速度推向数百吉比特每秒（例如，200 GbE，400 GbE）。为了与InfiniBand在HPC/AI方面的低延迟能力竞争，RoCE（基于融合以太网的RDMA）协议被开发出来。RoCE允许RDMA操作运行在标准以太网基础设施上。

- **RoCE v1：** 直接在以太网MAC层上运行。在同一二层域内需要无损以太网网络（通常通过优先级流控制，PFC进行配置）。
- **RoCE v2：** 通过UDP/IP运行。这允许跨三层网络路由，使其更灵活，但仍强烈受益于（且通常需要）精心配置的无损网络，以实现与InfiniBand可比的性能。

RoCE的主要优势在于其利用现有以太网基础设施和专业知识的潜力，可能提供更低的入门成本。然而，与内置无损操作的InfiniBand相比，在大型以太网架构中实现必要的无损行为可能配置和管理起来更复杂。在性能方面，配置良好的现代高速以太网与RoCE可以实现与当今InfiniBand速度非常接近的延迟和带宽。

### 网络拓扑

服务器的连接方式（网络拓扑）也影响通信性能，特别是对于涉及许多节点的集体操作。在大型集群中，常用胖树（Fat-Tree）或蜻蜓（Dragonfly）等拓扑结构。它们旨在提供高二分带宽，这意味着即使许多节点需要在网络的不同部分同时交换数据，也有充足的通信能力。不足的拓扑可能导致瓶颈，即使单个链路速度很高。

### 软件集成

深度学习 (deep learning)框架和通信库抽象化了许多硬件细节。NVIDIA的NCCL（NVIDIA集体通信库）等库针对NVIDIA GPU上的集体操作（例如`AllReduce`、`Broadcast`、`AllGather`）进行了高度优化。

```python
import torch
import torch.distributed as dist
import os

def setup_distributed(backend='nccl'):
  """初始化分布式进程组。"""
  # 假设已设置环境变量 MASTER_ADDR、MASTER_PORT、
  # RANK 和 WORLD_SIZE。
  rank = int(os.environ['RANK'])
  world_size = int(os.environ['WORLD_SIZE'])
  master_addr = os.environ['MASTER_ADDR']
  master_port = os.environ['MASTER_PORT']

  # 初始化进程组
  # NCCL 后端将自动尝试使用 NVLink 进行节点内通信，
  # 并使用 InfiniBand/RoCE（如果可用且已配置）
  # 进行节点间通信。
  dist.init_process_group(
      backend=backend,
      init_method=f'tcp://{master_addr}:{master_port}',
      rank=rank,
      world_size=world_size
  )
  # 为当前进程设置设备
  torch.cuda.set_device(rank % torch.cuda.device_count())
  print(
      f"进程 {rank}/{world_size} 已在设备 "
      f"{torch.cuda.current_device()} 上初始化。"
  )

# --- 示例用法 ---
# if __name__ == "__main__":
#   setup_distributed()
#
#   # 集体操作示例
#   tensor = (torch.ones(1, device=torch.cuda.current_device())
#             * dist.get_rank())
#   print(f"进程 {dist.get_rank()} 在 all_reduce 前: {tensor}")
#
#   # NCCL 处理通过 NVLink/InfiniBand/RoCE 路由此操作
#   dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
#
#   print(f"进程 {dist.get_rank()} 在 all_reduce 后: {tensor}")
#
#   dist.destroy_process_group()
```

> PyTorch 代码，使用 NCCL 后端初始化分布式进程组。NCCL 智能选择最佳可用互连方式（NVLink、InfiniBand、RoCE）进行`dist.all_reduce`等通信操作。

当`torch.distributed`使用`nccl`后端初始化时，NCCL会探测系统硬件，并自动利用NVLink进行同一节点内GPU之间的快速通信，并利用InfiniBand或RoCE（如果可用且正确配置）进行不同节点上GPU之间的通信。因此，选择和配置正确的互连技术是一项硬件和基础设施任务，但其优势通过这些高级软件库在训练期间得以实现。

总之，高速互连是高效大规模LLM训练不可或缺的组成部分。NVLink提供服务器内GPU之间高效的高带宽、低延迟通道，这对模型并行和快速节点内集体操作非常重要。InfiniBand和高速以太网（带RoCE）提供扩展训练到多服务器所需的网络架构，支持大规模数据和流水线并行方案。InfiniBand和以太网/RoCE之间的选择涉及性能一致性、成本和配置复杂性方面的权衡，但两者都旨在最大程度地减少通信开销，否则通信开销可能严重限制训练吞吐量 (throughput)。

## 参考资料

- [NVIDIA Hopper Architecture In-Depth](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper) — NVIDIA Corporation (2022)
  Publisher: NVIDIA
  描述了 NVIDIA Hopper H100 GPU 的架构，包括 NVLink 4.0 及其在 GPU 之间通信中的作用。
- [NVIDIA Collective Communications Library (NCCL) Developer Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html) — NVIDIA Corporation (2023)
  Publisher: NVIDIA Corporation
  NCCL 的官方指南，详细介绍了其架构、支持的集体操作以及如何利用高速互连优化 GPU 通信。
- [Deep Learning Systems: Algorithms, Architectures, and Frameworks](https://link.springer.com/book/10.1007/978-981-15-4682-1) — Chen-Yi Lee, Chang-Won Jin, Ru-Han Jhou, Hsin-Fu Lin, Wei Li, Ya-Ping Hsieh, Yu-Syuan Jhou (2020)
  Publisher: Springer Nature Singapore; DOI: [10.1007/978-981-15-4682-1](https://doi.org/10.1007/978-981-15-4682-1)
  一本教材，概述了深度学习系统，包括分布式训练和高性能互连作用的讨论。

---

[上一节](03-%E5%86%85%E5%AD%98%E9%9C%80%E6%B1%82%EF%BC%88HBM%E3%80%81GPU%E6%98%BE%E5%AD%98%EF%BC%89.md) · [下一节](05-%E7%A1%AC%E4%BB%B6%E9%80%89%E6%8B%A9%E7%9A%84%E6%9D%83%E8%A1%A1%20%28%E6%88%90%E6%9C%AC%E3%80%81%E6%80%A7%E8%83%BD%E3%80%81%E5%8F%AF%E7%94%A8%E6%80%A7%29.md)
