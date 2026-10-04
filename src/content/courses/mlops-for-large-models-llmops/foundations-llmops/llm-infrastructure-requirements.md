---
course: "mlops-for-large-models-llmops"
chapter: "foundations-llmops"
lesson: "llm-infrastructure-requirements"
sourceId: 3106
sourceUrl: "https://apxml.com/zh/courses/mlops-for-large-models-llmops/chapter-1-foundations-llmops/llm-infrastructure-requirements"
title: "大型模型所需的基础设施"
description: "详细说明大型语言模型整个运行流程所需的硬件（GPU/TPU集群）和软件设施。"
order: 3
plots: []
sourceHash: "5e1c63fc4ad66c50289df2c6034b38072f7ece9ddf82b1c962f924fe2007238a"
sourceCorrections: []
---

运行标准机器学习 (machine learning)模型通常依赖现成的计算资源，也许是一台性能强大的机器或一个适度的集群。然而，扩展到大型语言模型 (LLMs)，其参数 (parameter)数量 $P$ 达到数十亿甚至数万亿 ($P \gg 10^9$)，这彻底改变了基础设施配置。训练和推理 (inference)所需的计算与内存资源，都需要专门的高性能环境，远超一般的MLOps配置。在此分析所需的硬件和软件构成。

## 计算：大型语言模型的驱动核心

大型语言模型运行的核心是专为大规模并行计算设计的硬件加速器，主要是图形处理单元 (GPUs) 和张量处理单元 (TPUs)。

### GPU（图形处理单元）

GPU，特别是专为数据中心设计的型号，如NVIDIA的A100或H100系列，是大部分大型语言模型任务的主力。其架构拥有数千个核心，擅长处理深度学习 (deep learning)任务中占主导地位的矩阵乘法和张量运算。以下几个因素使得选择特定GPU非常重要：

- **内存容量 (VRAM)：** 大型语言模型具有极大的内存占用。模型参数 (parameter)、训练过程中的中间激活值、梯度和优化器状态都必须容纳在加速器的内存中。一个拥有1750亿参数的模型，若使用混合精度（16位浮点数），仅权重 (weight)就需要至少350GB（$175 \times 10^9 \text{ 参数} \times 2 \text{ 字节/参数}$）。这常超出单个GPU的容量（例如A100的80GB），因此需要多个GPU共同运作。
- **计算能力 (FLOPS)：** 以每秒浮点运算次数衡量，这决定了GPU执行计算的速度。更高的FLOPS意味着更快的训练和推理 (inference)速度。
- **互联：** NVIDIA的NVLink或NVSwitch等高速互联技术可连接单个服务器内的GPU，相比标准PCIe通道，实现了GPU之间更快的直接内存访问。这对于模型并行技术非常重要，因为模型的某些部分会分布在不同的GPU上。

训练甚至微调 (fine-tuning)大型模型通常不仅需要一个GPU，而是需要由数十、数百乃至数千个GPU共同运作的集群。

### TPU（张量处理单元）

TPU由Google开发，是专用集成电路 (ASICs)，专门设计用于加速神经网络 (neural network)中的核心矩阵计算。它们对于大规模训练和推理 (inference)任务特别有效，通常通过Google Cloud Platform访问。虽然GPU为通用并行任务提供更大的灵活性，但对于其优化的特定大型Transformer工作负载，TPU能够带来显著的性能和成本优势。

## 网络：分布式系统的命脉

当训练或推理 (inference)跨越多台机器（节点）时，连接它们的网络就成为影响性能的重要因素。大型语言模型训练，尤其是使用数据并行时，涉及在所有参与节点之间频繁同步梯度或模型参数 (parameter)。

- **高带宽：** 网络必须处理大量数据传输，以定期同步可能达数千兆字节的梯度数据。通常需要InfiniBand（例如200-400 Gbps）或支持RDMA（远程直接内存访问）的高速以太网（100/200/400 GbE）等技术。
- **低延迟：** 通信延迟直接导致GPU空闲，减缓整个流程。低延迟对于紧密耦合的分布式训练算法不可或缺。

网络瓶颈很容易抵消拥有强大加速器的优势，使得网络设计成为重要的基础设施考量。

> 这是大型语言模型多节点计算集群的简化视图，突出显示了节点内部互联的GPU (NVLink)，以及通过专用网卡 (NICs) 和交换机实现节点间的高速网络连接。

## 存储：为系统提供动力

大型语言模型在海量数据集上训练，数据量常以太字节或拍字节计算。高效地将这些数据传输到计算集群并存储模型检查点，需要高性能存储解决方案。

- **训练数据存储：** 需要高吞吐量 (throughput)系统，能够同时向数百或数千个GPU工作器传输数据，而不会造成输入/输出 (I/O) 瓶颈。选项包括高性能计算 (HPC) 中常见的并行文件系统（例如Lustre、BeeGFS），或针对云对象存储（例如AWS S3、Google Cloud Storage、Azure Blob Storage）的优化访问模式。
- **检查点存储：** 训练大型模型可能需要数周或数月。频繁进行检查点保存（保存模型状态）对于容错能力不可或缺。这需要快速、可靠的存储，以最大限度地减少保存和加载检查点所需的时间，这些检查点可能大小达数百GB或数TB。

## 软件栈：编排与优化

单独的硬件不足以完成任务。需要一套复杂的软件栈来管理这些资源并高效执行大型语言模型工作流。

- **分布式计算框架：** PyTorch Distributed Data Parallel (DDP) 或 TensorFlow MirroredStrategy 等库提供了基础工具。然而，DeepSpeed (Microsoft) 和 Megatron-LM (NVIDIA) 等专用框架在此基础上构建，提供针对大型模型优化的内存优化（例如ZeRO优化器）和混合并行（结合数据、张量和流水线并行）等高级技术。Horovod 是另一个用于分布式训练的常用框架。
- **集群编排：** Kubernetes（常带有GPU调度扩展）或Slurm（在HPC环境中常用）等工具用于管理集群资源、调度训练和推理 (inference)作业、处理节点故障以及管理容器化应用。
- **容器化：** Docker或类似的容器技术对于打包大型语言模型工作负载所需的复杂依赖项（CUDA库、Python包、框架版本）不可或缺，确保开发、测试和生产环境之间的一致性。
- **监控与日志：** 考虑到规模和成本，对整个集群的GPU利用率、温度、内存使用、网络流量和应用日志进行全面监控不可妥协。Prometheus、Grafana和专用日志平台等工具是基础设施的组成部分。

## 推理 (inference)与训练基础设施对比

虽然大部分基本基础设施（GPU、网络）是相似的，但推理的具体要求和优化目标与训练有所不同：

- **训练：** 针对大型专用集群的最大吞吐量 (throughput)和最小化总训练时间进行优化。成本高昂，但通常分摊到模型的整个生命周期中。
- **推理：** 针对低延迟（快速响应时间）、高查询吞吐量（每秒请求数）和每次查询的成本效益进行优化。这通常涉及不同的硬件配置（可能每个服务器的GPU数量更少，但服务器在地理上分布更广）、优化的推理服务器（例如NVIDIA Triton Inference Server、vLLM），以及模型量化 (quantization)或蒸馏等技术，以减少资源需求。

了解这些不同的基础设施要求是构建LLMOps策略的第一步。其规模和复杂性需要仔细规划和选择硬件及软件，以适应大型语言模型的独特需求，从而构建起高效训练、微调 (fine-tuning)、部署和监控流程的支撑。后续章节将讨论如何管理这些基础设施以及实现大型语言模型成功部署所需的运行工作流。

## 参考资料

- [ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models](https://proceedings.neurips.cc/paper/2020/hash/b77f88262705edf177b425b7a13c4196-Abstract.html) — Samyam Rajbhandari, Cong Guo, Erin Grant, Zhun Liu, Hao Ma, Aakanksha Chowdhery, Yuxiong He (2020)
  Journal: Advances in Neural Information Processing Systems 33 (NeurIPS 2020); Publisher: NeurIPS; Volume: 33; Pages: 8027-8038; DOI: [10.5591/978-1-7138-0457-3.NeurIPS.2020.00762](https://doi.org/10.5591/978-1-7138-0457-3.NeurIPS.2020.00762)
  本文介绍了ZeRO，一种通过在设备间分区优化器状态、梯度和参数来高效训练大型深度学习模型的内存优化技术。
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro (2019)
  Journal: arXiv preprint arXiv:1909.08053; DOI: [10.48550/arXiv.1909.08053](https://doi.org/10.48550/arXiv.1909.08053)
  描述了如何通过在多个GPU上应用模型并行（包括张量并行和流水线并行）来高效训练超大型语言模型。
- [A Cloud-Scale Architecture for Accelerating AI](https://dl.acm.org/doi/10.1145/3465033.3467614) — Norman P. Jouppi, Zhifeng Chen, David Pattison, Carla J. del Rosario, Nils Hogstrom, David A. Huffman, Daniel F. Kramer, Andrew B. New, Christopher B. Perry, Evan E. Radecki, Timothy B. Smith, Ryan M. Some, David R. Turnbull, Andrew J. Veit, John W. Weigelt, Dean M. Wilkes, Cliff C. Young, Yifeng Chao, Andrew L. Chien, Patrick K. H. Chiang, John A. Gunnels, Mark R. O'Connor, Anant N. Agarwal, Jeffrey Dean, Paul N. Hilfinger, Jeffrey S. Riegel, Edward A. Sacks, Manjot Singh, Stephen W. Smith, Jonathan S. Taylor, David L. Wells, Kenneth C. Yocum (2021)
  Journal: Proceedings of the 48th Annual International Symposium on Computer Architecture (ISCA '21); Publisher: ACM; Pages: 1-14; DOI: [10.1145/3465033.3467614](https://doi.org/10.1145/3465033.3467614)
  介绍了谷歌张量处理单元（TPU）的架构，该架构专为云规模AI工作负载设计，详细阐述了其专用互连和处理能力。
- [NVIDIA H100 Tensor Core GPU Architecture](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHpBolHGpAidZplPdU-LyMQlVC_g-4OOlmzo5sKz_S9qn8zvJlHZTgKy1vc3EW6TmMVcAq29qWXmDzg2izPPjNEPEeS9xQUDk5zJGfBxVHAhQ27dKy0TaqxrIhU-Gbphkl5KJhDcTFlApgQEZNmIiytyRVymul3eagQtZejPqm3tImUg5rledytU69xAxL-RVIIIC7e) — NVIDIA Corporation (2022)
  Publisher: NVIDIA
  NVIDIA H100 GPU架构的官方白皮书，介绍了Transformer Engine和NVLink等创新技术，这些技术与大型语言模型计算相关。
