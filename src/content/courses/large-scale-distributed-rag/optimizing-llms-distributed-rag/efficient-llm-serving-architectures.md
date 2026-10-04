---
course: "large-scale-distributed-rag"
chapter: "optimizing-llms-distributed-rag"
lesson: "efficient-llm-serving-architectures"
sourceId: 6185
sourceUrl: "https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-3-optimizing-llms-distributed-rag/efficient-llm-serving-architectures"
title: "高效LLM服务架构"
description: "学习用于在高吞吐量RAG系统中部署LLM的高效服务架构（例如，vLLM，TGI）。"
order: 1
plots: []
sourceHash: "12e6a3d0a05b31a9b672b30dba9fe1d00410e438846e43eee3c8d2bc626273d7"
sourceCorrections: []
---

在分布式检索增强生成（RAG）系统中部署大型语言模型时，为这些模型选择的服务架构是一个核心要素。它直接影响吞吐量 (throughput)、延迟、成本和整体用户体验。适用于小型应用或离线批处理的简单模型部署策略，通常不足以满足大规模实时RAG的需求。这些系统需要能够处理高并发请求量、管理大量内存占用（特别是对于来自检索文档的长上下文 (context)的KV缓存），并优化计算资源使用的服务方案。

### 分布式RAG中LLM服务的主要难题

在RAG管道中大规模有效服务LLM面临一些工程方面的难题，这些需要服务架构来解决：

- **高内存带宽要求：** LLM中的自回归 (autoregressive)解码过程主要受限于内存带宽，而不仅仅是计算能力。每个生成的令牌都需要加载模型权重 (weight)并访问键值（KV）缓存。有效管理这种数据移动十分要紧。
- **KV缓存管理：** KV缓存存储先前生成令牌的注意力键和值，显著加快后续令牌的生成。然而，其大小随序列长度呈平方级增长，并随批次大小线性增长。在RAG中，由于检索到的文档，输入上下文 (context)可能非常大，这加剧了KV缓存的内存压力。简单的管理会导致内存碎片化和浪费。
- **动态请求负载：** RAG系统经常面临波动的请求模式。服务层必须通过高效地批处理具有不同输入和输出长度的传入请求来适应。如果请求等待时间过长，静态批处理可能导致GPU利用率低下或高延迟。
- **低延迟和高吞吐量 (throughput)：** 交互式RAG应用要求低端到端延迟。同时，系统必须支持高每秒请求数（RPS），以实现成本效益并服务大量用户。这两个目标通常相互制约。
- **可扩展性和成本效益：** 随着需求增长，服务基础设施必须扩展。这不仅涉及增加计算资源，还包括确保这些资源得到有效利用以控制运营成本。

### 高性能LLM服务架构原则

为应对这些难题，现代LLM服务架构包含几项先进原则：

- **连续批处理（在途批处理）：** 与必须在处理新请求之前完成整个批次的静态批处理不同，连续批处理允许在当前批次中的单个序列完成后立即将新请求添加到批次中。这显著提高了GPU利用率并降低了平均延迟，特别是在RAG中常见的异构请求长度情况下。
- **PagedAttention：** 这种由vLLM推广的创新性内存管理算法，将KV缓存视为操作系统中的虚拟内存。它以固定大小的块（页面）不连续地分配KV缓存内存。这几乎消除了内部碎片（已分配块内的内存浪费），并显著减少了外部碎片，从而实现了更大的有效批次大小和更好的内存利用，这对于长上下文 (context)RAG尤其有利。
- **优化内核：** 为注意力计算、层归一化 (normalization)和激活函数 (activation function)等核心操作使用定制的CUDA内核，可以比简单实现带来显著的性能提升。这些内核通常针对特定的GPU架构进行手动调优。
- **模型并行（张量和流水线）：** 对于超出单个加速器内存容量的特大型LLM，会采用模型并行。
  - **张量并行：** 将模型的单个层或张量拆分到多个GPU上。对这些分布式张量的操作需要GPU之间的通信。
  - **流水线并行：** 将模型层划分为多个阶段，每个阶段由不同的GPU处理。数据微批次流过此流水线。
    尽管这些方法使得服务更大的模型成为可能，但它们引入了通信开销和复杂度。
- **量化 (quantization)感知服务：** 服务量化模型（例如，8位、4位）可以减少内存占用并加速推理 (inference)。高效的服务系统为这些量化格式提供优化运行时，确保理论优势转化为实际收益。
- **推测解码：** 这种技术使用更小、更快的草稿模型生成一系列候选令牌，然后由更大、更准确的目标LLM在一次前向传播中进行验证或纠正。如果大比例的草稿令牌被接受，这可以显著减少通过大型模型的顺序前向传播次数，从而降低延迟。这是当前研究和实现中的一个活跃方向。

### 主要LLM服务框架

几个开源框架已经出现，旨在简化LLM的部署和优化，其中包含了上述许多原则。这些对于在分布式RAG系统中构建LLM服务层特别适用。

#### vLLM

vLLM是一个开源的LLM推理 (inference)和服务引擎，专门为高吞吐量 (throughput)设计。其标志性特点是**PagedAttention**，如前所述，它优化了KV缓存管理。
主要特点：

- **高吞吐量：** 通过PagedAttention和连续批处理实现，带来高效的GPU利用。
- **优化内核：** 包含高度优化的CUDA内核。
- **OpenAI API兼容性：** 通常提供与OpenAI聊天完成兼容的API端点，简化集成。
- **支持多种模型：** 兼容广泛的Hugging Face Transformer模型。

vLLM特别适合于，以最大化具有可能长且多变的上下文 (context)长度（RAG中常见）的模型的吞吐量为首要目标的场景。

#### Hugging Face的Text Generation Inference (TGI)

TGI是由Hugging Face开发的工具包，用于部署和服务大型语言模型。它为生产环境设计，并在性能和易用性之间提供了平衡。
主要特点：

- **连续批处理：** 实现传入请求的动态批处理。
- **张量并行：** 内置支持张量并行，用于在多个GPU上服务特大型模型。
- **量化 (quantization)支持：** 与bitsandbytes（用于NF4、FP4）和GPT-Q等量化格式集成，允许开箱即用地服务量化模型。
- **优化Transformer代码：** 采用FlashAttention和其他优化内核，用于流行的模型架构。
- **Safetensors和模型流：** 支持使用Safetensors进行安全模型加载，并可以流式传输模型权重 (weight)以实现更快启动。

在Hugging Face生态系统内工作时，需要支持多种模型、量化和张量并行的情况下，TGI是一个有力选择。

#### NVIDIA Triton Inference Server与FasterTransformer

NVIDIA Triton Inference Server是一款更通用的推理服务软件，可以部署来自各种框架（TensorFlow、PyTorch、TensorRT、ONNX）的模型。对于LLM，其能力通常通过其**FasterTransformer后端**发挥。FasterTransformer是NVIDIA为基于Transformer的模型提供的高度优化库，它为编码器和解码器层提供了高效实现。
主要特点：

- **高性能：** 使用FasterTransformer进行优化的LLM执行。
- **动态批处理：** Triton本身管理传入请求的动态批处理。
- **并发模型执行：** 可以在可用GPU上并发服务多个模型或同一模型的多个实例。
- **模型集成和流水线：** 支持复杂的推理工作流。
- **灵活性：** 支持广泛的模型格式和硬件后端。

对于需要服务多种模型类型或需要LLM服务以及模型集成等高级功能的环境，Triton与FasterTransformer是一个有力选择。与vLLM或TGI等更专业的LLM服务器相比，它通常需要更多的配置工作。

简化比较：

| 特性 | vLLM | Text Generation Inference (TGI) | Triton with FasterTransformer |
| --- | --- | --- | --- |
| **主要侧重** | 最大吞吐量，KV缓存效率 | 易用性，HF生态系统，功能 | 通用性，高性能 |
| **主要创新** | PagedAttention | 集成量化，张量并行 | FasterTransformer内核 |
| **批处理** | 连续 | 连续 | 动态 |
| **模型并行** | 有限（侧重单GPU优化） | 张量并行 | 张量和流水线（通过FT） |
| **量化** | 新兴支持 | bitsandbytes, GPT-Q | FP8, INT8 (通过FT) |
| **生态系统** | 以Python为中心，集成度不断提升 | Hugging Face | NVIDIA，多框架 |

此表格提供了一个高级概览。最佳选择在很大程度上取决于您的RAG系统所用的具体LLM、硬件和性能目标。

### RAG中LLM服务集成架构模式

在分布式RAG系统中，LLM服务组件通常部署为独立的、可扩展的服务。

> 一种典型架构，其中RAG组件与专用、负载均衡的LLM服务集群交互。编排器管理流程，将格式化提示发送到LLM集群并接收生成的文本。

此模式的组成：

1. **专用LLM服务：** LLM推理 (inference)引擎（vLLM、TGI、Triton等）作为独立的集群服务运行。这种关注点分离允许检索和生成组件的独立扩展和优化。
2. **负载均衡器：** 负载均衡器将来自RAG编排器的传入生成请求分发到可用的LLM服务器实例。这对于处理高请求量和确保高可用性是必不可少的。
3. **自动扩展：** LLM服务集群应配置为根据GPU利用率、请求队列长度或平均延迟等指标自动扩展。这确保了资源按需供应，兼顾性能和成本优化。云平台为容器化服务（例如，在Kubernetes上）提供自动扩展能力。
4. **API契约：** RAG编排器与LLM服务层之间明确的API契约（例如，REST或gRPC）很重要。这通常模仿OpenAI等流行API以方便集成，或使用优化二进制格式以提升性能。

这种集中式服务模型因其可管理性、可扩展性和资源池化优势，通常在大型分布式RAG中受到青睐。

### 多LLM考量

本章导言也提及多LLM架构。所讨论的服务框架（vLLM、TGI、Triton）能够同时托管多个不同模型，无论是在同一GPU上（如果内存允许且它们被多路复用）还是在同一服务器或集群内的不同GPU上。一个智能路由层（可能作为RAG编排器的一部分，或是一个专用的路由服务）会根据任务、用户、成本或其他标准，决定查询哪个具体的LLM（或LLM实例）。此处描述的高效服务架构，通过确保每个模型实例得到最优服务，为使此类多LLM策略在大规模下可行奠定了基础。

选择LLM服务架构和框架是构建大型RAG系统的一个重要决定。通过理解潜在难题并借助vLLM或TGI等专业工具，您可以构建一个不仅强大而且高效、可扩展的生成层，准备好满足生产环境的需求。本章稍后的动手实践将提供一个直接操作微调 (fine-tuning)的机会，其产出随后使用这些服务方案进行部署。

## 参考资料

- [vLLM: Efficiently Serve LLMs with PagedAttention](https://doi.org/10.48550/arXiv.2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了PagedAttention以实现高效的KV缓存管理和高LLM服务吞吐量。
- [Text Generation Inference: A production-ready inference container for LLMs](https://huggingface.co/blog/text-generation-inference) — Maxime Labonne, Thomas Simonini, and Pedro Cuenca (2023)
  Journal: Hugging Face Blog; Publisher: Hugging Face
  Hugging Face TGI的概述，涵盖了LLM部署的连续批处理和量化等功能。
- [Accelerating Large Language Model Inference with NVIDIA FasterTransformer and Triton Inference Server](https://developer.nvidia.com/blog/accelerating-large-language-model-inference-with-nvidia-fastertransformer-and-triton-inference-server/) — Apoorv Bansal, Bin Zhang, Bo Li, Chintan Shah, David W. Ho, Hanwen Chang, Jie Ren, Jike Li, Kuan Wang, Long Lu, Luyang Liu, Pranav Kashyap, Santhosh Tumma, Shuo Yang, Xiaodi Lu, Yichi Zhang, Yiran Shao, Yongmin Li, Zhongshuai Wang, and Zhen Jia (2023)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  解释了如何使用NVIDIA的FasterTransformer后端与Triton服务器进行优化的LLM服务。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍了FlashAttention，一种内存高效的注意力算法，用于优化基于Transformer的模型。
