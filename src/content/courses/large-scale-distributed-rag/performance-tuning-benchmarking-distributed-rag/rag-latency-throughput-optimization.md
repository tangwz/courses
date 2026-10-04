---
course: "large-scale-distributed-rag"
chapter: "performance-tuning-benchmarking-distributed-rag"
lesson: "rag-latency-throughput-optimization"
sourceId: 6250
sourceUrl: "https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-7-performance-tuning-benchmarking-distributed-rag/rag-latency-throughput-optimization"
title: "延迟与吞吐量优化技术"
description: "用于优化分布式RAG系统中延迟和吞吐量的高级方法。"
order: 2
plots: ["plots/6250-0.json"]
sourceHash: "646102de38bb4cda9ab8ef24a673d1676d373bb84cb4e18ef871c619b7d60d70"
sourceCorrections: []
---

分布式RAG系统中的性能工程有两个主要目标：最小化端到端延迟（$L_{total}$）和最大化系统吞吐量 (throughput)（每秒查询数，或$QPS$）。这些目标通常相互制约，要达到最佳平衡，需要对系统行为以及调整每个组件的可用具体方法有全面理解。提供增强延迟和吞吐量的策略，使您的RAG系统能够满足严格的性能要求。

### 延迟（$L_{total}$）优化

RAG系统中的端到端延迟是各个阶段延迟的总和。一个典型的分解可能包括：
$L_{total} = L_{query\_preprocess} + L_{retrieval} + L_{reranking} + L_{context\_assembly} + L_{llm\_inference} + L_{network} + L_{orchestration}$

识别哪些部分对$L_{total}$的贡献最大是第一步，这通常通过细致的追踪和性能分析来实现。一旦主要瓶颈确定后，便可以应用有针对性的优化。

> RAG顺序流程中延迟构成部分的简化视图。网络和编排开销发生在各阶段之间。

#### 降低检索延迟（$L_{retrieval}$）的方法

检索阶段，特别是涉及在海量索引上进行密集向量 (vector) (dense vector)搜索时，可能是主要的延迟贡献者。

- **ANN参数 (parameter)调整**：对于近似最近邻（ANN）搜索算法，如HNSW或IVFADC，`ef_search` (HNSW) 或 `nprobe` (IVF) 等参数直接在搜索精度和速度之间进行权衡。系统地实验这些参数，为您的目标召回率找到可接受的最低延迟，这一点很重要。对于HNSW，在索引构建时增加`ef_construction`有时可以使图在查询时遍历更快，尽管前期成本更高。
- **索引结构优化**：对于基于IVFADC的索引，质心数量（Voronoi单元）和乘积量化 (quantization)程度（子向量数量，每个子向量的位数）都非常重要。质心太少会导致过大且扫描速度慢的倒排列表；质心太多则可能增加达到良好召回率所需的`nprobe`。类似地，更细粒度的量化可以减少索引大小和扫描时间，但可能降低精度。
- **查询端嵌入 (embedding)优化**：查询嵌入自身的生成会增加延迟。如果使用复杂模型进行查询嵌入，请确保它们得到高效服务，例如通过专用的优化推断服务器。对于某些应用，更简单或经过蒸馏的查询编码器可能会提供有利的延迟/性能权衡。
- **分片和复制策略**：尽管主要为了扩展性，但分片策略会影响延迟。查询必须路由到相关分片，或者扇出到所有分片，并聚合结果。这种扇出和聚合的开销直接增加了$L_{retrieval}$。有效的路由（例如，基于映射到特定分片的元数据过滤器）可以限制查询的分片数量。

#### 简化重排序（$L_{reranking}$）

重排序器通常是较小的Transformer模型，用于优化初始检索结果。它们的延迟影响不容忽视，特别是当它们处理大量候选时。

- **级联重排序器**：采用多阶段重排序流程。一个非常快的、可能是非神经网络 (neural network)的初始重排序器（例如，在全文片段上使用BM25，或如果尚未完成则在嵌入上进行简单点积）可以在更强大但更慢的神经网络重排序器应用于更小规模的文档集（例如，前50-100个而不是前1000个）之前，修剪候选集。
- **模型简化与量化**：与LLM一样，重排序模型可以被量化或剪枝。如果质量影响可接受，请考虑使用层数更少或隐藏维度更小的模型。
- **优化服务**：在专用的优化推断基础设施上服务重排序模型，如果模型复杂度需要，可以借助GPU加速。

#### 加速LLM推断（$L_{llm\_inference}$）

LLM推断通常是$L_{total}$中最大的单一贡献者。

- **连续批处理**：vLLM或Text Generation Inference (TGI) 等技术使用连续批处理（也称为动态批处理或飞行中批处理），通过在请求到达时和令牌生成时处理它们，而不是等待固定批次满或完成，从而大幅提升GPU利用率并降低每令牌延迟。
- **模型优化方法**：如第三章所述，量化（例如，INT8, AWQ, GPTQ）、剪枝和知识蒸馏 (knowledge distillation)是减少模型大小和计算需求的主要方法，从而加快推断速度。
- **优化内核**：使用优化的注意力机制 (attention mechanism)，如FlashAttention或内存高效的注意力变体，可以提供大幅加速，特别是对于长上下文 (context)。
- **推测解码**：该方法涉及使用一个更小、更快的“草稿”模型来预测多个未来的令牌，然后由更大、更准确的模型进行验证（或纠正）。如果预测通常是正确的，这可以大幅减少主LLM所需的正向传播次数。
- **硬件选择与配置**：选择合适的GPU（例如，具有更高内存带宽或专用张量核心的GPU），并确保GPU之间用于模型并行的NVLink等高带宽互连，对于最小化推断延迟很重要。

#### 最小化网络和编排开销（$L_{network}, L_{orchestration}$）

将分布式部分连接在一起的“粘合剂”也会引入延迟。

- **服务联合部署**：将相互通信的服务（例如，检索器、重排序器、LLM）部署在近网络距离（相同数据中心、可用区，甚至在合适时部署在相同主机上），以减少网络RTT。
- **高效序列化**：在服务间通信中，使用gRPC与Protocol Buffers或Apache Avro等高效二进制序列化格式，而不是JSON等基于文本的格式，特别是对于大型数据负载（例如，检索到的文档）。
- **异步执行和流水线**：设计工作流以并行执行独立操作。例如，如果从多个来源（例如，不同的向量索引或向量与关键词搜索的混合）检索，这些检索通常可以并发执行。在可能的情况下，使计算与I/O重叠。在服务逻辑中使用异步编程模式（例如Python中的`async/await`）以防止I/O操作阻塞。

### 吞吐量 (throughput)（QPS）优化

最大化QPS涉及同时处理更多查询，或减少系统范围内每个查询的平均处理时间，这通常通过提升资源利用率来实现。

#### 批处理的重要作用

在不同阶段进行请求批处理是提升吞吐量的基本方法，特别是对于神经网络 (neural network)模型推断而言。

- **检索批处理**：执行嵌入 (embedding)推断和向量 (vector)搜索的密集检索模型，从查询批处理中大幅受益。这使得在嵌入和初始搜索阶段可以更充分地使用GPU/CPU资源。
- **LLM批处理**：如前所述的用于降低延迟的连续批处理，这些系统通过最大化GPU占用率，本身就能提升吞吐量。对于LLM推断，在单次正向传播中处理多个输入序列（提示）可以减少开销，并发挥GPU的并行处理能力。
- **对资源利用率和延迟的影响**：更大的批处理量通常会提升吞吐量，直至硬件饱和。然而，非常大的批处理量可能会增加单请求延迟，因为单个请求可能需要等待更长时间才能被纳入完整批处理或等待批处理完成处理。



![批处理量对LLM推断的影响](plots/6250-0.json)



> 增加批处理量对LLM推断吞吐量和P95延迟影响的示例图。最佳批处理量能平衡这些相互冲突的指标。

#### 水平扩展和智能负载均衡

将负载分散到RAG的多个实例上对于高吞吐量很重要。

- **无状态部分扩展**：查询预处理器、重排序器和LLM推断端点等服务通常是无状态的，可以通过在负载均衡器后增加更多副本轻松进行水平扩展。
- **有状态部分策略**：对于分片向量数据库等有状态部分，扩展涉及增加更多分片和副本。负载均衡必须具备分片感知能力，或者查询必须适当扇出。
- **负载均衡算法**：选择适合您的RAG部分特点的负载均衡算法（例如，轮询、最少连接、延迟感知）。对于LLM推断，如果使用没有连续批处理的系统，请考虑能够考虑序列长度的请求感知均衡。

#### 资源管理和并发

高效利用可用硬件和管理并发操作很重要。

- **最大化GPU/CPU利用率**：对您的应用进行性能分析，以确保CPU密集型任务不会成为GPU密集型任务的瓶颈，反之亦然。例如，LLM的数据加载和预处理应该高度优化，以确保GPU得到充分利用。
- **管理并发请求**：配置服务器框架（例如，Python服务中的Uvicorn、Gunicorn），使其具有适当数量的工作进程或线程，以处理预期的并发负载，避免过多的上下文 (context)切换或资源争用。
- **连接池**：为数据库（向量数据库、元数据存储）和其他后端服务使用连接池，以避免为每个请求建立新连接的开销。根据预期的并发量调整连接池大小。

### 平衡：延迟、吞吐量 (throughput)和成本

优化延迟和吞吐量很少单独进行，通常涉及多重权衡，而成本则是始终存在的第三个维度。

- **内在权衡**：提升吞吐量的方法，如积极的批处理，可能会增加单个请求的平均延迟。反之，最小化每个请求的延迟（例如，使用非常小的批处理量或不批处理）会严重限制整体QPS。
- **系统特定的最佳点**：理想的平衡取决于应用。交互式聊天机器人要求低$L_{total}$，可能牺牲一些峰值QPS。批处理文档处理流程可能优先考虑QPS而非单个项目的延迟。
- **成本影响**：增加更多硬件（副本、更强大的GPU）可以提升延迟和吞吐量，但会增加运营成本。量化 (quantization)或更高效模型服务等优化可以提升现有硬件上的性能，从而更具成本效益。
- **适应性策略**：对于具有可变负载或不同类型请求的系统，请考虑适应性策略。这可能涉及资源的动态扩展，甚至将具有不同QoS要求的请求路由到不同配置的RAG流程（例如，用于高级用户的低延迟流程，用于后台任务的高吞吐量流程）。

分布式RAG中的性能调整是一个迭代过程。它需要持续监控、基准测试，并乐于尝试这些方法，以找到最符合您的延迟、吞吐量和成本特定服务水平目标（SLO）的配置。关于基准测试和识别瓶颈的后续部分将进一步帮助您进行这项持续的优化工作。

## 参考资料

- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://ieeexplore.ieee.org/document/8576974) — Yu. A. Malkov, D. A. Yashunin (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 42; Pages: 824-836; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  介绍了分层可导航小世界图（HNSW）算法，这是一种高效的近似最近邻搜索方法，对向量检索性能至关重要。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems; Volume: 35; Pages: 16347-16361; DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  提出了FlashAttention，一种I/O感知的精确注意力算法，通过减少内存访问开销显著加速了Transformer模型的推理速度，与LLM推理加速直接相关。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Yaniv Leviathan, Matan Kalman, Yossi Matias (2022)
  Journal: Proceedings of the International Conference on Machine Learning (ICML) 2023; DOI: [10.48550/arXiv.2211.17192](https://doi.org/10.48550/arXiv.2211.17192)
  介绍了推测解码，一种利用较小草稿模型加速大型Transformer模型推理的技术，从而降低了每令牌生成延迟。
