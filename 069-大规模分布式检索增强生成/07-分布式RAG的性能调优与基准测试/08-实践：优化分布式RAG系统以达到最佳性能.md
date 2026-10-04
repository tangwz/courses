# 实践：优化分布式RAG系统以达到最佳性能

来源：[原文](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-7-performance-tuning-benchmarking-distributed-rag/practice-optimize-distributed-rag-performance)

[返回章节目录](README.md) · [返回课程目录](../README.md)

一个调优复杂分布式检索增强生成（RAG）系统的模拟过程提供指导。我们将假定一个已有的架构，使用常见的可观测性信号诊断性能问题，并应用有针对性的优化。我们的目标是显著提升端到端延迟（$L_{total}$）和可持续每秒查询数（QPS），使系统达到生产可用状态。

### 阶段1：系统架构和基线性能

让我们思考一个包含以下组件的分布式RAG系统，如下图所示。这种架构常用于处理大规模请求。

> 正在优化的分布式RAG系统的高层架构。与分布式缓存的交互发生在多个阶段。

**初始性能评估**

部署此系统并使用`k6`或`Locust`等工具运行初始负载测试后，我们在50 QPS的中等负载下观察到以下基线性能特征：

- **P50 延迟 ($L_{total}$)**: 1500 毫秒
- **P90 延迟 ($L_{total}$)**: 5000 毫秒
- **P99 延迟 ($L_{total}$)**: 15000 毫秒
- **最大可持续QPS (在错误率 $E$ > 5%之前)**: 60 QPS
- **在50 QPS时的错误率 $E$**: 2%

这些数字远未达到我们的目标，即P99 $L_{total} < 2000$ 毫秒和可持续QPS > 500。我们的任务是找出并解决瓶颈。

### 阶段2：迭代优化周期

我们现在将进行几个瓶颈识别、解决方案实施（模拟）和影响评估的周期。假设我们有一个监控堆栈（Prometheus, Grafana, Jaeger），提供详细的遥测数据。

**周期1：处理检索延迟**

- **识别瓶颈**:

  - Jaeger对高延迟请求的追踪持续显示，`VectorDB -> ReRanker -> Orchestrator`的时间段贡献了P99延迟的60%以上。
  - Prometheus针对分片向量 (vector)数据库（Milvus）的指标表明，一两个分片的CPU利用率和查询队列显著高于其他分片，这说明存在热点分片问题或分段合并效率低下。
  - 重排序服务，使用BERT-large交叉编码器，显示出高单请求处理时间（50个候选约800毫秒），即使有自动扩缩容，其所有Pod的CPU利用率也接近100%。
- **假设与实施解决方案**:

  1. **向量数据库优化**:
     - **操作**: 重新评估Milvus集合的分片策略。如果使用基于实体ID的分片，检查是否存在倾斜分布。考虑实现一致性哈希或手动重新分发段。
     - **操作**: 分析Milvus查询执行计划。根据特定数据集的召回率/性能权衡，调整HNSW索引参数 (parameter)（例如，`efConstruction`，`efSearch`）或IVF\_FLAT/PQ参数（`nprobe`）。如果召回率有问题，可以稍微增加`search_k`，但要监控对延迟的影响。
     - **操作**: 在编排器或检索服务内实现一个小型本地缓存（例如，Java中的Caffeine，或Python中的内存LRU），如果适用，用于非常常见、基数低的查询，尽管这对于通用RAG来说不太常见。
  2. **重排序器优化**:
     - **操作**: 如果质量下降可以接受，量化 (quantization)交叉编码器模型（例如，使用ONNX Runtime或TensorRT进行int8量化）。
     - **操作**: 如果初始检索质量足够高，例如对于前200个候选，但只重排序前50个，则在重排序器*之前*实现智能候选截断。
     - **操作**: 为重排序器引入结果缓存，以输入文档ID的哈希值为键。如果相同的文档集经常因类似查询而进行重排序，这会有所帮助。
     - **操作**: 增加重排序服务副本的数量，并确保Kubernetes水平Pod自动扩缩器（HPA）配置了适当的CPU/内存目标。
- **评估影响（模拟）**:

  - 经过这些更改，检索+重排序阶段的P99延迟从约9000毫秒降至约2500毫秒。总体P99 $L_{total}$现在约为8500毫秒。这是一个显著的改进，但还需要更多。

**周期2：优化LLM推理 (inference)**

- **识别瓶颈**:

  - 随着检索速度加快，Jaeger追踪现在突出显示LLM服务是剩余延迟的主要因素，贡献了P99 $L_{total}$的约5000毫秒。
  - 来自DCGM（通过Prometheus）的vLLM集群GPU利用率指标显示出波动性的利用率，有时尽管有大量积压请求，利用率仍较低。`gpu_cache_usage_perc`（KV缓存）很高，但对于A100s/H100s上的Llama-3-70B来说，`avg_prompt_processing_time`和`avg_generation_time`每token高于预期。
- **假设与实施解决方案**:

  1. **vLLM配置调优**:
     - **操作**: 优化vLLM的连续批处理。调整`max_num_batched_tokens`（例如，4096，8192，取决于GPU显存 (VRAM)和典型上下文 (context)长度）和`max_num_seqs`（例如，256）以提高批处理密度并减少调度开销。
     - **操作**: 如果尚未启用，请确保为每个模型实例的多GPU推理正确设置了`tensor_parallel_size`。
     - **操作**: 尝试`max_model_len`以平衡上下文窗口需求与KV缓存压力。如果提示经常超过特定长度，可能导致过度的分页或重新计算。
  2. **LLM响应缓存**:
     - **操作**: 在分布式缓存（Redis）中为LLM响应实现语义缓存。标识符可以是（查询+Top-N检索到的上下文块）的哈希值。这需要仔细考虑缓存命中率和语义相似性阈值（如果不是精确匹配）。这可以显著减少重复或非常相似的高级查询的LLM调用。
  3. **LLM扩容与替代方案（高级）**:
     - **操作**: 扩增vLLM副本数量。根据GPU利用率或进行中的请求配置HPA。
     - **操作**: 如果RAG系统处理多种查询类型，考虑多LLM策略（如第三章所述）。将更简单的查询或需要较少推理的查询路由到更小、更快或量化程度更高的LLM（例如，Llama-3-8B-Instruct-quantized）。这需要在编排器中有一个智能路由层。
- **评估影响（模拟）**:

  - 调优vLLM并实现一个基本的精确匹配LLM响应缓存（由于热门话题，它出人意料地命中了15%的请求）将P99 LLM延迟降低到约1500毫秒。总体P99 $L_{total}$现在约为4000毫秒。

**周期3：编排与系统整体改进**

- **识别瓶颈**:

  - 编排服务本身现在显示出一些开销。尽管单个组件调用更快，但`检索 -> 重排序 -> 生成`的顺序性质仍然累加。日志显示编排器在等待下游调用时存在空闲时间。
  - 在更高的模拟负载下（接近200 QPS），API网关开始报告503错误，并且Redis集群缓存访问延迟增加。
- **假设与实施解决方案**:

  1. **编排器并行化**:
     - **操作**: 寻找并行执行的机会。例如，如果使用多查询检索或子查询生成，这些可以并行化。
     - **操作**: 对于某些RAG模式，如HyDE（文档嵌入 (embedding)），初始LLM生成答案的调用可以与初始关键词搜索并行进行，结果稍后合并。
  2. **缓存策略改进**:
     - **操作**: 确保所有缓存条目有适当的TTL（生存时间）。如果适用，实现分层缓存（例如，服务Pod中的L1内存缓存，L2分布式Redis）。
     - **操作**: 对于向量数据库，调查底层引擎（例如，Milvus）是否提供查询结果缓存或者代理层是否可以为非常热门的文档集提供此功能。
  3. **负载均衡与连接池**:
     - **操作**: 审查所有服务的负载均衡策略（例如，轮询，最少连接）。确保客户端连接池针对Redis和LLM端点等服务进行了优化，以减少连接建立开销。
     - **操作**: 确保API网关和底层Kubernetes Ingress控制器针对目标QPS进行了充分的资源配置和设置。
- **评估影响（模拟）**:

  - 编排器中的少量并行化和优化的连接池从P99中削减了300-500毫秒。积极的缓存现在处理了更大比例的流量。总体P99 $L_{total}$现在约为2500毫秒。由于缓存和更快的组件，最大QPS显著增加。

### 阶段3：系统整体重新基准测试和压力测试

在这些迭代优化之后，我们进行一次全面的重新基准测试运行。



[交互图表：端到端延迟 ($L_{total}$) 对比](https://apxml.com/zh/courses/large-scale-distributed-rag/chapter-7-performance-tuning-benchmarking-distributed-rag/practice-optimize-distributed-rag-performance#plot-1sch2zg)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "P50",
        "P90",
        "P99"
      ],
      "y": [
        1500,
        5000,
        15000
      ],
      "name": "优化前",
      "marker": {
        "color": "#ff8787"
      }
    },
    {
      "type": "bar",
      "x": [
        "P50",
        "P90",
        "P99"
      ],
      "y": [
        180,
        450,
        1300
      ],
      "name": "优化后",
      "marker": {
        "color": "#69db7c"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "端到端延迟 ($L_{total}$) 对比"
    },
    "barmode": "group",
    "xaxis": {
      "title": {
        "text": "延迟百分位数"
      }
    },
    "yaxis": {
      "title": {
        "text": "延迟 (毫秒)"
      },
      "type": "log",
      "dtick": 1
    },
    "height": 450,
    "width": 700,
    "font": {
      "family": "Helvetica, sans-serif"
    }
  }
}
```

</details>



> P50、P90和P99端到端延迟在优化周期前后的对比。注意Y轴的对数刻度，它突出了所有百分位数的显著改进。

**新性能指标:**

- **P50 延迟 ($L_{total}$)**: 180 毫秒
- **P90 延迟 ($L_{total}$)**: 450 毫秒
- **P99 延迟 ($L_{total}$)**: 1300 毫秒 (达到目标 < 2000 毫秒)
- **最大可持续QPS (在错误率 $E$ > 1%之前)**: 650 QPS (显著提升)
- **在500 QPS时的错误率 $E$**: 0.5%

**压力测试:**
接下来，通过逐渐增加负载到预期峰值来执行压力测试。监控：

- 各组件的延迟。
- 各服务的饱和点（CPU、GPU、内存、网络I/O）。
- 队列长度（例如，vLLM、向量 (vector)数据库中的）。
- 所有组件和API网关的错误率。
  这有助于识别在更高负载下可能出现的下一组瓶颈，并为容量规划提供信息（例如，“在1000 QPS时，重排序服务再次成为瓶颈，将需要X个额外的副本，并可能需要更高效的模型架构”）。

### 阶段4：成本效益分析

性能虽然重要，但成本对于生产系统来说同样重要。所应用的优化（例如，扩增LLM副本，为向量 (vector)数据库使用更大的实例，缓存基础设施）总是会影响运营开支。

- **追踪成本**:
  - 利用云服务提供商的成本管理工具（例如，AWS Cost Explorer，GCP Billing Reports）将成本与特定组件或服务关联起来（通常通过标记 (token)Kubernetes资源）。
- **每美元性能**:
  - 计算每小时每美元QPS（QPS/\/hr\$）或每百万查询成本等指标。
- **权衡**:
  - 分析权衡。例如，更昂贵、更大的LLM可能提供更高质量的响应，并减少对复杂重排序或多跳逻辑的需求，潜在地简化系统并降低其他地方的成本。反之，激进的量化 (quantization)和更小的模型可能降低推理 (inference)成本，但需要更复杂的检索来保持质量。
- **自动扩缩容优化**:
  - 精细调整HPA配置和集群自动扩缩器设置，以便资源分配与实际需求紧密匹配，最大限度地减少空闲资源，同时确保在峰值负载期间达到性能SLA。考虑使用KEDA等解决方案，根据队列长度或其他自定义指标进行事件驱动的扩缩容。

### 总结

优化大规模分布式RAG系统是一个迭代且持续的过程。本实践练习模拟了一个典型的工作流程，从基线评估开始，经过不同子系统（检索、重排序、LLM生成、编排）的有针对性的瓶颈解决，最终进行系统整体压力测试和成本考量。

主要在于以下几点：

1. **全面可观测性**：详细的指标、追踪和日志对于确定工作重点是必不可少的。
2. **系统化方法**：一次处理一个瓶颈，评估影响，然后迭代。
3. **理解组件交互**：一个区域的优化可能将瓶颈转移到另一个区域。整体视角非常重要。
4. **策略性缓存**：多层缓存通常是改善延迟和吞吐量 (throughput)，同时减少昂贵计算资源负载的最有效方法之一。
5. **持续调优**：随着数据规模扩大、查询模式变化和模型演变，定期重新评估和调优将是维持最佳性能和成本效益的必要条件。

这种结构化方法，结合对RAG流程中每个组件和底层分布式系统基础设施的渊博专业知识，对于构建和运行大规模高性能RAG解决方案至关重要。

## 参考资料

- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  一本关于分布式系统原理的综合性书籍，涵盖了数据存储、处理以及构建可扩展和高性能应用程序所涉及的权衡。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023)
  Journal: SOSP 2023; DOI: [10.48550/arXiv.2309.06180](https://doi.org/10.48550/arXiv.2309.06180)
  介绍了 vLLM 和 PagedAttention，这是一种通过高效 KV 缓存管理和持续批处理优化大型语言模型服务的技术，与大型语言模型推理优化直接相关。
- [Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs](https://doi.org/10.1109/TPAMI.2018.2889473) — Yu. A. Malkov, D. A. Yashunin (2018)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 42; Pages: 824-836; DOI: [10.1109/TPAMI.2018.2889473](https://doi.org/10.1109/TPAMI.2018.2889473)
  这篇论文描述了分层可导航小世界图（HNSW），这是一种在向量数据库中广泛采用的近似最近邻搜索算法。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2024)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  提供了检索增强生成（RAG）系统、其架构、当前挑战以及各种优化技术的概述，包括性能考量。

---

[上一节](07-%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83%E4%B8%AD%E7%9A%84%E6%80%A7%E8%83%BD%E5%88%86%E6%9E%90%E4%B8%8E%E8%B0%83%E8%AF%95.md)
