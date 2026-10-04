---
course: "agentic-llm-memory-architectures"
chapter: "evaluation-optimization-agentic-systems"
lesson: "optimization-techniques-agent-performance"
sourceId: 4545
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-6-evaluation-optimization-agentic-systems/optimization-techniques-agent-performance"
title: "代理系统性能优化方法"
description: "提高代理系统速度、降低成本和增强稳定性的策略。"
order: 7
plots: ["plots/4545-0.json"]
sourceHash: "e1f7b33bc643e88d997e0c1d6e07d134fc4a5cb07d5e8723b9c8e312eb68237d"
sourceCorrections: []
---

在讨论过的评估和调试阶段之后，重心转向提升代理系统的运行表现。优化不只是关于速度；它包含提升成本效益、增强稳定性和保证代理在不同条件下的良好运行。这通常需要在这些因素之间进行权衡，并以您应用的具体需求为导向。

### 提升性能的提示工程 (prompt engineering)

提示的结构和内容直接影响代理的性能和成本。LLM 处理的每个 token 都会增加延迟和开销。

- **简洁性：** 去除提示中重复的指令、示例或对话填充内容。对于像 ReAct 这样的复杂推理 (inference)链，分析生成的思考内容。能否让推理步骤更直接，同时不降低准确性？有时，指导模型在其内部独白中更精炼会带来好处。
- **少样本示例优化：** 尽管少样本示例能提高准确性，但它们会增加提示的长度。仔细筛选示例。使用信息量最大、最具代表性的示例。尝试减少示例的数量或长度。
- **指令微调 (fine-tuning)：** 如果您有能力微调模型，专门针对鼓励更短、更直接的响应或推理步骤的指令格式进行调整，与仅依赖通用模型进行复杂提示工程相比，可以带来显著的性能提升。

### 模型选择、量化 (quantization)和剪枝

选择底层 LLM 是一个重要的优化决策。

- **模型规模调整：** 更大的模型功能更强，但会带来更高的延迟和成本。评估较小的、可能经过微调 (fine-tuning)的模型是否能充分完成代理中的特定任务或子任务。例如，复杂的规划步骤可能需要一个前沿模型，但简单的工具选择或响应格式化则可能使用一个小得多、速度更快的模型。
- **量化：** GPTQ（通用 Transformer 量化）、AWQ（激活感知权重 (weight) 量化）或 GGUF（由 llama.cpp 使用）等方法降低模型权重的精度（例如，从 16 位浮点数到 8 位或 4 位整数）。这显著减少模型大小和内存带宽需求，从而加快推理 (inference)速度并降低 VRAM 使用量，特别是在边缘设备或消费级 GPU 上。然而，量化可能导致较小的精度下降，这需要根据您的具体应用场景进行评估。权衡通常在于性能提升与可接受的质量损失之间。
- **剪枝：** 与量化相比，模型剪枝在直接应用部署中较不常见，它涉及从网络中移除不重要的权重或结构。这也能减小模型大小和计算量，但通常需要重新训练或大量微调以恢复损失的性能。



![模型性能 vs. 成本](plots/4545-0.json)



> 较小的模型通常具有较低的延迟和成本，但与更大、更昂贵的模型相比，其能力可能会有所降低。选择合适的模型需要在性能需求和预算限制之间取得平衡。

### 缓存机制

通过在不同层面实现缓存，避免冗余计算和 API 调用。

- **LLM 响应缓存：** 对于相同的输入（提示、历史记录），缓存生成的 LLM 响应。这对于确定性任务或常见问题非常有效。
- **嵌入 (embedding)缓存：** 如果您的代理使用检索增强生成（RAG）处理静态或缓慢变化的文档，请缓存文档块的嵌入。这避免了每次需要检索时进行昂贵的重新计算。
- **工具结果缓存：** 缓存确定性工具调用的输出（例如，使用相同参数 (parameter)调用的计算器函数）。对访问易变数据（例如，实时股票价格）的工具要谨慎。
- **中间步骤缓存：** 在多步骤推理 (inference)过程（ReAct、ToT）中，如果代理可能重复访问类似状态，则缓存中间思考或子问题的结果。

有效的缓存失效策略很重要，以确保当底层数据或状态发生变化时，代理不会依赖过期信息。

### 优化推理 (inference)和规划循环

代理行为的核心循环通常涉及多次 LLM 调用。减少这些调用的数量或成本意义重大。

- **减少 LLM 调用：** 分析代理的决策过程。某些选择能否通过启发式方法或更简单的基于规则的系统来完成，而不是通过完整的 LLM 调用？例如，简单的输入验证或在两个预定义工具之间进行选择可能不总是需要复杂的推理。
- **并行执行：** 识别计划中的独立步骤或并行工具执行的机会。如果代理需要调用两个不同的 API，且它们的输入互不依赖输出，则并行执行这些调用以减少实际耗时。Python 的 `asyncio` 很适合这种场景。
- **优化思考生成：** 在 ReAct 风格的代理中，引导模型生成简洁的思考或为思考和行动提供结构化格式（如 JSON），这有时比自由格式文本处理起来更高效、更稳定。
- **提前退出：** 如果代理的任务允许在完全考虑所有可能性之前找到可接受的解决方案（在思考树或搜索算法中常见），一旦找到足够好的解决方案，就实施提前终止的条件。

> 代理执行流程图，凸显了从初始计划衍生的独立工具调用（步骤 1 和步骤 2）的潜在并行化。顺序执行需要等待每个步骤完成。

### 内存系统改进

高效的内存交互对性能很重要，尤其是在长时间运行的代理中。

- **检索优化：**
  - **索引：** 调整向量 (vector)索引的参数 (parameter)（例如，HNSW 中的 `ef_construction`、`ef_search`），以平衡搜索速度和检索准确性（召回率）。
  - **嵌入 (embedding)模型：** 尝试不同的嵌入模型。有些模型在速度/性能之间提供了更好的权衡，或针对特定领域进行了优化。考虑专为检索任务设计的模型。
  - **重排序：** 使用轻量级交叉编码器模型对初始向量搜索的前 k 个结果进行重排序。这会增加一个小的计算步骤，但可以显著提高传递给 LLM 的最终文档的相关性，可能允许使用更小的初始 `k` 并缩短上下文 (context)长度。
- **批处理操作：** 当向向量数据库或结构化内存（如知识图谱）读写多条信息时，将这些操作批量处理，而不是执行单个事务。这减少了网络开销，并可以利用优化的数据库操作。
- **高效摘要：** 如果使用内存摘要方法，优化其频率和方式。摘要过于频繁会增加计算开销；摘要过于不频繁可能导致重要细节丢失或上下文窗口过长。考虑分层摘要或有针对性的更新。

### 工具执行改进

与外部工具的交互可能成为瓶颈。

- **异步工具调用：** 使用异步编程（Python 中的 `asyncio`）来执行工具调用（尤其是网络绑定的 API 调用），而不会阻塞代理的主处理线程。这使得代理在等待工具响应时，有可能准备下一步或执行其他计算。
- **批量 API 请求：** 如果工具的 API 支持批量处理（例如，在一个请求中查询多个项目），请加以利用以减少网络往返次数。
- **优化工具选择：** 如果代理有很多工具，选择合适工具的过程可能会很耗时。实施高效的选择机制，例如使用工具描述的嵌入 (embedding)、基于任务类型进行预过滤，甚至使用一个专门用于工具路由的较小型 LLM。

### 基础设施和部署考量

底层基础设施对性能有重要作用。

- **硬件加速：** 利用 GPU 或 TPU 进行 LLM 推理 (inference)。专用 AI 加速器与 CPU 相比能提供数量级的速度提升。
- **推理服务器/框架：** 使用优化的推理服务器，如 Nvidia Triton Inference Server 或 vLLM、TensorRT-LLM、TGI（文本生成推理）等框架。它们提供连续批处理、分页注意力、优化内核等功能，显著提高吞吐量 (throughput)并降低延迟，特别是在并发请求负载下。
- **部署模型：**
  - **无服务器：** 函数即服务（FaaS）对于偶发性工作负载而言可能具有成本效益，但可能会出现冷启动问题，增加首次请求的延迟。
  - **专用实例：** 提供持续的低延迟，但会产生持续的成本。自动扩缩配置有助于平衡成本和性能。

优化代理系统需要一个全面的方法，考虑提示、模型、算法、内存、工具和基础设施。这是一个迭代过程，由细致的评估以及对速度、成本和代理输出质量之间权衡的清晰认识来引导。您的具体应用需求将决定哪些方法能带来最显著的好处。

## 参考资料

- [Scaling Instruction-Finetuned Models](https://arxiv.org/abs/2210.11416) — Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, Jason Wei (2022)
  Journal: arXiv preprint arXiv:2210.11416; DOI: [10.48550/arXiv.2210.11416](https://doi.org/10.48550/arXiv.2210.11416)
  本文介绍了FLAN-PaLM，并讨论了指令微调对LLM性能和效率的影响，这是提示工程中一项重要的技术。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://doi.org/10.48550/arXiv.2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  提供了RAG技术的全面概述，涵盖了检索优化、嵌入模型和重排等方面，这些都是增强记忆系统的重要部分。
