---
course: "langchain-production-llm"
chapter: "optimizing-scaling-langchain"
lesson: "identifying-performance-bottlenecks"
sourceId: 3917
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/identifying-performance-bottlenecks"
title: "查明性能瓶颈"
description: "用于性能分析LangChain应用以找出LLM调用、数据检索或自定义逻辑中延迟问题的方法。"
order: 1
plots: ["plots/3917-0.json"]
sourceHash: "5e8a5e187bc83f40a3ecead0501099f2bcbd7da75ee2d8a2afadfdc5b4b2747b"
sourceCorrections: []
---

优化LangChain应用始于清楚了解时间和资源花在哪里。性能瓶颈可能存在于系统的多个部分，从基本的LLM交互到精细的自定义逻辑。猜测效率低下；系统性地查明对于有效调整非常必要。要确定链和代理中的这些性能限制，需要特定方法。

在尝试任何优化之前，你必须先进行测量。如果没有具体数据，改善性能的工作可能会误入歧途，可能只关注影响最小的区域，甚至带来新问题。目标是找出对整体延迟或资源占用贡献最大的组件或步骤。

### 性能分析工具与技术

标准Python性能分析工具提供一个起点。`cProfile` 等模块可以为你的应用程序代码提供函数级别的计时信息。

```python
import cProfile
import pstats
from io import StringIO
# 假设 'my_chain' 是你的 LangChain 可运行组件
# 并且 'input_data' 是输入字典

profiler = cProfile.Profile()
profiler.enable()

# 执行你的 LangChain 逻辑
result = my_chain.invoke(input_data)

profiler.disable()
s = StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
stats.print_stats(20) # 打印累计耗时最多的前20个函数

print(s.getvalue())
```

虽然对分析你的自定义Python函数有用，但标准性能分析器通常将LangChain组件调用（如LLM请求或检索器查询）视为单一、不透明的操作。它们可能会显示 `.invoke()` 或 `.ainvoke()` 调用很慢，但没有说明 *原因*。

针对LangChain的细致分析，**LangSmith** 不可或缺。LangSmith提供LangChain执行的详细追踪，直观呈现内部操作的顺序和持续时间。链或代理执行中的每一步，包括LLM调用、工具使用、检索器查询和解析器操作，都记录了计时信息。分析这些追踪通常是查明LangChain框架本身瓶颈的最直接方式。

### 常见瓶颈位置

LangChain应用中的性能问题通常出现在几个常见方面：

1. **LLM交互：**

   - **API延迟：** 网络往返时间和LLM提供商的推理 (inference)时间相结合，可能很多。这通常是最主要因素。LangSmith追踪清楚显示每个 `LLM` 或 `ChatModel` 调用的持续时间。
   - **Token生成时间：** 生成长响应的模型自然会花费更长时间。时间通常随输出token数量而变化。
   - **顺序调用：** 需要按顺序进行多次LLM调用的链会固有地累积延迟。如果 `LLM_B` 有赖于 `LLM_A` 的输出，则总时间至少是 `latency(LLM_A) + latency(LLM_B)`。
2. **数据检索 (RAG)：**

   - **向量 (vector)数据库查询：** 查询向量存储所需的时间有赖于数据库实现、索引大小、查询复杂程度、过滤复杂程度和硬件资源。效率低下的索引或资源不足的硬件可能导致检索缓慢。
   - **文档加载/处理：** 如果在请求期间动态加载或处理文档，这会增加大量开销。这包括从源获取、文本分割以及计算嵌入 (embedding)（如果未预先计算）。
   - **检索逻辑：** 先进的RAG技术，如混合搜索（将向量搜索与关键词搜索结合）或带重新排序的多阶段检索，增加了计算步骤，每个都增加了整体延迟。LangSmith追踪通常将检索器执行分解为组成部分，有助于分离慢速步骤。
3. **代理执行和工具使用：**

   - **工具执行时间：** 代理依赖工具与外部系统（API、数据库等）交互。工具调用的慢速外部API将直接影响代理响应时间。
   - **决策开销：** 采用ReAct或Plan-and-Execute等框架的复杂代理涉及多次LLM调用，用于推理、规划和观察处理。每一步都会增加延迟。
   - **效率低下的循环：** 如果代理的逻辑或工具不够强，它可能会陷入重复循环或错误处理循环中，显著增加执行时间。
4. **自定义组件和逻辑：**

   - **解析器：** 复杂的输出解析器，特别是那些涉及重试或额外LLM调用以进行纠正的，可能会引入延迟。
   - **自定义可运行组件/函数：** 任何集成到你的链中的自定义Python代码（`RunnableLambda`、自定义类）都可能成为瓶颈源，如果它执行慢速I/O操作、复杂计算或效率低下的数据操作。标准Python性能分析在此处有用。

### 使用追踪直观呈现瓶颈

考虑一个简化的RAG代理执行流程示例：

> 一个简化的RAG代理流程。像 `Retrieve Docs`（如果向量 (vector)存储慢）或 `Generate Response`（如果LLM调用慢或生成大量token）这样的步骤是常见瓶颈，此处用虚线边框呈现。LangSmith追踪为每个框提供了实际的计时数据。

LangSmith追踪提供了一个类似于此图的具体视图，但带有每一步的精确计时信息。通过查看追踪，你可以快速查看执行图中的哪些节点（组件）耗时最长。查找与其他步骤相比持续时间过长的步骤。

### 定量分析与基准

虽然单个追踪对调试特定请求有用，但了解整体应用性能需要定量分析。

- **汇总指标：** 使用LangSmith或自定义日志记录来收集统计数据，如平均延迟、中位延迟和尾部延迟（例如 P95、P99），针对整个应用程序和核心内部组件（LLM调用、检索器）。高尾部延迟通常表明间歇性但主要的性能问题。
- **相关性：** 分析性能是否根据输入特性而变化。例如，检索延迟是否随查询复杂程度显著增加？LLM响应时间是否与请求的输出长度密切相关？
- **组件延迟分布：** 直观呈现不同组件的延迟分布可以显示哪些部分持续地对总时间贡献最大。



![组件延迟分布 (ms)](plots/3917-0.json)



> 不同LangChain组件在多个请求中的延迟分布示例。对数y轴有助于直观呈现变化。此处，LLM调用表现出高的中位延迟和主要变异，而检索器也显示出偶尔的高延迟异常值（尾部延迟）。解析器始终很快。

在实施更改之前，建立这些 **基准指标**。记录你的应用程序在典型负载下的当前性能特性。任何未来的优化工作都应以此基准进行衡量，以确认其有效性。

查明性能瓶颈是一个迭代过程。当你优化一个方面时，另一个方面可能会成为新的限制因素。持续监控和定期性能分析是必要的，以便在应用程序演进和用户负载变化时保持性能。清楚了解延迟发生的位置，你就可以接着应用具体的优化技术，这些技术在接下来的部分中介绍。

## 参考资料

- [\`cProfile\` and \`pstats\` - The Python Profilers](https://docs.python.org/3/library/profile.html) — Python Software Foundation (2024)
  Python内置性能分析模块的官方文档，对分析应用程序代码性能有帮助。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2024)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  该综述文章描述了RAG系统的架构和组成部分，有助于识别性能敏感区域。
- [Site Reliability Engineering: How Google Runs Production Systems](https://sre.google/sre-book/table-of-contents/) — Niall Richard Murphy, Betsy Beyer, Chris Jones, and Jennifer Petoff (2017)
  Publisher: O'Reilly Media
  涵盖监控、指标和管理系统性能以建立基线和分析问题的原则资源。
