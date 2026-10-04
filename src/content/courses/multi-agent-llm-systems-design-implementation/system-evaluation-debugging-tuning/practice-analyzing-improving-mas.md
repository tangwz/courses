---
course: "multi-agent-llm-systems-design-implementation"
chapter: "system-evaluation-debugging-tuning"
lesson: "practice-analyzing-improving-mas"
sourceId: 6709
sourceUrl: "https://apxml.com/zh/courses/multi-agent-llm-systems-design-implementation/chapter-6-system-evaluation-debugging-tuning/practice-analyzing-improving-mas"
title: "实践：分析和优化多智能体系统示例"
description: "专注于日志分析、问题识别和示例多智能体系统优化的实践环节。"
order: 8
plots: ["plots/6709-0.json"]
sourceHash: "4c496a3cef2df04f15bc1ff4a674e342d1038b770ea9d205c1a37cf720c1af3e"
sourceCorrections: []
---

此动手练习引导您分析和优化一个多智能体 LLM 系统，即“自动化研究摘要系统”（ARDS）。本练习专注于诊断问题、实施优化并评估此类系统的结果。ARDS 旨在接收一个通用研究课题，将其分解，搜索相关的学术信息，并编制一份结构化摘要。

我们的 ARDS 由以下智能体组成：

1. **`QueryPlannerAgent`**：接收一个宽泛的研究课题，并将其分解为几个具体、可回答的问题。
2. **`ParallelSearchAgent`**：管理一个 `SearchSubAgent` 实例池。它将 `QueryPlannerAgent` 的具体问题分发给这些子智能体。
3. **`SearchSubAgent` （多个实例）**：每个实例接收一个问题，构建查询，并与外部（模拟）`AcademicSearchAPI` 交互以获取信息。
4. **`ContentSynthesizerAgent`**：收集来自不同 `SearchSubAgent` 的所有搜索结果，整合信息，并为每个初始问题准备连贯的答案。
5. **`ReportGeneratorAgent`**：接收整合后的内容，并将其格式化为最终的结构化研究摘要。

预期工作流程如下：

> 自动化研究摘要系统（ARDS）的工作流程，从用户输入到最终报告。

尽管有这样的设计，用户还是反映了一些问题：ARDS 经常缓慢，报告可能不完整，且运行成本（主要是 LLM 令牌使用量）高于预期。

### 1. 初始性能基线与日志分析

在进行更改之前，我们建立一个基线。假设我们已经在一些测试课题上运行了 ARDS 并收集了以下初始指标：

| 指标 | 优化前 | 目标 |
| --- | --- | --- |
| 平均任务完成时间 | 125 秒 | < 60 秒 |
| API 调用成功率 | 85% | > 99% |
| 平均 API 延迟（每次调用） | 8 秒 | < 4 秒 |
| 每份报告成本（令牌） | 150,000 | < 100,000 |
| 报告完整度评分 | 3/5（平均） | 5/5 |

对系统日志的检查显示出这样的模式：

```text
[2023-11-10 14:30:10] [QueryPlannerAgent] INFO: Topic "AI in Healthcare Diagnostics" decomposed into 5 questions.
[2023-11-10 14:30:11] [ParallelSearchAgent] INFO: Dispatched 5 search tasks to sub-agents.
[2023-11-10 14:30:15] [SearchSubAgent-3] INFO: Querying AcademicSearchAPI for "Ethical concerns of AI in diagnostics".
[2023-11-10 14:30:28] [SearchSubAgent-3] INFO: Received 7 results from API. Latency: 13250ms.
[2023-11-10 14:30:30] [SearchSubAgent-1] INFO: Querying AcademicSearchAPI for "Current AI algorithms for cancer detection".
[2023-11-10 14:30:45] [SearchSubAgent-1] ERROR: API call failed for "Current AI algorithms for cancer detection". Error: Connection Timeout. Attempt 1/1.
[2023-11-10 14:30:46] [SearchSubAgent-2] WARN: AcademicSearchAPI rate limit likely hit. Delaying next request.
...
[2023-11-10 14:32:05] [ContentSynthesizerAgent] WARN: Received results for only 3 out of 5 planned questions. Proceeding with available data.
[2023-11-10 14:32:15] [ContentSynthesizerAgent] INFO: Synthesis complete. Total input tokens: 95000. Output tokens: 8000.
```

从这些日志和指标，我们可以推断出：

- **API 交互**：`SearchSubAgent` 调用 `AcademicSearchAPI` 的速度较慢（例如，13250ms 延迟），且容易失败（超时、速率限制）。这直接影响总体任务完成时间和报告完整度。
- **错误处理**：`SearchSubAgent-1` 的错误表明只尝试了一次。失败没有重试，导致 `ContentSynthesizerAgent` 丢失数据。
- **令牌消耗**：`ContentSynthesizerAgent` 处理大量输入令牌（95,000），这表明原始、冗长的搜索结果被传递，导致高成本。

### 2. 诊断瓶颈和故障点

根据我们的初始分析，我们可以明确需要改进的具体方面：

- **瓶颈 1：`AcademicSearchAPI` 交互效率和可靠性**

  - **高延迟**：API 本身可能很慢，或者网络状况可能是影响因素。我们的查询也可能过于宽泛，返回过多数据。
  - **失败**：超时和速率限制没有得到稳固处理。这是报告不完整的主要原因。
- **故障点 1：`ParallelSearchAgent` 和 `SearchSubAgent` 缺乏弹性**

  - 当 `SearchSubAgent` 未能获取数据（例如，由于 API 错误）时，该信息会直接从最终报告中缺失。系统没有有效的重试机制，也无法向上游发出问题信号以采取替代措施。
- **成本驱动因素 1：`ContentSynthesizerAgent` 中数据处理和令牌使用效率低下**

  - `ContentSynthesizerAgent` 的高输入令牌计数表明它可能从搜索结果中接收到大量未经筛选的文本。这使得 LLM 的工作更困难，增加了处理时间，并显著提高了令牌成本。

### 3. 实施和评估改进措施

让我们制定策略来解决这些问题。

**改进 A：增强 `SearchSubAgent` 和 `AcademicSearchAPI` 的交互**

- **实施重试**：修改 `SearchSubAgent`，使用指数退避策略重试失败的 API 调用。例如，在重试之间等待 2 秒，然后 4 秒，然后 8 秒，最多尝试 3-5 次。这有助于克服短暂的网络问题或临时速率限制。

  ```python
  # SearchSubAgent API 调用的伪代码
  # async def call_academic_api(query, max_retries=3):
  #     delay = 2
  #     for attempt in range(max_retries):
  #         try:
  #             response = await actual_api_call(query)
  #             return response
  #         except APITimeoutError as e:
  #             log.warn(f"API 超时（尝试 {attempt+1}/{max_retries}）。在 {delay} 秒后重试...")
  #             await asyncio.sleep(delay)
  #             delay *= 2
  #         except APIRateLimitError as e:
  #             log.warn(f"API 速率限制（尝试 {attempt+1}/{max_retries}）。在 {delay*2} 秒后重试...") # 速率限制需要更长的延迟
  #             await asyncio.sleep(delay*2)
  #             delay *= 2
  #     log.error(f"对查询 {query} 的 API 调用在 {max_retries} 次尝试后失败")
  #     return None
  ```
- **引入缓存**：为 `AcademicSearchAPI` 响应实现缓存（例如，在此示例中使用内存字典，或在生产环境中使用 Redis 等持久存储）。如果在短时间内再次提出相同（或非常相似）的具体问题，则提供缓存结果以减少延迟和 API 负载。
- **优化查询提示**：优化 `SearchSubAgent` 用于构建查询的提示。指示智能体使查询更具体，或者如果 API 支持此类功能，则请求 API 返回摘要，从而减少传输和处理的数据量。

**改进 B：提升 `ParallelSearchAgent` 的弹性**

- **增强状态追踪**：`ParallelSearchAgent` 应主动追踪每个 `SearchSubAgent` 任务的成功或失败。
- **回退策略**：如果 `SearchSubAgent` 在所有重试后明确失败，`ParallelSearchAgent` 可以：
  1. 将缺失的信息通知 `ContentSynthesizerAgent`，以便它在报告中承认这一空白。
  2. （更高级）为缺失项触发一个备用 `SearchSubAgent`，使用稍宽泛或重新措辞的查询。

**改进 C：优化 `ContentSynthesizerAgent` 的成本和效率**

- **`SearchSubAgent` 的预处理**：修改 `SearchSubAgent`。在从 API 获取结果后，每个 `SearchSubAgent` 使用一次 LLM 调用执行初步的、简明扼要的摘要或提取与其特定问题相关的重要事实，*然后*再将数据发送给 `ContentSynthesizerAgent`。这显著降低了 `ContentSynthesizerAgent` 的输入令牌负载。
  - *`SearchSubAgent` 预摘要步骤的提示*：“你是一名研究助理专家。根据问题‘{original\_question}’的以下搜索结果，提取 3-5 个最重要的事实或提供不超过 150 字的简明摘要。只关注直接回答问题的信息。”
- **`ContentSynthesizerAgent` 的优化提示**：调整 `ContentSynthesizerAgent` 的提示，使其期望接收预处理、已摘要的输入。其任务变为将这些重点突出的摘要整合到连贯的叙述中。
  - *`ContentSynthesizerAgent` 的提示*：“你是一名报告撰写专家。你已收到几段摘要信息，每段都回答了主研究课题‘{main\_topic}’的一个具体子问题。你的任务是将这些摘要整合为该子问题的一个连贯部分。确保过渡平滑，逻辑清晰。以下是子问题‘{sub\_question\_text}’的摘要信息：{list\_of\_summaries\_for\_sub\_question}”

**改进 D：动态智能体伸缩（简要提及）**

对于处理可变负载的系统，`ParallelSearchAgent` 可以设计为根据 `QueryPlannerAgent` 生成的问题数量或 `AcademicSearchAPI` 响应的实时反馈来动态调整并发 `SearchSubAgent` 实例的数量。这可以防止 API 过载并优化资源使用。

### 4. 优化后分析

实施这些更改后，我们重新运行测试。新的日志可能如下所示：

```text
[2023-11-10 15:00:12] [QueryPlannerAgent] INFO: Topic "AI in Healthcare Diagnostics" decomposed into 5 questions.
[2023-11-10 15:00:13] [ParallelSearchAgent] INFO: Dispatched 5 search tasks to sub-agents.
[2023-11-10 15:00:16] [SearchSubAgent-3] INFO: Querying AcademicSearchAPI for "Ethical concerns of AI in diagnostics". (Cache MISS)
[2023-11-10 15:00:22] [SearchSubAgent-3] INFO: Received 7 results. Latency: 6150ms. Performing pre-summary.
[2023-11-10 15:00:24] [SearchSubAgent-3] INFO: Pre-summary complete. Tokens: 800 -> 120.
[2023-11-10 15:00:25] [SearchSubAgent-1] INFO: Querying AcademicSearchAPI for "Current AI algorithms for cancer detection". (Cache MISS)
[2023-11-10 15:00:30] [SearchSubAgent-1] WARN: API Timeout (Attempt 1/3). Retrying in 2s...
[2023-11-10 15:00:33] [SearchSubAgent-1] INFO: Received 5 results. Latency: 2800ms (after retry). Performing pre-summary.
[2023-11-10 15:00:35] [SearchSubAgent-1] INFO: Pre-summary complete. Tokens: 650 -> 100.
...
[2023-11-10 15:01:05] [ContentSynthesizerAgent] INFO: Received pre-summarized results for all 5 planned questions.
[2023-11-10 15:01:10] [ContentSynthesizerAgent] INFO: Synthesis complete. Total input tokens: 2500 (sum of pre-summaries). Output tokens: 7500.
```

更新后的指标表反映了这些改进：

| 指标 | 优化前 | 优化后 | 目标 |
| --- | --- | --- | --- |
| 平均任务完成时间 | 125 秒 | 45 秒 | < 60 秒 |
| API 调用成功率 | 85% | 99.5% | > 99% |
| 平均 API 延迟（每次调用） | 8 秒 | 2.5 秒 | < 4 秒 |
| 每份报告成本（令牌） | 150,000 | 45,000 | < 100,000 |
| 报告完整度评分 | 3/5（平均） | 4.9/5（平均） | 5/5 |



![平均任务完成时间](plots/6709-0.json)



> ARDS 优化前后的平均任务完成时间。

优化带来了显著成果：

- **更快的执行**：降低的 API 延迟（由于缓存和成功重试）和更高效的整合有助于大幅缩短任务完成时间。
- **更高的可靠性**：重试和更好的错误感知显著提高了 API 调用成功率和报告完整度。
- **成本降低**：`SearchSubAgent` 的预摘要处理大幅减少了 `ContentSynthesizerAgent` 的令牌输入，从而节省了大量成本。

这种分析、诊断、实施和重新评估的迭代过程，对于维护和提升多智能体 LLM 系统非常重要。

### 5. ARDS 的高级优化考量

尽管我们最初的改进是显著的，但进一步的优化始终可能。对于 ARDS 这样的系统，请考量以下持续改进的高级方面：

- **细粒度可观测性**：实施结构化日志记录和分布式追踪（使用 OpenTelemetry 等库和 LangSmith 或 Langfuse 等平台）。这使您能够可视化请求在所有智能体之间的完整流程，检查单个 LLM 提示和响应，追踪每个智能体的令牌计数，并定位每个步骤的精确延迟。这种详细视图对于调试细微问题和识别更多优化机会具有极高价值。
- **人机协作（HITL）集成**：对于 ARDS 可能难以处理细节或生成高质量查询计划的特别复杂的查询课题，可以考虑添加 HITL 检查点。

  - **查询验证**：在 `QueryPlannerAgent` 生成问题后，人类专家可以简要审查并批准/编辑这些问题，然后再将其传递给 `ParallelSearchAgent`。这可以避免在不良问题上浪费精力。
  - **最终报告审查**：人类可以审查 `ReportGeneratorAgent` 的报告以进行质量保证，特别是对于关键应用。系统甚至可以标记 (token) `ContentSynthesizerAgent` 认为置信度低或数据缺失的报告。
- **A/B 测试智能体配置**：尝试主要智能体的不同配置。例如：

  - 测试 `ContentSynthesizerAgent` 的两个版本：一个使用标准摘要提示，另一个使用鼓励更多批判性分析或信息比较的提示。
  - 尝试不同的 LLM 模型（例如，`SearchSubAgent` 预摘要使用更快、更便宜的模型，而最终的 `ContentSynthesizerAgent` 使用更强大的模型）。
  - 衡量这些更改对报告质量（通过人工评估或自动化指标）、成本和延迟的影响。
- **外部 API 交互的安全性**：

  - 如果 `AcademicSearchAPI` 需要 API 密钥，请确保它安全存储（例如，在秘密管理器中），并由 `SearchSubAgent` 访问，而不会在日志或代码中暴露。
  - 尽管对于学术 API 较不常见，但如果外部工具可能返回任意网络内容，请在该内容输入 LLM 之前实施输入验证和净化。这可以降低提示注入或处理有害数据的风险。
- **细粒度成本归因与管理**：

  - 追踪每个智能体进行的每次 LLM 调用的令牌使用量（提示和完成令牌）。这使您能够精确识别哪个智能体或哪种类型的任务是大部分成本的来源。
  - 如果单个 ARDS 任务的累积令牌使用量超过预定义阈值，则实施预算警报甚至断路器，防止成本失控。
  - 寻求在进行 LLM 调用之前估算令牌数量的技巧，特别是对于可能生成非常长提示的智能体。

通过系统地应用评估、调试和调优技术，您可以将一个功能性多智能体 LLM 系统转变为一个高效、可靠且具有成本效益的系统。请记住，随着系统需求的演变和新的交互模式的出现，优化通常是一个持续进行的过程。

## 参考资料

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  本文介绍了一种利用LLM模拟人类行为的计算代理框架，对多代理LLM系统设计和交互具有参考价值。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEgsAtayLkqUudg-gQ67aQxTDm1NuSoqPYVqr4KH1qkM47GPooNz2i0V78Ti4VZqTJN6zJdo88ha9iZ9SHeYafRnpVmMw5sUxm1nRnbu4i4T68tORmO9CssZTWCyj-SWt9F3mnA59N8InK8h_wYx3AoY3T9J-ZUClsedgFSRH_n_ML84wNUtwEP1Fal) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  提供了关于可靠性、可扩展性、缓存和错误处理等概念的全面内容，对于构建与外部API交互的多代理系统很有用。
- [OpenAI Best Practices Guide for Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2023)
  Publisher: OpenAI
  提供了改进LLM交互的策略和实践，包括成本效率和可靠性技术，适用于代理的提示设计和总结任务。
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/) — OpenTelemetry Authors (2019)
  OpenTelemetry 的官方文档，提供分布式追踪、指标和日志的标准和工具，可用于多代理系统的监控和调试。
