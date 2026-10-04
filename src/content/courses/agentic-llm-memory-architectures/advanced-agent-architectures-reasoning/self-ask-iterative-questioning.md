---
course: "agentic-llm-memory-architectures"
chapter: "advanced-agent-architectures-reasoning"
lesson: "self-ask-iterative-questioning"
sourceId: 4447
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-2-advanced-agent-architectures-reasoning/self-ask-iterative-questioning"
title: "自问：通过迭代提问提升事实准确性"
description: "理解Self-Ask代理分解复杂问题的机制。"
order: 3
plots: []
sourceHash: "9a83e8aeff760c77563b41f4ea937f88c448c94f19213f5b7b239a28d657afcb"
sourceCorrections: []
---

尽管ReAct等架构侧重于交错进行推理 (inference)和行动以完成任务，其他框架则侧重于改进LLM响应背后的事实依据和显式推理过程，尤其对于需要信息整合的复杂问题。Self-Ask机制通过使代理能够将复杂查询明确地分解为一系列更简单、相互关联的子问题来处理此点。此迭代过程旨在在尝试给出最终答案之前收集必要的事实片段，从而增加可追溯性并通常提高准确性。

LLM在特定提示的引导下，不会直接回答一个可能多方面的问题，而是首先判断该问题是否可以直接回答，或者是否需要分解。如果需要分解，LLM会形成一个后续问题，目标是特定缺失的信息片段。这个子问题通常被设计成可以通过外部查询来回答，常使用搜索引擎或专用知识库API。一旦子问题的答案被检索到（我们称之为中间答案 $i_k$），它就会被反馈回上下文 (context)中。LLM随后评估它是否有足够的信息来回答原始问题，或者是否需要另一个后续问题（$q_{k+1}$）。这个循环持续进行，直到收集到所有必要的中间事实，从而使LLM能够综合出最终、有依据的答案。

### Self-Ask迭代循环

考虑这样一个问题：“马来西亚和泰国首都的人口差异是多少？”

一个Self-Ask代理将按如下方式进行：

1. **初步分析：** LLM识别出需要特定事实：马来西亚的首都、泰国的首都及其各自的人口。
2. **生成子问题1：** LLM输出一个后续问题：`Follow-up: What is the capital of Malaysia?`
3. **外部查询1：** 一个外部工具（例如，搜索API）被调用，查询内容为“马来西亚的首都是什么？”。它返回`Intermediate Answer: Kuala Lumpur`。
4. **整合答案并生成子问题2：** 上下文 (context)现在包含“吉隆坡是马来西亚的首都”。LLM识别出下一个缺失的信息并提问：`Follow-up: What is the capital of Thailand?`
5. **外部查询2：** 该工具搜索“泰国的首都是什么？”，并返回`Intermediate Answer: Bangkok`。
6. **整合答案并生成子问题3：** 上下文已更新。下一个缺失的信息：`Follow-up: What is the population of Kuala Lumpur?`
7. **外部查询3：** 工具搜索“吉隆坡人口”，返回`Intermediate Answer: ~1.9 million`。
8. **整合答案并生成子问题4：** 上下文已更新。下一个：`Follow-up: What is the population of Bangkok?`
9. **外部查询4：** 工具搜索“曼谷人口”，返回`Intermediate Answer: ~10 million`。
10. **综合最终答案：** 所有必要的事实都已收集。LLM现在拥有首都（吉隆坡、曼谷）和人口（约190万，约1000万）。它计算出差异并形成最终响应：`Final Answer: The difference in population between Kuala Lumpur (~1.9 million) and Bangkok (~10 million) is approximately 8.1 million.`

这种结构化的分解使推理 (inference)过程透明，并依赖外部工具进行事实检索，从而降低组成事实出现幻觉 (hallucination)的可能性。

> Self-Ask机制的迭代流程。LLM判断是否需要分解，生成子问题，使用外部工具获取答案，整合结果，并重复此过程，直到原始问题可被回答。

### 专家系统的实现细节

有效地实现Self-Ask需要仔细考虑以下几个方面：

- **提示工程 (prompt engineering)：** Self-Ask的核心在于提示结构。它必须引导LLM：

  - 识别何时问题复杂且需要事实分解。
  - 明确输出带有特定前缀的后续问题（例如，`Follow-up:`）。
  - 在预期外部查询结果时发出信号（通常通过在后续问题后停止生成）。
  - 为最终答案使用一个独特的的前缀（例如，`Final Answer:`）。
  - 将中间答案（由系统在查询后提供）整合到其后续推理 (inference)步骤中。
    这通常涉及在提示中包含少量示例，以显示所需的迭代行为。
- **工具整合与错误处理：** 系统需要与外部查询工具（例如，搜索API封装器）进行紧密整合。该工具的故障（网络错误、无结果、结果模糊）必须谨慎处理。代理可能需要逻辑来：

  - 重新措辞子问题并重试。
  - 尝试使用替代工具或来源进行查询。
  - 将无法找到特定信息的情况报告回LLM，使其可能调整策略或在最终答案中说明限制。
- **管理迭代深度：** 无限制的分解可能导致过长的延迟或循环。实施保障措施是必要的：

  - **最大深度：** 限制允许的后续问题数量（$k_{max}$）。
  - **置信度评分：** 如果LLM能够输出其需要提后续问题的置信度分数，则可以使用一个阈值。
  - **依赖关系分析：** 更精密的实现可能会跟踪问题的依赖图，以检测循环或冗余查询。
- **与其他架构的关系：** Self-Ask的优势在于其结构化方法，用于处理组合式问题的事实查找。与ReAct不同，其主要循环不侧重于在环境中行动，而是通过有针对性的信息收集来完善理解。与思维树（ToT）相比，Self-Ask遵循由感知到的信息缺口驱动的更线性的分解路径，而ToT则同时考量多个推理分支。当答案路径涉及顺序解决事实依赖时，它表现出色。

### 优势与局限

**优势：**

- **事实准确性提升：** 通过将中间步骤建立在外部知识源上，Self-Ask可以大幅减少最终综合中的事实错误。
- **可解释性：** 子问题和中间答案的明确序列提供了代理推理 (inference)过程的清晰轨迹。
- **组合性：** 它自然地处理需要组合多条信息的问题。

**局限：**

- **延迟：** 每次涉及外部查询的迭代都会增加延迟，使其比直接生成更慢。
- **错误传播：** 中间答案（来自外部工具）或有缺陷的子问题（来自LLM）中的错误可能会使整个过程偏离轨道。
- **工具依赖性：** 性能严重依赖于外部查询工具的质量和可靠性。
- **适用范围：** 不太适合需要创意生成、涉及在环境中行动的复杂规划，或主观判断的任务，在这些任务中，事实分解并非主要挑战。

总之，Self-Ask代表着一种构建更可靠、更透明问答代理的重要技术。通过强制分解复杂查询并借助外部知识源来奠定事实依据，它提供了一种结构化机制，用于提高LLM在信息密集型场景中响应的准确性和可信度。它是设计高级代理系统的重要组成部分，在此类系统中，事实精确度是优先考虑的高项。

## 参考资料

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601) — Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan (2023)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2305.10601](https://doi.org/10.48550/arXiv.2305.10601)
  描述了思维树（Tree of Thoughts）框架，该框架允许语言模型探索多个推理路径，并通过显式状态评估进行自我纠正。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  关于检索增强生成（RAG）的奠基性论文，该方法通过从外部知识库检索信息来增强语言模型。
