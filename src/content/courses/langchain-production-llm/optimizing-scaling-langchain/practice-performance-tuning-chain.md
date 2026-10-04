---
course: "langchain-production-llm"
chapter: "optimizing-scaling-langchain"
lesson: "practice-performance-tuning-chain"
sourceId: 3923
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-6-optimizing-scaling-langchain/practice-performance-tuning-chain"
title: "实践：LangChain 链的性能调优"
description: "识别并解决示例 LangChain 应用中的性能问题。"
order: 7
plots: ["plots/3923-0.json"]
sourceHash: "8da780f13361e81a20f2f30fe7559d60131c4d46f3871a81948948ab7c72408e"
sourceCorrections: []
---

调优一个示例 LangChain 链，为优化提供了实际应用。该过程包括找出性能问题、应用特定方法以及衡量其效果。

### 场景：一个多步文档分析链

设想一个旨在根据技术报告集合回答问题的链。该过程包括：

1. **检索：** 使用向量 (vector)存储找出相关文档片段。
2. **初步答案生成：** 使用大型语言模型，*仅*基于检索到的片段生成答案。
3. **答案优化：** 使用*第二个*、可能更强大大型语言模型调用，以优化初步答案、提升连贯性，并根据原始问题和初步答案增加背景信息。

这种多步过程很常见，但由于多次大型语言模型交互和数据检索操作，可能引入延迟并增加成本。

假设我们最初的链实现如下所示：

```python
# 假设 retriever, llm_initial, llm_refine 已预先配置
# retriever: 一个向量存储检索器
# llm_initial: 一个中等大小的大型语言模型，用于快速生成答案
# llm_refine: 一个更大、功能更强的大型语言模型，用于优化

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 简化的 RAG 设置
retrieve_docs = RunnablePassthrough.assign(
    context=(lambda x: x["question"]) | retriever
)

# 初步答案提示和链
initial_prompt_template = ChatPromptTemplate.from_template(
    "基于此背景信息：\n{context}\n\n回答问题：{question}"
)
initial_answer_chain = initial_prompt_template | llm_initial | StrOutputParser()

# 优化提示和链
refine_prompt_template = ChatPromptTemplate.from_template(
    "基于原始问题：'{question}'，优化此初步答案：'{initial_answer}'。确保连贯性和准确性。"
)
refine_chain = refine_prompt_template | llm_refine | StrOutputParser()

# 结合步骤的完整链
full_chain = retrieve_docs | RunnablePassthrough.assign(
    initial_answer=initial_answer_chain
) | RunnablePassthrough.assign(
    final_answer = (lambda x: {"initial_answer": x["initial_answer"], "question": x["question"]}) | refine_chain
)

# 调用示例
# result = full_chain.invoke({"question": "What are the scaling limits of System X?"})
# print(result["final_answer"])
```

### 步骤 1：基线性能测量

在优化之前，我们需要一个基线。我们可以使用简单计时或与 LangSmith 等追踪工具（第 5 章介绍）结合使用。为简便起见，我们使用基本计时。我们使用一个示例问题多次运行该链，并取结果的平均值。

```python
import time
import statistics
import random

question = "概括在负载下性能下降的主要发现。"
num_runs = 5
latencies = []
# 假设令牌跟踪是单独实现或通过 LangSmith 实现

for _ in range(num_runs):
    start_time = time.time()
    # result = full_chain.invoke({"question": question}) # 执行链
    # 模拟执行时间用于演示
    time.sleep(12 + (random.random() * 6 - 3)) # 模拟 9-15 秒延迟
    end_time = time.time()
    latencies.append(end_time - start_time)

average_latency = statistics.mean(latencies)
print(f"平均延迟（基线）：{average_latency:.2f} 秒")
# 假设通过日志/LangSmith 观察到的基线令牌计数：每次查询约 2100 个令牌
```

假设我们的基线测量结果如下：

- **平均延迟：** 13.5 秒
- **平均令牌使用量：** 2100 个令牌（估计）

使用 LangSmith 或详细日志记录，我们可能会看到具体细分：

- 检索：1.5 秒
- 初步答案大型语言模型 (`llm_initial`)：4.0 秒 (800 个令牌)
- 优化大型语言模型 (`llm_refine`)：8.0 秒 (1300 个令牌)

优化步骤 (`llm_refine`) 是延迟和令牌消耗两方面最主要的瓶颈。

### 步骤 2：应用优化方法

让我们应用之前提到的一些方法。

**方法 1：缓存大型语言模型响应**

相同的问题或中间处理步骤可能频繁出现。缓存大型语言模型响应可以显著降低重复请求的延迟和成本。让我们添加一个内存缓存。在生产环境中，通常会使用更持久的缓存，如 Redis、SQL 或专用向量 (vector)缓存。

```python
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# 设置一个简单的内存缓存
set_llm_cache(InMemoryCache())

# 如果大型语言模型已全局配置，则链定义本身无需修改
# 或者，在初始化大型语言模型时直接应用缓存：
# llm_initial = ChatOpenAI(..., cache=InMemoryCache())
# llm_refine = ChatOpenAI(..., cache=InMemoryCache())

# 重新运行计时测试，确保多次运行*相同*的问题
# 首次运行会较慢，后续相同的运行应该快很多。
```

添加缓存并再次运行*相同查询*后：

- **首次运行延迟：** 约 13.5 秒（缓存未命中）
- **后续运行延迟：** 约 1.6 秒（两个大型语言模型均缓存命中，主要受检索影响）
- **令牌使用量（后续运行）：** 0 个令牌（从缓存提供）

缓存对于重复输入非常有效，但对新查询无效。

**方法 2：优化优化步骤**

优化大型语言模型调用是我们新查询的主要瓶颈。

- **提示工程 (prompt engineering)：** 我们能否使优化提示更简洁？也许初步提示可以要求更结构化的输出，从而减少优化需求。假设我们对 `refine_prompt_template` 进行优化，使其稍微缩短，平均每次调用可节省约 50 个令牌。
- **模型选择：** 强大的 `llm_refine` 是否绝对必要？一个稍小、更快的模型能否达到可接受的质量？假设我们将 `llm_refine` 切换到一个已知在类似任务中平均快约 30%、使用令牌少约 30% 的模型，也许会接受一些轻微的质量权衡。
- **条件执行：** 也许并非总是需要优化。我们可以在优化*之前*添加一个步骤，使用更简单的模型或基于规则的检查来确定初步答案是否足够好。如果足够好，则完全跳过优化调用。

让我们模拟将 `llm_refine` 切换到更快模型并稍微优化提示的效果。

```python
# 假设 llm_refine_faster 已配置（一个更快、能力稍弱的模型）
# 假设 refine_prompt_template_optimized 稍微缩短

# 更新 refine_chain 部分
refine_chain_optimized = refine_prompt_template_optimized | llm_refine_faster | StrOutputParser()

# 更新完整链定义以使用优化后的优化链
full_chain_optimized = retrieve_docs | RunnablePassthrough.assign(
    initial_answer=initial_answer_chain
) | RunnablePassthrough.assign(
    final_answer = (lambda x: {"initial_answer": x["initial_answer"], "question": x["question"]}) | refine_chain_optimized
)

# 重新运行新查询的计时测试（缓存在此处最初不会有帮助）
# ... 计时代码 ...
```

### 步骤 3：重新评估性能

让我们测量 `full_chain_optimized` 针对新查询（缓存未命中情况）的性能：

```python
# 模拟优化后的执行时间用于演示
# 检索：1.5 秒（无变化）
# 初步大型语言模型：4.0 秒（无变化，800 个令牌）
# 优化大型语言模型（更快模型 + 提示）：8.0 秒 * 0.7 ≈ 5.6 秒 (1300 个令牌 * 0.7 - 50 ≈ 860 个令牌)
# 总延迟 ≈ 1.5 + 4.0 + 5.6 = 11.1 秒
# 总令牌 ≈ 800 + 860 = 1660 个令牌

# --- 用于模拟和测量的 Python 代码 ---
latencies_optimized = []
for _ in range(num_runs):
     start_time = time.time()
     # result = full_chain_optimized.invoke({"question": question}) # 执行优化后的链
     # 模拟优化后的执行时间
     time.sleep(10 + (random.random() * 3 - 1.5)) # 模拟 8.5 - 11.5 秒延迟
     end_time = time.time()
     latencies_optimized.append(end_time - start_time)

average_latency_optimized = statistics.mean(latencies_optimized)
print(f"平均延迟（优化后）：{average_latency_optimized:.2f} 秒")
# 估计优化后的令牌计数：约 1660 个令牌
```

我们针对新查询的新测量结果可能如下：

- **平均延迟：** 10.5 秒（原为 13.5 秒）
- **平均令牌使用量：** 1660 个令牌（原为 2100 个令牌）

### 结果对比



![链性能优化结果](plots/3923-0.json)



> 平均延迟和令牌使用量在应用缓存和模型优化方法前后的对比。请注意缓存查询的显著提升。

### 成本影响

减少令牌计数直接影响成本。如果 `llm_initial` 和 `llm_refine` 的总成本为每 1K 令牌 \$0.002：

- **基线成本：** (2100 / 1000) \* $0.002 = 每次查询$0.0042
- **优化后成本（新查询）：** (1660 / 1000) \* $0.0018 (假设更快的模型更便宜) ≈ 每次查询$0.0030（约节省 28%）
- **优化后成本（缓存）：** 每次查询 \$0.00（节省 100%）

### 权衡与后续步骤

我们取得了显著提升：

- **缓存：** 对重复查询非常有效，代码改动极小，缓存本身可能产生内存/存储成本。
- **模型切换/提示调优：** 降低了所有查询的延迟和成本，但可能涉及优化答案质量上的轻微权衡。评估这种质量差异很重要（第 5 章介绍）。

这个练习展示了一个典型的调优流程：

1. **测量：** 建立基线。
2. **识别：** 找出最耗费的步骤（时间、令牌、成本）。
3. **优化：** 应用有针对性的方法，如缓存、模型选择、提示工程 (prompt engineering)或结构调整（例如，条件执行）。
4. **重新评估：** 衡量你的改动效果。
5. **迭代：** 性能调优通常是一个迭代过程。进一步的提升可能包括优化检索步骤、考虑独立任务的并行执行或实现更复杂的缓存。

记住使用 LangSmith 等工具进行详细追踪和分析，这将大大简化复杂应用中的识别和测量阶段。

## 参考资料

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: NeurIPS 2020; DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  一篇介绍检索增强生成（RAG）方法的论文，它是多步骤文档分析链的基础。
- [OpenAI: Prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) — OpenAI (2024)
  Publisher: OpenAI
  提供设计有效提示的通用方法，这直接影响LLM的性能、成本和输出质量。
