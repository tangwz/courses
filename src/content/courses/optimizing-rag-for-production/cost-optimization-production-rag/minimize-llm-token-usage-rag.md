---
course: "optimizing-rag-for-production"
chapter: "cost-optimization-production-rag"
lesson: "minimize-llm-token-usage-rag"
sourceId: 6223
sourceUrl: "https://apxml.com/zh/courses/optimizing-rag-for-production/chapter-5-cost-optimization-production-rag/minimize-llm-token-usage-rag"
title: "减少 LLM Token 使用量的方法"
description: "在 RAG 中实施提示词优化、上下文管理和摘要等方法，以减少 LLM token 消耗。"
order: 3
plots: []
sourceHash: "5e7838981e5d9cca6ad2c9840b7d27a50cc94f499b5e43120a5cdcb98b935915"
sourceCorrections: []
---

大语言模型（LLM）API 调用的成本通常按 token 计费，这占 RAG 系统运营开支的很大一部分。因此，有效管理 token 消耗是直接控制成本的手段。有多种方法可以减少 LLM token 使用，同时又不致过度影响 RAG 系统输出的质量和实用性。这些策略涵盖了从完善提示词 (prompt)到调整提供给 LLM 的上下文 (context)。

### 理解 Token 化及其成本影响

在优化之前，需要认识到 LLM 看待词语的方式与我们不同。它们将文本处理为“token”，token 可以是单词、单词的一部分，甚至单个字符，这取决于特定模型使用的分词 (tokenization)器 (tokenizer)。大多数 LLM 提供商根据输入 token（上下文 (context)和提示词 (prompt)）和输出 token（生成的回复）的数量来计费。因此，节省的每个 token 都直接转化为成本的降低。

不同的模型使用不同的分词器。例如，一个短语对模型 A 来说可能是 10 个 token，而对模型 B 来说可能是 8 个或 12 个。务必查阅模型提供商的文档，以获取其分词和定价的详细信息。像 `tiktoken`（用于 OpenAI 模型）这样的库允许你以编程方式计算给定文本和模型的 token 数量，从而能在进行 API 调用前估算成本。

```python
import tiktoken

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """估算给定文本和 OpenAI 模型的 token 数量。"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # 对于不在 encoding_for_model 中的模型，使用备用方案
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# 使用示例：
prompt = "Summarize the following document in three bullet points."
retrieved_text = "This is a sample document retrieved by the RAG system. It contains several sentences that might be relevant to the user's query about LLM token optimization."
full_input_to_llm = f"Context: {retrieved_text}\n\nInstruction: {prompt}"
token_count = count_tokens(full_input_to_llm)
# print(f"估计输入 token 数量：{token_count}")
```

> `count_tokens` 函数提供 token 使用量的估算，这对于成本分析和 RAG 流水线中的预检非常有价值。

### 提示词 (prompt)工程的策略性应用以实现简洁

你构建提示词的方式显著影响所消耗的 token 数量，包括输入和 LLM 可能的输出。

#### 1. 简洁明确的指令

避免在提示词中使用冗长或过于客套的措辞。直接清晰地说明任务。

- **效率较低**: “亲爱的 LLM，您能否分析一下所提供的文本并给我一个摘要？如果摘要简短，我将不胜感激。”（更多 token，不必要的客套语）
- **效率更高**: “简洁地总结所提供的文本。”

尽管简洁很好，但不要牺牲清晰度。一个含糊不清的短提示词可能会导致不佳的输出，需要重试或更详细的后续提示词，最终会增加 token 使用量。

#### 2. 角色提示以实现隐含的简洁性

指导 LLM 采用特定角色可以隐含地引导其回复长度和风格，有时可减少提示词中对明确长度限制的需求。

- **示例**: “你是一位简洁的技术作家。解释 RAG。”
  这通常会带来更集中且不那么冗长的回复，相比于一般请求。

#### 3. 输出格式规定

请求特定的输出格式可以引导 LLM 生成更短、结构更清晰的回复。

- **不要这样说**: “告诉我混合搜索的优点。”
- **尝试这样说**: “列出 RAG 系统中混合搜索的 3 个优点。”
  这引导 LLM 生成一个列表，通常比散文段落更节省 token，用于传达相同的信息。你也可以请求 JSON 格式的输出，这既节省 token 又便于后续解析。

#### 4. 迭代优化提示词

持续监测提示词的 token 数量和 LLM 回复的质量。迭代优化你的提示词，以寻求 token 效率和期望输出质量之间的最佳平衡。措辞的微小调整有时可以带来显著的 token 节省。

### 有效的上下文 (context)管理

提供给 LLM 的上下文（检索到的文档）通常是输入 token 数量的最大贡献者。优化此上下文是必要的。

#### 1. 以相关性为中心的上下文精简

确保你的检索系统高度准确。将不相关或微弱相关的文档传递给 LLM 不仅增加 token 数量，还可能使模型困惑并降低输出质量。

- **方法**: 实施重排序机制（第二章已介绍），以优先选择最相关的文本块。
- **策略**: 对传递给 LLM 的文本块数量或总上下文长度设定严格限制。这个限制应根据你的任务、模型能力和成本限制，通过经验来确定。

#### 2. 上下文摘要或压缩

如果生成任务不需要检索到的文本块的完整细节，考虑在它们到达主 LLM 之前对其进行摘要或压缩。

- **使用更小、更经济的 LLM**: 可以使用更小、更快、更经济的 LLM 来总结每个检索到的文本块或文本块集合。然后，主要、更强大（且更昂贵）的 LLM 接收这些摘要。
- **抽取式摘要**: 非 LLM 方法可以从文本块中选择最突出的句子。
- **抽象式摘要**: 如果使用 LLM 进行摘要，请提示它做到极致简洁。

下面的图表展示了上下文优化在 RAG 流水线中的位置：

> 上下文优化作为一个重要的中间步骤，它优化传递给主要 LLM 的信息，以减少 token 负载并可能提升回复质量。

#### 3. 管理对话历史

对于进行多轮对话的 RAG 系统，累积的历史记录会迅速导致上下文过长。

- **总结过往对话轮次**: 定期总结目前的对话，用摘要替换较早的对话轮次。
- **滑动窗口**: 只包含最近的 N 轮对话。
- **选择性包含**: 使用 LLM 或启发式方法来决定哪些过往对话轮次与当前查询最相关，并仅包含这些。

### 控制 LLM 输出长度

你也可以通过管理 LLM 生成的 token 数量来降低成本。

#### 1. 使用 `max_tokens`（或类似）参数 (parameter)

大多数 LLM API 提供一个参数（例如 `max_tokens`、`max_output_tokens`），以限制生成的回复长度。

- **注意**: 设置过低可能截断回复，导致不完整或无意义的输出。根据你的使用场景中典型的预期回复长度来确定合适的值。通常最好通过提示词 (prompt)引导 LLM 保持简洁，而不是仅仅依靠 `max_tokens` 来突然截断它。
- **用途**: 对于预期非常简短回复的任务很有用（例如，“是/否”问题，分类）。

#### 2. 提示词引导简洁性

明确要求 LLM 在回复中保持简洁。

- **示例**:
  - “提供简短解释。”
  - “用一句话回答。”
  - “将回复控制在 50 字以内。”

尽管 LLM 并非总是能完美遵守精确的字数或句数，但此类指令通常会使输出更短。

### 运用模型等级和任务专用模型

并非所有任务都需要最强大（且最昂贵）的 LLM。

- **针对简单任务使用较小模型**: 如果你的 RAG 流水线的一部分涉及简单任务（例如，在检索前从查询中提取关键词，为上下文 (context)总结一小段文本），更小、更经济的模型可能就足够了，并且整体消耗更少的 token（或具有更低的每 token 成本）。
- **微调 (fine-tuning)模型**: 针对特定任务（例如摘要或根据提供上下文回答问题）微调的较小模型，在性能和该狭窄任务的 token 使用量方面都可能高效。

### 缓存 LLM 回复

尽管并非严格意义上减少*单次独立*调用的 token，但缓存相同或非常相似的输入提示词 (prompt)（包括上下文 (context)）的回复可以完全消除冗余的 LLM 调用，从而在大规模应用中带来可观的成本节省。如果你经常遇到相同的问题和相同的检索上下文，缓存 LLM 生成的答案会非常有效。本方法在第四章“在 RAG 流水线中实施缓存策略”中进行更多介绍。

### 监测与迭代

token 优化并非一次性设置。这是一个持续的过程。

- **追踪 Token 使用量**: 对每次 LLM 调用实施详细的 token 数量（输入和输出）日志记录。分析这些日志以识别：
  - 导致异常高 token 消耗的查询或上下文 (context)类型。
  - 不同提示词 (prompt)策略对 token 数量的影响。
- **A/B 测试提示词策略**: 试验不同的措辞、上下文长度和输出限制。不仅要衡量 token 使用量，还要衡量 LLM 回复的质量，以找到最佳平衡点。
- **保持更新**: LLM 提供商经常发布新模型或更新现有模型。新模型可能更高效，或提供更好的能力来控制输出长度和冗余度。

通过系统地应用这些方法，你可以显著减少 RAG 系统与 LLM 交互的 token 足迹。这不仅降低了运营成本，还可以提高延迟，因为处理更少的 token 通常所需时间更短。重要的是找到恰当的平衡点，既保持高质量输出，又注意相关开支。

## 参考资料

- [API Pricing](https://openai.com/pricing) — OpenAI (2024)
  Publisher: OpenAI
  提供关于大型语言模型 (LLM) 分词、定价模型以及令牌使用如何影响运营成本的官方详情。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2023)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  提供了 RAG 系统的全面概述，讨论了各种上下文优化技术，例如重新排序和摘要，以提高效率并减少令牌使用。
- [Prompt Engineering Guide](https://cloud.google.com/vertex-ai/docs/generative-ai/prompt-engineering/best-practices) — Google Cloud (2025)
  Publisher: Google
  提供了编写有效提示的实用策略，包括实现简洁和清晰指令的方法，有助于优化 LLM 输入和输出令牌数量。
- [tiktoken (Official GitHub Repository)](https://github.com/openai/tiktoken) — OpenAI (2024)
  tiktoken 的官方 GitHub 仓库和文档，它是 OpenAI 快速的字节对编码 (BPE) 分词器，用于 OpenAI 模型的准确令牌计数和成本估算。
