---
course: "getting-started-with-llm-toolkit"
chapter: "conversational-applications-with-memory"
lesson: "summary-memory-for-long-conversations"
sourceId: 7780
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-7-conversational-applications-with-memory/summary-memory-for-long-conversations"
title: "使用摘要记忆进行长对话"
description: "实现摘要记忆，以浓缩长对话历史并保持在上下文限制内。"
order: 3
plots: []
sourceHash: "c133cf79fd09316a6832c9478993a2c500e0f6ce1ec1f657516362d99a8e78d1"
sourceCorrections: []
---

直接存储对话的完整历史记录固然可行，但它带来了一个很大的难题：上下文 (context)窗口。随着对话变长，消息列表很快就会超出即使是最强大模型的令牌限制。这种情况会导致消息被截断（即丢弃旧消息）或API报错。对于需要长时间对话的应用，一种更具扩展性的方法是必需的。

这时，摘要记忆便派上用场。这种方法不是保留每一条消息，而是通过语言模型逐步将对话历史浓缩成一个连续的摘要。这种做法可以在控制令牌数量的同时，保留对话中的核心内容。

### 渐进式摘要方法

核心思想是在几次对话轮次后，创建一个新的、更精简的摘要。这个过程会把最新的消息与之前的摘要结合起来，生成一个更新后的摘要。这种“滚动摘要”随着对话的进展而演变，在加入新内容的同时，控制其整体长度。

> 渐进式摘要过程。新消息与之前的摘要结合，生成一份更新、精简的历史记录。

`memory` 模块为此目的提供了 `create_progressive_summary` 函数。它接收一个新消息列表和一个可选的 `existing_summary` 作为基础进行构建。

```python
from kerb.core.types import Message
from kerb.memory.summaries import create_progressive_summary

# 对话的初始消息
messages_turn_1 = [
    Message("user", "学习Python进行数据科学的最佳方法是什么？"),
    Message("assistant", "从pandas开始进行数据处理，matplotlib进行数据可视化。")
]

# 创建第一个摘要
summary_1 = create_progressive_summary(messages_turn_1, summary_length="short")
print(f"第1轮后的摘要:\n{summary_1}\n")

# 下一轮添加的新消息
messages_turn_2 = [
    Message("user", "机器学习库呢？"),
    Message("assistant", "对于机器学习，scikit-learn是经典算法的绝佳起点。")
]

# 使用旧摘要作为上下文创建新摘要
summary_2 = create_progressive_summary(
    messages_turn_2,
    existing_summary=summary_1,
    summary_length="short"
)
print(f"第2轮后的摘要:\n{summary_2}")
```

您可以通过 `summary_length` 参数 (parameter)控制生成的摘要的详细程度，该参数接受 `"short"`（短）、`"medium"`（中）或 `"long"`（长）的值。这使您能够在节省令牌和保留细节之间取得平衡。

### ConversationBuffer 中的自动摘要

虽然您可以手动管理渐进式摘要，但 `ConversationBuffer` 类可以使这个过程自动化。当您使用 `enable_summaries=True` 初始化一个缓冲区时，它会自动为那些超出 `max_messages` 限制而被推出内存的旧消息创建摘要。这提供了一种实用且高效的方法来处理长对话，无需人工操作。

让我们用一个被强制修剪消息的小缓冲区来观察这一点。

```python
from kerb.memory import ConversationBuffer

# 创建一个将快速修剪消息的小缓冲区
small_buffer = ConversationBuffer(
    max_messages=5,
    enable_summaries=True
)

print(f"向 max_messages=5 的缓冲区添加 8 条消息...\n")
for i in range(8):
    small_buffer.add_message("user", f"这是第 {i+1} 条消息。")

print(f"当前存储的消息数量: {len(small_buffer.messages)}")
print(f"从修剪消息创建的摘要数量: {len(small_buffer.summaries)}")

if small_buffer.summaries:
    first_summary = small_buffer.summaries[0]
    print(f"\n修剪消息的摘要:")
    print(f"  '{first_summary.summary}'")
    print(f"  (此摘要涵盖 {first_summary.message_count} 条消息)")
```

随着消息的添加，缓冲区会修剪最旧的消息以维持其大小限制，并将它们浓缩成一个摘要，该摘要存储在 `small_buffer.summaries` 中。此摘要随后可以包含在未来的 LLM 调用上下文 (context)中。

### 分层摘要

分层摘要是单一渐进式摘要的一种替代方案。这种方法是将对话分成若干块，并独立地为每块内容创建摘要。这对于非常长的对话记录很有用，因为您可能希望从对话的不同部分检索特定信息点，而不是依赖于一个单一的、不断变化的摘要。

`create_hierarchical_summary` 函数通过为对话的每个片段创建 `ConversationSummary` 对象列表来实现这一点。

```python
from kerb.core.types import Message
from kerb.memory.summaries import create_hierarchical_summary

long_conversation = [Message("user", f"消息 {i+1}") for i in range(12)]

# 为每5条消息的块创建摘要
hierarchical_summaries = create_hierarchical_summary(long_conversation, chunk_size=5)

print(f"创建了 {len(hierarchical_summaries)} 个分层摘要:\n")
for i, summary_obj in enumerate(hierarchical_summaries):
    print(f"块 {i+1} 的摘要 (涵盖 {summary_obj.message_count} 条消息):")
    print(f"  '{summary_obj.summary}'\n")
```

### 权衡

摘要记忆是一种强大的技术，但了解其优缺点很有必要。

**优点：**

- **令牌效率：** 它能高效控制对话历史的规模，支持非常长的互动而不会超出上下文 (context)限制。
- **扩展性：** 通过持续浓缩历史记录，它理论上可以处理任意长度的对话。

**缺点：**

- **信息丢失：** 摘要过程本身会造成信息损失。某些细节或特定措辞可能会在摘要中被省略。
- **延迟和成本增加：** 每个摘要步骤都需要额外的 LLM 调用，这会增加应用的延迟和成本。
- **内容偏移的可能：** 在非常长的对话中，摘要可能会偏离最初意图，或丢失对话早期重要的基础内容。

选择摘要记忆取决于您应用的需求。对于需要处理长时间对话的聊天机器人，它是一个不可或缺的工具。对于需要完美回溯之前轮次的任务，可能需要采用不同的策略。

## 参考资料

- [Dialogue Summarization with Hierarchical Structure and Global Information](https://aclanthology.org/2022.emnlp-main.730/) — Tianhao Shen, Mingtong Liu, Ming Zhou, Deyi Xiong (2022)
  Journal: Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing; Publisher: Association for Computational Linguistics; Pages: 10659-10670; DOI: [10.18653/v1/2022.emnlp-main.730](https://doi.org/10.18653/v1/2022.emnlp-main.730)
  探讨了利用分层结构总结对话的方法，直接对应于内容中介绍的“分层摘要”概念。
- [Conversational AI: Dialogue Systems, Large Language Models, and the Future of AI](https://link.springer.com/book/10.1007/978-3-031-26792-2) — Michael J. McTear, Zong-Yuan Huang, Liming Luke Liu (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-26792-2](https://doi.org/10.1007/978-3-031-26792-2)
  一本关于会话式人工智能的最新书籍，全面概述了会话式人工智能，包括对内存管理、上下文处理以及设计有效对话系统中的挑战和权衡的详细讨论。
