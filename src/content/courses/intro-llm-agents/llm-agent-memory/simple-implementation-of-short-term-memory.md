---
course: "intro-llm-agents"
chapter: "llm-agent-memory"
lesson: "simple-implementation-of-short-term-memory"
sourceId: 6616
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-6-llm-agent-memory/simple-implementation-of-short-term-memory"
title: "短期记忆的简单实现"
description: "关于如何为你的大型语言模型代理实现一种基本短期记忆的指导。"
order: 4
plots: []
sourceHash: "7b4023f98aa012b9bcb8bc018c3f614813200b4783362f57cec756d8ce586f65"
sourceCorrections: []
---

既然已经明确了记忆对于代理在持续对话中保持上下文 (context)的重要性，我们现在转向一种实用的实现方法。启用短期记忆最直接的技术是维护交互历史的按时间顺序的记录。

### 维护日志：列表方法

理解短期记忆的最简单方式就像保留对话记录。每次用户提供输入和代理响应时，这种交互都会被记录下来。在大多数编程语言中，列表（在其他语境中常被称为数组）作为一种基本的数据结构来达到此目的。数据结构仅仅是一种组织和存储数据的方式。在这种情况下，我们的列表将存储配对，每个配对包含用户的输入和代理的相应回复。

例如，如果用户询问“伦敦天气怎么样？”，并且代理在咨询其大型语言模型后回复“目前多云，有阵雨的可能性”，我们的记忆列表就会存储此交互。

以下是您在 Python 中可能如何表示此情况：

```python
# 一个用于存储对话历史的Python列表
# 列表中每个元素都是一个字典，代表一轮对话
conversation_history = []

# 第一轮对话示例
user_input_turn1 = "What is the weather like in London?"
agent_response_turn1 = "It's currently cloudy with a chance of rain."

# 将此交互添加到我们的历史列表中
conversation_history.append({
    "user_query": user_input_turn1,
    "agent_answer": agent_response_turn1
})

# 第二轮对话示例
user_input_turn2 = "What about Paris?"
agent_response_turn2 = "It's sunny in Paris."

# 添加第二轮交互
conversation_history.append({
    "user_query": user_input_turn2,
    "agent_answer": agent_response_turn2
})

# 此时，conversation_history 将会是这样：
# [
#   {"user_query": "伦敦天气怎么样？", "agent_answer": "目前多云，有阵雨的可能性。"},
#   {"user_query": "巴黎呢？", "agent_answer": "巴黎阳光明媚。"}
# ]
```

在此 Python 代码片段中，`conversation_history` 是一个字典列表。每个字典整齐地存储了一个用户查询和代理对此的回答。

### 将记忆整合到代理的工作流程中

那么，代理如何使用这个 `conversation_history` 呢？核心思路是，将此历史作为信息（即提示）的一部分，发送给大型语言模型（LLM），用于任何后续交互。

这个过程通常遵循以下步骤：

1. **接收新的用户输入：** 代理收到用户的新查询或陈述。
2. **准备提示：** 在查询大型语言模型之前，代理检索当前的 `conversation_history`。然后，它将此历史与新的用户输入一起，格式化为一个单一、完整的提示。在此格式化的提示中，清楚区分用户消息和代理消息非常重要，这样大型语言模型才能理解对话流程。
   例如，在我们的天气场景中，如果用户现在询问“巴黎呢？”，发送给大型语言模型的提示可能这样构建：

   ```
   用户：伦敦天气怎么样？
   代理：目前多云，有阵雨的可能性。
   用户：巴黎呢？
   代理：
   ```

   通过提供之前的交互，大型语言模型可以推断出“巴黎呢？”也是一个关于天气的问题，从而保持了对话的上下文 (context)。
3. **大型语言模型处理：** 大型语言模型接收这个组合提示（历史 + 新查询）并生成适当的回复。
4. **更新记忆：** 代理随后获取用户最近的输入和自己新生成的回复，并将这次最新交互追加到 `conversation_history` 列表中。

这个使用记忆向大型语言模型提供信息，然后用最新交互更新记忆的循环，是代理维护对话上下文的根本方法。

### 记忆更新周期的可视化

以下图表说明了这种基于列表的短期记忆如何整合到代理的运作循环中。

> 代理将新的用户输入与现有对话历史结合起来，为大型语言模型形成一个提示。大型语言模型的回复和初始用户输入（构成当前轮次）随后被添加回历史记录。

### 具体示例：多轮问答

让我们用一个简单的问答代理来追踪这个过程。

**初始状态：** `conversation_history = []` (我们的记忆为空)

**第一轮：**

- **用户询问：** “谁写了戏剧《哈姆雷特》？”
- **代理准备提示（尚无历史记录）：**

  ```
  用户：谁写了戏剧《哈姆雷特》？
  代理：
  ```
- **代理（在调用大型语言模型后）：** “戏剧《哈姆雷特》由威廉·莎士比亚所写。”
- **记忆更新：** `conversation_history` 现在包含：
  `[{"user_query": "谁写了戏剧《哈姆雷特》？", "agent_answer": "戏剧《哈姆雷特》由威廉·莎士比亚所写。"}]`

**第二轮：**

- **用户询问：** “他何时出生？”
- **代理准备提示（使用历史记录）：**

  ```
  用户：谁写了戏剧《哈姆雷特》？
  代理：戏剧《哈姆雷特》由威廉·莎士比亚所写。
  用户：他何时出生？
  代理：
  ```
- **代理（在调用带上下文 (context)的提示后）：** “威廉·莎士比亚出生于1564年4月。”
- **记忆更新：** `conversation_history` 现在是：

  ```
  [
    {"user_query": "谁写了戏剧《哈姆雷特》？", "agent_answer": "戏剧《哈姆雷特》由威廉·莎士比亚所写。"},
    {"user_query": "他何时出生？", "agent_answer": "威廉·莎士比亚出生于1564年4月。"}
  ]
  ```

如你所见，这种基于列表的记忆使代理能够理解第二个问题中的“他”指的是威廉·莎士比亚，这得益于第一轮提供的上下文。

### 这种简单方法的优点

使用列表作为短期记忆的主要优点是其简单性。它相对容易实现、理解和调试。对于许多应用，特别是涉及较短对话或只需最近上下文 (context)的交互，这种方法被证明相当有效。它使代理能够生成连贯且上下文相关的回复，而无需复杂的数据库系统或复杂的记忆管理技术。

尽管这种简单的基于列表的记忆是一个很好的起点，但它并非没有局限性，特别是在对话变得非常长或需要更详细的记忆形式时。我们将在下一节讨论这些限制和潜在的挑战。

## 参考资料

- [The Python Tutorial - Data Structures](https://docs.python.org/3/tutorial/datastructures.html) — Python Software Foundation (2024)
  提供关于 Python 内置列表类型及其操作的基础知识，这些是实现所描述的短期记忆的基础。
- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2000)
  Publisher: Pearson Education (Singapore)
  一本经典教科书，涵盖对话系统的基本原理以及在自然语言交互中维护对话上下文和记忆的重要性。
- [Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  官方文档，说明了如何将对话历史作为一系列消息传递给大型语言模型 API 以维护对话上下文的实际实现，直接反映了本节的方法。
