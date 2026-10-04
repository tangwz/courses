---
course: "getting-started-with-llm-toolkit"
chapter: "context-and-token-management"
lesson: "understanding-context-window"
sourceId: 7759
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-3-context-and-token-management/understanding-context-window"
title: "上下文窗口的作用"
description: "了解什么是上下文窗口，以及为何管理它对于应用程序性能和成本是必要的。"
order: 1
plots: []
sourceHash: "86825b41e16eddc51fb89c1fc70c2fdb5f2ec80080c108c66a096463b2b00992"
sourceCorrections: []
---

大型语言模型在一个有限的工作空间内处理信息，这个空间叫做上下文 (context)窗口。可以将其视为模型在单次交互中的短期记忆。模型为生成响应所“知”的一切，包括系统指令、对话历史、检索到的文档以及您的最新查询，都必须容纳在这个空间内。这个窗口不是以单词或字符衡量的，而是以令牌（token）衡量的，令牌是模型处理的单词或字符片段。

这个窗口的大小是一个固定的架构限制。例如，一个模型可能有一个8,192个令牌的上下文窗口。发送一个超出此限制的提示并非是对模型进行总结的建议；它是一个强制性限制，将导致API错误并阻止您的应用程序正常运行。

### 上下文 (context)管理为何必要

管理上下文窗口是构建生产级LLM应用程序的基本技能。低效的管理会影响三个主要方面：应用程序的可靠性、运行成本和响应质量。

#### 可靠性

频繁发送超出模型上下文窗口大小的提示的应用程序，其故障也会频繁发生。这些故障是显而易见的；API提供商会直接拒绝请求。对于会随着时间推移积累上下文的应用程序，例如维护过往交互记忆的聊天机器人或代理，超出限制的风险会随着每一次轮次增加。如果没有主动管理，长时间的对话几乎肯定会最终触及这个上限，导致糟糕的用户体验。

#### 成本

LLM的API费用与您发送和接收的令牌数量成正比。如章节介绍所述，费用是根据输入和输出令牌共同计算的：

$成本 = (输入价格 \times 输入令牌) + (输出价格 \times 输出令牌)$

每一个发送到模型的令牌，无论是否与最终答案相关，都会增加输入成本。设想一个场景，您的RAG系统检索了十份文档，但实际上只有前三份是回答用户查询所必需的。如果您发送全部十份文档，您可能会用数千个不必要的令牌填充上下文窗口，为模型会忽略的信息付费。这种浪费在处理量大的应用程序中会迅速累积，使一个高效的系统变得昂贵。

例如，假设一个提示长达4,000个令牌，而只有1,000个令牌是相关的。按照每百万输入令牌0.50美元的典型价格，处理1,000个此类请求的成本将是：

- **低效：** 1,000次请求 \* 4,000令牌/请求 \* ($0.50 / 1,000,000令牌) =$2.00
- **高效：** 1,000次请求 \* 1,000令牌/请求 \* ($0.50 / 1,000,000令牌) =$0.50

通过简单高效地管理上下文，您可以在这种情况下将成本降低75%。

#### 响应质量

LLM输出的质量高度依赖于其输入的质量。上下文窗口不仅仅是一个需要填充的空间；它是模型处理当前任务的整个参考框架。研究表明，模型通常表现出U形注意力曲线，对上下文窗口开头和结尾的信息给予更多关注。“在中间丢失”的信息可能会被忽略。

> 提示的每个组成部分都会争夺模型在上下文窗口中有限的注意力和空间。

如果您用不相关或冗余的信息填充上下文窗口，就会产生干扰，使模型偏离提示中核心的信息。这可能导致几个问题：

- **关联性降低：** 模型可能会根据它在上下文中间发现的关联性较低的信息生成答案。
- **答案不完整：** 开头或结尾的核心指令或上下文可能得到适当关注，但中间的细节会丢失。
- **幻觉 (hallucination)增加：** 当信噪比低时，模型更有可能生成不基于所提供上下文的事实。

有效的上下文管理涉及精心准备一个清晰、精炼且相关的提示，只为模型提供执行任务所需的内容，不多余。此过程的第一步是能够准确测量您的文本将消耗多少令牌。`tokenizer`模块为此基本任务提供了工具。

```python
from kerb.tokenizer import count_tokens, Tokenizer

text_short = "这是一个短句。"
text_long = "这是一个更长的句子，旨在说明令牌计数如何随着所提供文本的长度而增加。"

# 使用GPT-4和GPT-3.5-turbo的令牌器
tokens_short = count_tokens(text_short, tokenizer=Tokenizer.CL100K_BASE)
tokens_long = count_tokens(text_long, tokenizer=Tokenizer.CL100K_BASE)

print(f"短句: '{text_short}'")
print(f"令牌计数: {tokens_short}\n")

print(f"长句: '{text_long}'")
print(f"令牌计数: {tokens_long}")

# 短句: '这是一个短句。'
# 令牌计数: 6
#
# 长句: '这是一个更长的句子，旨在说明令牌计数如何随着所提供文本的长度而增加。'
# 令牌计数: 24
```

正如您所见，令牌计数是一个精确的测量值。在接下来的章节中，您将学习如何使用此功能来构建可靠、经济高效并能产生高质量结果的应用程序。

## 参考资料

- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008
  介绍了Transformer架构，它是大型语言模型的基础，并定义了输入序列上的注意力机制，从而确立了上下文窗口的概念。
- [OpenAI Platform - Models](https://platform.openai.com) — OpenAI (2024)
  官方文档，详细说明了可用的大型语言模型、各自的上下文窗口大小（以token计）以及其他API限制，这些对于管理应用程序的可靠性和成本至关重要。
