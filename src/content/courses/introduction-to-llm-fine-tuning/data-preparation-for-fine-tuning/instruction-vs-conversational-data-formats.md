---
course: "introduction-to-llm-fine-tuning"
chapter: "data-preparation-for-fine-tuning"
lesson: "instruction-vs-conversational-data-formats"
sourceId: 7269
sourceUrl: "https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-2-data-preparation-for-fine-tuning/instruction-vs-conversational-data-formats"
title: "指令式与对话式数据格式"
description: "了解指令遵循型和对话型数据格式的区别及其组织方式。"
order: 2
plots: []
sourceHash: "2fb2f3d412f0406f9dc755f034217d1436aa5bbd438a35ec196c9c1f071de4dc"
sourceCorrections: []
---

微调 (fine-tuning)数据的结构直接决定了模型学习到的能力。要让模型掌握一项新能力，您必须以一致、机器可读的格式提供该能力的示例。对于大多数微调任务，这些数据通常采用两种主要格式之一组织：指令式或对话式。您选择的格式是一项基础性决定，它使您的数据集与模型的预期使用场景相符。

### 指令式格式

指令式数据集旨在训练模型执行特定、独立的任务。模型学习将给定的提示映射到期望的完成内容。这种格式非常适合需要一次性请求和响应的应用，例如文本摘要、翻译、分类和问答。

微调 (fine-tuning)数据的结构直接影响模型学习到的能力。要让模型掌握一项新技能，必须以一致的、机器可读的格式呈现该技能的示例。对于大多数微调任务，数据通常采用两种主要格式之一：指令型或对话型。所选择的格式是一个基础性决定，它将数据集与模型的预期用途对齐 (alignment)。

设想一个用于训练模型总结文章的数据集。一个数据点可能看起来像这样：

```json
{
  "instruction": "用三句话总结以下文章。",
  "input": "The Falcon 9 is a two-stage rocket designed and manufactured by SpaceX for the reliable and safe transport of satellites and the Dragon spacecraft into orbit. The rocket's first stage is reusable, capable of re-entering the atmosphere and landing vertically after separating from the second stage. This reusability has significantly reduced the cost of access to space.",
  "output": "The Falcon 9 is a two-stage, reusable rocket from SpaceX used for orbital transport. Its first stage can land vertically after launch, a feature that lowers launch costs. This capability makes it a cost-effective choice for deploying satellites and spacecraft."
}
```

训练期间，这些结构化字段通常使用**提示模板**组合成一个字符串。该模板统一了输入格式，告知模型需要执行哪种任务。对于上述示例，模板可能是：

```python
prompt = f"""以下是一个描述任务的指令，以及提供更多上下文的输入。请写出恰当完成请求的回复。

### 指令：
{example['instruction']}

### 输入：
{example['input']}

### 回复：
"""
```

模型随后被训练，以生成`output`字段中的文本作为对该格式化提示的完成。提示模板在所有示例中的一致性对于稳定的训练是很有帮助的。

### 对话式格式

对话式数据集训练模型进行多轮对话。数据不以单一指令和响应的形式，而是以一系列交流的形式组织。这种格式对于构建聊天机器人、虚拟助手或任何必须在多次交互中保持上下文 (context)的应用来说都很必要。

数据通常表示为回合列表，其中每个回合都指定说话者的角色（`user`或`assistant`）及其消息（`content`）。许多开源模型（如Llama）都是使用这种结构的数据进行微调 (fine-tuning)的。

以下是一个用于训练技术支持助手的两轮对话示例：

```json
{
  "messages": [
    {
      "role": "用户",
      "content": "我正在尝试连接到我的新数据库，但收到“连接超时”错误。我应该首先检查什么？"
    },
    {
      "role": "助手",
      "content": "“连接超时”错误通常指向网络或防火墙问题。您能确认您的机器可以访问数据库服务器的IP地址，并且您与服务器之间的任何防火墙上端口5432已打开吗？"
    },
    {
      "role": "用户",
      "content": "我可以ping通IP，但我怎么检查端口呢？"
    },
    {
      "role": "助手",
      "content": "您可以使用`telnet`或`nc`（netcat）这样的工具。例如，在命令行中，您可以运行`telnet YOUR_DATABASE_IP 5432`。如果连接成功，则端口是开放的；否则，它很可能被阻止了。"
    }
  ]
}
```

这个完整的消息列表构成一个训练示例。模型学习根据整个之前的对话历史来生成`assistant`的回复。处理过程中，回合列表会被转换为一个格式化字符串，通常使用基础模型分词 (tokenization)器 (tokenizer)定义的特殊标记 (token)来区分角色和回合。

### 比较数据结构

这两种格式的主要区别在于它们对上下文 (context)的表示。指令式格式是无状态的，将每个示例视为一个独立任务。对话式格式是有状态的，训练模型基于对话中的先前回合进行回复。

> 该图说明了指令式（单轮）和对话式（多轮）数据格式的信息流。

### 如何选择合适的格式

您选择的数据格式应完全取决于最终应用的需求。

- **在以下情况使用指令式格式：**

  - 任务是独立的，不依赖于之前的交互。例如文档总结、情感分析或根据规范生成代码。
  - 您希望为明确定义、可重复的功能构建一个可靠工具。
  - 输入和输出明确定义且独立完整。
- **在以下情况使用对话式格式：**

  - 模型必须记住交互的先前部分才能连贯地回应。
  - 应用是聊天机器人、客户支持代理或角色扮演角色。
  - 交互预期是多轮演进的对话。

也可以将单轮任务构建到对话式结构中。例如，一条指令可以简单地成为两轮对话中的第一个`user`消息。然而，对于严格面向任务的微调 (fine-tuning)，指令式格式更为直接，通常也更易于构建。选择合适的数据结构是确保模型学习您打算实现的行为的第一步。

## 参考资料

- [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  Publisher: Stanford University
  第28章对聊天机器人和对话系统进行了讨论，涵盖了多轮对话AI的基础概念。
- [How to format inputs for chat models](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) — OpenAI (2024)
  Publisher: OpenAI
  提供了使用`messages`数组格式构建OpenAI聊天补全模型的会话数据的官方指南和示例。
