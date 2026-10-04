---
course: "building-llm-apps-with-langchain"
chapter: "models-prompts-parsers"
lesson: "interfacing-with-llms-chat-models"
sourceId: 7410
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers/interfacing-with-llms-chat-models"
title: "与LLM和聊天模型交互"
description: "了解LangChain中普通LLM与聊天模型的区别，以及如何针对特定任务使用它们。"
order: 1
plots: []
sourceHash: "67c07a823e37d62ac0445ea09bdd4ddf3ca7fc0c87a91f91a57a81e94311e87d"
sourceCorrections: []
---

LangChain提供了一个统一的接口，用于与多种语言模型交互，屏蔽了OpenAI、Anthropic或Hugging Face等提供商之间具体的API差异。这种标准化让您只需少量代码改动即可在模型间切换。此抽象的核心是两种主要类型的模型：`LLM`和`Chat Model`。了解它们的不同是构建高效应用的第一步。

### LLM接口：直接的文本输入，文本输出模型

`LLM`类代表了与语言模型最直接的交互方式。它基于简单的文本输入、文本输出模式运行。您提供一个字符串作为提示，模型返回一个字符串作为补全或答案。此接口最适合用于设计为文本补全的模型，例如旧模型或经过特定指令调整的补全模型。

将此接口视为一个直接的函数调用，其中输入和输出都是纯文本。

```python
# 请确保安装所需的包：
# pip install langchain-openai

import os
from langchain_openai import OpenAI

# 设置您的API密钥
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 实例化模型
# 我们指定'gpt-3.5-turbo-instruct'，它专为此补全式接口设计。
llm = OpenAI(model="gpt-3.5-turbo-instruct")

# 准备输入字符串
prompt = "What is the capital of Japan?"

# 调用模型
response = llm.invoke(prompt)

print(response)
# 预期输出：
# 日本的首都是东京。
```

`invoke`方法将提示发送到模型的API，并直接返回生成的文本。这种方法对于摘要、翻译或回答事实性问题等单轮任务很高效，而无需会话背景。

### ChatModel接口：管理结构化对话

相对而言，`ChatModel`接口专为更复杂的对话式交互设计。大多数现代且功能强的模型，例如GPT-4或Claude 3，都为聊天格式做了优化。`ChatModel`接收一个消息对象列表作为输入，而非单个字符串。这种结构让模型能够明白对话背景，包括谁说了什么。

消息主要有三种类型：

- **`SystemMessage`**：这设定AI的整体基调和指令。它就像给模型一个角色设定或一套在整个对话中要遵循的规则。它通常是列表中的第一个消息。
- **`HumanMessage`**：这代表用户输入。
- **`AIMessage`**：这代表模型自身的先前回复。通过在输入中包含`AIMessage`对象，您可以为模型提供对话历史。

让我们看看这如何与`ChatOpenAI`一起使用，它是OpenAI模型的聊天专用集成。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 实例化聊天模型
chat = ChatOpenAI(model="gpt-4o")

# 准备消息列表
messages = [
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Hello, how are you?"),
]

# 调用模型
response = chat.invoke(messages)

print(response)
# 预期输出：
# AIMessage(content='Bonjour, comment allez-vous ?', response_metadata=...)

# 若要获取文本内容，请访问 .content 属性
print(response.content)
# 预期输出：
# Bonjour, comment allez-vous ?
```

请注意，输出不是字符串，而是一个`AIMessage`对象。此对象在其`.content`属性中包含字符串内容，以及来自API响应的其他有用元数据。这种结构化的输入和输出使得`ChatModel`成为构建聊天机器人、多步推理 (inference)代理或任何对话历史很要紧的应用的首选。

### 两种接口的比较

这两种模型类型之间的区分反映了LLM的发展。早期模型主要是文本补全器，而现代模型则被微调 (fine-tuning)为会话助手。下图描绘了每种接口的不同数据流。

> `LLM`接口处理单个字符串，而`ChatModel`接口处理结构化的消息列表，使其适用于对话情境。

以下是区别汇总：

| 特点 | LLM接口 | ChatModel接口 |
| --- | --- | --- |
| **输入格式** | 单个文本字符串。 | 一个消息对象列表（`SystemMessage`、`HumanMessage`等）。 |
| **输出格式** | 单个文本字符串。 | 一个包含内容和元数据的`AIMessage`对象。 |
| **常见用途** | 简单的文本补全、摘要、单轮问答。 | 对话代理、多轮对话、角色扮演场景。 |
| **底层模型** | 为文本补全做了优化（例如，`gpt-3.5-turbo-instruct`）。 | 为遵循指令和对话做了优化（例如，`gpt-4o`、`claude-3-sonnet`）。 |

一般来说，您应优先使用`ChatModel`，因为它们是大多数现代高性能语言模型的标准。

在明确了如何与模型沟通后，下一步是掌握如何制定指令。现在我们将关注`PromptTemplates`，它能让您为这些模型构建动态且可复用的输入。

## 参考资料

- [Models](https://python.langchain.com/docs/modules/model_io/language_models/) — LangChain Contributors (2023)
  官方文档，解释了 LangChain 中 LLM 和 ChatModel 接口及其用法。
- [Chat Completions API](https://platform.openai.com/docs/guides/gpt/chat-completions-api) — OpenAI (2024)
  Publisher: OpenAI
  提供 OpenAI 聊天模型的规范，详细说明 LangChain ChatModel 接口所抽象的消息格式（系统、用户、助手）。
- [Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-claude-api) — Anthropic (2024)
  Publisher: Anthropic
  Anthropic Claude 模型的官方文档，展示了另一个主要供应商与 LangChain ChatModel 集成的会话式 AI 方法。
