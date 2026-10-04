---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "langchain-llm-providers"
sourceId: 5140
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/langchain-llm-providers"
title: "在 LangChain 中使用不同的 LLM 服务"
description: "使用 LangChain 整合各种 LLM 服务（OpenAI、Hugging Face Hub 等）。"
order: 3
plots: []
sourceHash: "41a497449fa76b21d41d05b51f58a2e03b68f2fdf8360e2ce38c37153d1d171f"
sourceCorrections: []
---

LangChain 旨在提供一个标准接口，用于与各种大型语言模型（LLM）进行交互。这种设计很有价值，因为不同的 LLM 服务商通常有不同的 API、认证方法和配置参数 (parameter)。通过使用 LangChain 的模型组件，您可以编写不那么依赖于特定底层 LLM 服务的应用程序逻辑。

这一点上，LangChain 的模型基类是：

1. **`LLM`**：此类主要用于执行文本补全的模型。您提供一个文本提示，模型会返回一个补全后的文本字符串。
2. **`ChatModel`**：此类专为优化对话交互的模型设计。与单个字符串提示不同，您通常提供一系列聊天消息（通常带有“系统”、“人类”、“AI”等角色），模型会返回一条聊天消息作为输出。

尽管它们的输入/输出格式为匹配底层模型类型而略有不同，但 LangChain 仍力求在 `LLM` 和 `ChatModel` 的集成中提供一致的方法，例如 `invoke`、`stream` 和 `batch`。

### 整合特定服务商

LangChain 通过专门的集成包支持多种 LLM 服务商。您通常需要安装基础的 `langchain` 库以及您所需服务商的特定软件包（例如 `langchain-openai`、`langchain-huggingface`）。

让我们看看如何为几个常见的服务商实例化模型对象。

#### OpenAI

OpenAI 模型（如 GPT-4、GPT-3.5 Turbo）得到广泛使用。LangChain 提供 `OpenAI`（用于较旧的补全端点）和 `ChatOpenAI`（用于聊天补全端点，这现在是标准方式）。

要使用 `ChatOpenAI`，首先确保您已安装必要的软件包：

```bash
pip install langchain-openai
```

您还需要设置您的 OpenAI API 密钥。正如第 2 章中讨论的，标准且安全的操作是将其设置为名为 `OPENAI_API_KEY` 的环境变量。LangChain 会自动查找此变量。

```python
import os
from langchain_openai import ChatOpenAI

# 确保您的 OPENAI_API_KEY 已设置为环境变量
# 示例：os.environ["OPENAI_API_KEY"] = "your_api_key_here" # 不建议用于生产环境

# 初始化 ChatOpenAI 模型实例
# 默认情况下，它使用 OPENAI_API_KEY 环境变量
# 您可以指定模型名称、温度等
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 调用示例（ChatModels 期望消息）
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="您是一个将英语翻译成法语的有用助手。"),
    HumanMessage(content="翻译这句话：'Hello, how are you?'")
]

# 使用 invoke 方法进行单次调用
response = llm.invoke(messages)

print(response.content)
# 预期输出（可能略有不同）：Bonjour, comment ça va?
```

请注意，初始化涉及创建特定于服务商的类 (`ChatOpenAI`) 的实例，并传递诸如 `model` 和 `temperature` 等配置参数 (parameter)。实际的交互使用标准的 `invoke` 方法。

#### Hugging Face Hub

Hugging Face 托管了大量开源模型。LangChain 允许通过 `HuggingFaceHub` 类与 Hugging Face Hub 上托管的模型进行交互。

首先，安装所需软件包：

```bash
pip install langchain-huggingface huggingface_hub
```

您需要一个 Hugging Face API 令牌。将其设置为 `HUGGINGFACEHUB_API_TOKEN` 环境变量。

```python
import os
from langchain_huggingface import HuggingFaceHub

# 确保您的 HUGGINGFACEHUB_API_TOKEN 已设置为环境变量
# 示例：os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_token_here"

# 初始化 HuggingFaceHub LLM 实例
# 您必须指定模型仓库 ID
# 示例：使用一个较小的免费模型进行演示
hf_llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature": 0.8, "max_length": 64}
)

# 调用示例（LLM 类期望一个字符串提示）
prompt = "将英语翻译成法语：'Hello, how are you?'"

# 使用 invoke 方法
response = hf_llm.invoke(prompt)

print(response)
# 预期输出（可能因模型而异）：Bonjour, comment allez-vous?
```

在这里，我们使用了 `LLM` 集成 (`HuggingFaceHub`)。我们为所需模型指定了 `repo_id`，并通过 `model_kwargs` 传递了模型特有的参数。同样，即使服务商和模型类型与 OpenAI 示例不同，交互仍使用标准的 `invoke` 方法。

### 抽象的优点

使用 LangChain 模型集成的主要优点是一致的接口。

> LangChain 提供封装类（如 `ChatOpenAI`、`HuggingFaceHub`），它们实现了通用接口（`LLM` 或 `ChatModel`）。您的应用程序代码与此标准接口交互，而封装类则处理与实际服务商 API 通信的具体细节。

这种设计让您可以：

1. **更容易切换服务商：** 如果您最初使用一个服务商（例如 OpenAI），后来决定切换到另一个（例如通过 Hugging Face 库或 Ollama 访问的自托管模型），您主要只需更改代码中模型实例化的部分。使用 `invoke`、`stream` 等的核心逻辑通常保持不变或只需少量修改。
2. **试验不同的模型：** 您可以通过在初始化时简单地更改 `model` 参数 (parameter)，来测试各种模型，即使是来自同一个服务商的模型（例如，在 `gpt-4` 和 `gpt-3.5-turbo` 之间切换）。
3. **一致的配置：** LangChain 尝试标准化 `temperature` 或 `max_tokens` 等常见参数，尽管特定于服务商的参数可能仍然存在（通常通过 `model_kwargs` 传递）。

尽管由于模型能力和特定 API 功能的差异，完美的互换性并非总能实现，但 LangChain 显著减少了使用不同 LLM 后端的麻烦。在接下来的部分中，当您使用提示和输出解析器构建更复杂的工作流程时，这种一致的模型接口会变得越来越有用。

## 参考资料

- [OpenAI API Reference: Chat Completions](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  Publisher: OpenAI
  详细介绍了用于创建聊天补全的 OpenAI 原生 API，为 LangChain 的 ChatOpenAI 集成提供了背景信息。
- [Hugging Face Inference API Documentation](https://huggingface.co/docs/api-inference/) — Hugging Face (2024)
  Publisher: Hugging Face
  概述了 Hugging Face 用于托管模型的推理 API，LangChain 的 HuggingFaceHub 类使用此 API 进行集成。
