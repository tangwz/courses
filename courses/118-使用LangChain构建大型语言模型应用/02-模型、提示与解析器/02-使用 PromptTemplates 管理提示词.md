# 使用 PromptTemplates 管理提示词

来源：[原文](https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers/managing-prompts-with-prompt-templates)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管可以使用 Python 的 f-字符串手动格式化字符串来构建提示词 (prompt)，但随着应用程序的复杂度提升，这种方式会变得难以处理。需要一种更有组织的方法来处理多个输入变量，确保其可重复使用性，并将提示词的逻辑与应用程序代码明确分开。LangChain 提供了 `PromptTemplate` 对象，通过规范化提示词的创建来解决这个问题。

`PromptTemplate` 是一个可重复使用的对象，用于生成提示词。它包含一个模板字符串和在格式化提示词时所需的一组输入变量。

### 创建一个基本 PromptTemplate

最简单的 `PromptTemplate` 接受一个模板字符串和一组变量。模板字符串使用 Python 的 f-字符串语法作为其占位符。

我们来为简单的语言翻译器创建一个模板。

```python
from langchain_core.prompts import PromptTemplate

template_string = "Translate the following text from {input_language} to {output_language}: {text}"

prompt_template = PromptTemplate(
    input_variables=["input_language", "output_language", "text"],
    template=template_string,
)

# 也可以使用便捷的 from_template 类方法
# prompt_template = PromptTemplate.from_template(template_string)

formatted_prompt = prompt_template.format(
    input_language="English",
    output_language="Spanish",
    text="Hello, how are you?"
)

print(formatted_prompt)
# 输出：Translate the following text from English to Spanish: Hello, how are you?
```

这个格式化后的字符串可以直接传递给 LLM。这里的主要优点是清晰性和安全性。如果您尝试在未提供所有 `input_variables` 的情况下格式化 `PromptTemplate`，它将引发错误，从而防止不完整的提示词 (prompt)发送给模型。

> 使用 PromptTemplate 为模型生成最终提示词的流程。

### 使用 ChatPromptTemplate 处理聊天模型

聊天模型与标准 LLM 的运行方式不同。它们不期望单个字符串，而是期望一系列聊天消息，每条消息都有特定的角色（如 `system`、`human` 或 `ai`）。为了适应这一点，LangChain 提供了 `ChatPromptTemplate`。

`ChatPromptTemplate` 是从消息模板列表中创建的。这使您能够构建多轮对话，并为提示词 (prompt)的不同部分分配独特的角色。

- **系统消息 (System Message)：** 此消息为 AI 设置上下文 (context)和指令。它告知模型应如何表现或采用何种角色。它通常是序列中的第一条消息。
- **用户消息 (Human Message)：** 这表示用户的输入或查询。
- **AI 消息 (AI Message)：** 这表示模型先前的响应，对于构建对话历史很有用。

以下是如何为我们的翻译器创建 `ChatPromptTemplate` 的方法。

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert translator. Translate the user's text from {input_language} to {output_language}."),
        ("human", "{text}"),
    ]
)

formatted_messages = chat_template.format_messages(
    input_language="English",
    output_language="Japanese",
    text="This is a test."
)

print(formatted_messages)
# 输出：
# [
#     SystemMessage(content='You are an expert translator. Translate the user's text from English to Japanese.'),
#     HumanMessage(content='This is a test.')
# ]
```

输出是一个包含 `SystemMessage` 和 `HumanMessage` 对象的列表，这正是像 `ChatOpenAI` 这样的聊天模型所期望的格式。这种有组织的方法比手动创建消息对象列表更规范，并且不易出错。

### 使用部分格式化预填充模板变量

有时，您可能希望通过预先填充一些变量，从通用模板创建一个更专业的模板。这被称为提示词 (prompt)的“部分化”。它有助于创建可重复使用的组件，或者当您事先知道某些输入值时非常有用。

比如，我们可以使用我们通用的 `chat_template` 来创建一个专用的英德翻译器。

```python
# 沿用上一个例子

# 通过部分化原始模板来创建一个专用模板
german_translator_template = chat_template.partial(
    input_language="English", 
    output_language="German"
)

# 现在，您只需提供剩余的变量：`text`
german_messages = german_translator_template.format_messages(
    text="That is a wonderful idea."
)

print(german_messages)
# 输出：
# [
#     SystemMessage(content="You are an expert translator. Translate the user's text from English to German."),
#     HumanMessage(content='That is a wonderful idea.')
# ]
```

通过使用 `partial`，我们创建了一个新的、更具体的 `ChatPromptTemplate`，它只需要 `text` 变量。这项技术通过减少冗余，并允许您以模块化的方式组合提示词模板，从而简化了您的代码。

使用 `PromptTemplate` 和 `ChatPromptTemplate` 对构建可扩展的 LangChain 应用程序是不可或缺的。它们提供了有效处理提示词所需的结构，使您的代码更整洁、更具可重用性，并且更易于调试。随着我们继续，您会发现这些模板是您构建几乎每个链和代理的起点。

## 参考资料

- [LangChain Expression Language (LCEL) | Prompts](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEFruBGhUzbrYNhnOWmEJ5vqFMu23z1yTbloSur9PqAMaD4SKtPKGlPEzytR8LelnMIjfhGuf5DIUeypZNsbDgOoEHKG721kO2YeBnDe3KzJTzeVBof0BKfPxwLA1LRBFChtP8zi-viHXzwuA==) — LangChain (2024)
  Publisher: LangChain
  LangChain官方文档，解释了PromptTemplate和ChatPromptTemplate对象及其用法，以及在LangChain表达式语言中的作用。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) —  (2024)
  一份全面的社区提示工程指南，为大型语言模型设计有效提示提供了更广阔的背景。
- [Chat completions](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI官方Chat Completions API文档，说明了ChatPromptTemplate所抽象的消息格式（系统、用户、助手角色）。

---

[上一节](01-%E4%B8%8ELLM%E5%92%8C%E8%81%8A%E5%A4%A9%E6%A8%A1%E5%9E%8B%E4%BA%A4%E4%BA%92.md) · [下一节](03-%E5%AE%9E%E7%8E%B0%E5%B0%91%E6%A0%B7%E6%9C%AC%E6%8F%90%E7%A4%BA.md)
