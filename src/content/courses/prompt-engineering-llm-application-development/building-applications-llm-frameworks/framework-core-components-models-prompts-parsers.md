---
course: "prompt-engineering-llm-application-development"
chapter: "building-applications-llm-frameworks"
lesson: "framework-core-components-models-prompts-parsers"
sourceId: 3045
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/framework-core-components-models-prompts-parsers"
title: "核心构成部分：模型、提示、解析器"
description: "运用模型接口、提示模板和输出解析器等基本构建模块。"
order: 2
plots: []
sourceHash: "500f61ab0b15f921861e1fd6c916cd7e1dfc07151ef8bf17884240817a0bf805"
sourceCorrections: []
---

直接与LLM API交互能够提供精细的控制，但通常需要编写重复的代码来格式化提示、处理API调用和解析响应。随着应用规模的扩大，手动管理会话历史、整合外部工具或将多个LLM调用串联起来会变得繁琐。

像LangChain这样的LLM开发框架，通过提供标准化的构建模块和连接它们的方式来解决这些问题。无需为每次交互编写样板代码，你可以使用更高级别的抽象。我们来考察这些构建模块中最基础的部分：模型、提示（特别是提示模板）和输出解析器。可以把这些看作是在框架内构建LLM驱动逻辑的必要组成部分。

### 模型：引擎接口

任何LLM应用的核心都是模型本身。框架将与各种LLM供应商（如OpenAI、Anthropic、Cohere，甚至本地部署的模型）交互的具体细节抽象化，统一到一致的`Model`接口背后。这通常有两种主要类型：

1. **LLMs：** 这些封装了接收字符串作为输入并返回字符串的模型。它们更简单，适用于直接的文本补全任务。
2. **ChatModels：** 这些封装了为对话交互优化的模型。它们接收一个聊天消息列表（通常包含“系统”、“人类”、“AI”等角色）作为输入，并返回一个聊天消息。这种结构更适合构建聊天机器人和代理。

使用框架的模型抽象简化了在不同底层LLM或API供应商之间切换的操作，只需极少的代码改动。你初始化一个模型对象，可能指定模型名称（例如`gpt-3.5-turbo`）和`temperature`等参数 (parameter)，然后通过`invoke()`或`predict()`等标准化方法与它进行交互。

```python
# LangChain类语法的示例
# 假设已完成必要的导入和API密钥设置
from langchain_openai import ChatOpenAI # 或其他供应商，例如langchain_anthropic

# 初始化模型（供应商细节已抽象化）
chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 使用标准化方法进行交互
prompt = "解释LLM框架的理念。"
# 框架负责构建API请求和解析基本响应
response = chat_model.invoke(prompt)

# 响应对象通常具有结构，例如'content'属性
print(response.content)
# 预期输出（示例）：LLM框架提供可重用组件和结构，以简化构建由大型语言模型驱动的应用。
```

这种统一接口处理了你之前在第四章直接使用`requests`等库时手动管理的底层API调用、认证、请求格式化和基本响应处理。

### 提示模板：构建你的输入

虽然你可以将原始字符串作为提示传递给模型接口，但实际应用通常需要从用户输入或其他数据源构建动态提示。在简单情况下，使用f-string或基本字符串拼接手动构建这些字符串是可行的，但对于复杂的、多部分的提示，它很快就会导致代码可读性差且容易出错。

框架引入了`提示模板`来解决这个问题。提示模板本质上是一种预定义结构，用于生成提示。它包含：

- **指令：** 描述LLM任务的固定文本。
- **输入变量：** 用于动态值的占位符（通常表示为`{variable_name}`），这些值在使用模板时将被插入。
- **格式化指令（可选）：** 关于最终输出应如何呈现的明确指导。虽然有用，但专门的输出解析器通常能更可靠地处理复杂的格式化要求。

你只需定义一次模板，然后重复使用它，每次为输入变量提供不同的值。这提升了代码的清晰度、可维护性和可重用性。

```python
# LangChain类语法的示例
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 示例1：LLM的基本字符串模板
template_string = """将以下英文文本翻译成法文：

文本：{english_text}
翻译："""
# 创建模板对象
prompt_template = PromptTemplate.from_template(template_string)

# 使用特定输入格式化模板
formatted_prompt = prompt_template.format(english_text="Hello!")
print("格式化的字符串提示：")
print(formatted_prompt)
# 预期输出：
# 格式化的字符串提示：
# 将以下英文文本翻译成法文：
#
# 文本：Hello!
# 翻译：

# 示例2：聊天模板（为聊天模型设计，结构更完善）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，能将{input_language}翻译成{output_language}。"),
    ("human", "{text}") # 用户的输入在此
])

# 使用角色的特定输入和文本格式化模板
formatted_messages = chat_template.format_messages(
    input_language="English",
    output_language="Spanish",
    text="How are you?"
)
print("\n格式化的聊天消息：")
print(formatted_messages)
# 预期输出：
# 格式化的聊天消息：
# [SystemMessage(content='你是一个有用的助手，能将英语翻译成西班牙语。'), HumanMessage(content='How are you?')]

# 接着你可以将'formatted_prompt'（字符串）传递给LLM的invoke方法，
# 或将'formatted_messages'（消息列表）传递给ChatModel的invoke方法。
```

使用模板将提示的固定结构与可变数据分离，使得管理和修改提示变得非常容易，且无需更改核心应用逻辑。聊天提示模板特别有效，因为它们直接符合大多数现代聊天模型期望的对话消息列表结构。

### 输出解析器：塑造LLM的响应

LLM本质上生成文本序列。正如在第二章中探讨并在第七章中回顾的那样，尽管指令遵循和少样本提示等技术可以*鼓励*LLM以特定格式（如JSON或编号列表）生成输出，但不能绝对保证模型每次都能完美遵循。仅仅依靠提示工程 (prompt engineering)来获得严格的输出结构可能会使你的应用脆弱，并在LLM响应略微偏离时容易出错。

这就是`输出解析器`变得不可或缺的地方。它们是专门设计的组件，旨在接收LLM生成的原始字符串输出，并将其转换为应用代码中更易于使用的结构化格式，例如Python字典、列表或自定义数据对象（如Pydantic模型）。

输出解析器的主要功能通常包括：

1. **指令生成：** 有些解析器会自动生成并将格式化指令附加到发送给LLM的提示中。这些指令明确告知模型如何格式化其响应，以便解析器能够理解。
2. **解析逻辑：** 它们包含根据预期格式解析LLM原始字符串输出所需的代码。这可能涉及查找和验证JSON、按分隔符拆分字符串，或使用正则表达式。
3. **错误处理：** 优秀的解析器提供机制来处理LLM输出不符合预期格式的情况。这可能涉及引发特定错误、尝试修正输出或触发重试机制（稍后介绍）。

框架通常提供多种预构建的解析器，以满足常见使用场景：

- **结构化输出解析器 / Pydantic解析器：** 对于将输出直接解析为由模式定义的Python对象（如Pydantic模型，将在第七章进一步讨论）非常有用。
- **JSON解析器：** 专门设计用于从LLM可能混乱的字符串输出中查找和提取有效的JSON对象。
- **列表解析器：** 对于LLM应返回项目列表（例如，逗号分隔、项目符号）的任务非常有用。
- **日期时间解析器：** 提取日期和时间信息。

> 输入变量填充提示模板，生成格式化提示。模型处理此提示，生成原始输出文本。输出解析器解析此文本，可能使用其注入到原始提示中的指令，从而生成可供应用使用的结构化输出。

通过将输出解析器整合到你的工作流程中，你可以显著提升LLM应用的可靠性和可预测性。与其自己编写复杂且可能脆弱的字符串操作或正则表达式逻辑，不如借助专门的、经过测试的组件来处理LLM输出的特定任务，这使得你的应用能够有效处理来自模型生成结果的结构化数据。

这些核心构成部分——模型、提示模板和输出解析器——构成了在框架内构建更复杂LLM应用逻辑的必要基础。它们抽象了低级细节，并提供可重用部分来构建复杂的流程。在下一节中，我们将介绍“链”（Chains），这是框架中用于将这些组件（及其他）连接起来执行一系列操作的机制。

## 参考资料

- [LangChain Core Building Blocks: LLMs, ChatModels, Prompts, OutputParsers](https://python.langchain.com/v0.2/docs/concepts/#model-io) — LangChain Team (2024)
  LangChain核心组件（包括模型、提示模板和输出解析器）的官方文档。
- [LLM Systems: A Practical Guide to Building Applications with Large Language Models](https://www.oreilly.com/library/view/llm-systems/9781098150498/) — Scott P. McDonald, Andrew C. P. Sledd, and Jeremy A. Howard (2024)
  Publisher: O'Reilly Media
  一本涵盖LLM驱动应用程序架构和实现的实用指南，包括模型和提示设计等核心组件。
- [Function calling and other API updates](https://openai.com/blog/function-calling-and-other-api-updates) — OpenAI (2023)
  Journal: OpenAI Blog; Publisher: OpenAI
  讨论模型如何生成结构化的JSON输出以供工具使用，这是框架为可靠的结构化响应而抽象的一种方法，补充了显式输出解析器。
