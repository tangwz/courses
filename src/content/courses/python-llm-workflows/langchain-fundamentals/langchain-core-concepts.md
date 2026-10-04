---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "langchain-core-concepts"
sourceId: 5138
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/langchain-core-concepts"
title: "主要构成：模型、提示词、输出解析器"
description: "了解 LangChain 应用的基本组成部分。"
order: 2
plots: []
sourceHash: "029b0a4060f4085d1547f16cc06123a9a1f7ede1e44956375c156384f7b8ff7e"
sourceCorrections: []
---

LangChain 围绕模型、提示词 (prompt)和输出解析器这三个主要组成部分来组织与大型语言模型 (LLM) 的交互。在 Python 中构建有效的 LLM 工作流，需要了解这些组成部分如何运行和相互配合。可以将它们视为构建 LLM 应用的必要成分。

### 模型：语言引擎接口

在 LangChain 中，`模型`是一个抽象层，提供了一个标准化接口，用于与各种语言模型进行交互。这种设计使得您可以在不同的模型提供方（如 OpenAI、Anthropic、Cohere 或通过 Hugging Face 的开源模型）之间切换，只需极少的代码改动。LangChain 区分两种主要类型的语言模型：

1. **LLMs**：这些封装了接受字符串输入并返回字符串输出的模型。它们适合直接的文本补全任务。
2. **聊天模型**：这些封装了以一系列聊天消息作为输入并返回聊天消息作为输出的模型。这种基于消息的结构更符合对话式人工智能，并允许在交互历史中存在不同的角色（系统、人类、AI），通常能为基于对话的任务带来更好的表现。

实例化模型通常很简单。例如，要使用 OpenAI 聊天模型，您首先需要确保已安装所需的库（`pip install langchain-openai`）并配置好您的 API 密钥（如第 2 章所述）。然后，您可以导入并初始化模型：

```python
from langchain_openai import ChatOpenAI

# 假设 OPENAI_API_KEY 环境变量已设置
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# 示例用法（我们将在后面介绍调用方法）
# response = llm.invoke("讲个短笑话。")
# print(response.content)
```

这里的 `temperature` 参数 (parameter)控制输出的随机性；较低的值使输出更确定，而较高的值会增加创造性（和潜在的可变性）。不同的模型和提供方提供了各种参数来微调 (fine-tuning)模型行为。LangChain 提供的主要优势就是这种一致的接口，无论底层模型提供方是谁。

### 提示词 (prompt)：引导模型的响应

模型提供了引擎，而 `提示词` 则是您操控引擎的方式。提示词是提供给语言模型的输入，经过精心设计以引出期望的响应。LangChain 提供了强大的工具来构建和管理提示词，主要通过 `PromptTemplate` 对象实现。

`PromptTemplate` 有助于根据用户输入或其他变量创建动态提示词。它定义了一个包含变量占位符的模板字符串，并指定预期哪些变量。

考虑一个任务，您希望 LLM 用简单的语言解释一个编程原理。您可以创建这样一个模板：

```python
from langchain_core.prompts import PromptTemplate

template_string = """
用简单的语言解释编程原理 '{concept}'，适合初学者。
聚焦核心思想，并提供一个简短、易于理解的类比。
"""

prompt_template = PromptTemplate(
    input_variables=["concept"],
    template=template_string
)

# 使用特定概念格式化提示词
formatted_prompt = prompt_template.format(concept="API")

# print(formatted_prompt)
# 输出将是模板字符串，其中 {concept} 被 "API" 替换
```

这种模板化方法使您的代码更清晰，并让您轻松管理发送给 LLM 的复杂指令。LangChain 也支持更复杂的模板化，例如用于聊天模型的 `ChatPromptTemplate`，它将输入构造为一系列消息（例如，设置上下文 (context)的系统消息，后跟包含查询的人类消息）。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的助手，能够简单地解释复杂主题。"),
    ("human", "解释 '{concept}' 的含义。")
])

formatted_messages = chat_template.format_messages(concept="Machine Learning")

# print(formatted_messages)
# 输出: [SystemMessage(content='...'), HumanMessage(content='解释 'Machine Learning' 的含义。')]
```

有效的提示词使用是艺术与科学的结合，第 8 章将讨论各种技术。LangChain 的提示词工具提供了以编程方式实现这些技术所需的结构。

### 输出解析器：结构化 LLM 的输出

LLM 通常生成非结构化文本。然而，应用程序通常需要特定格式的数据，例如 JSON、列表或自定义 Python 对象。`输出解析器` 是 LangChain 组件，旨在弥合这一差距。它们执行两个主要功能：

1. **指导模型**：它们可以在发送给 LLM 的提示词 (prompt)中附加格式化指令，引导其生成解析器能理解的输出。
2. **解析输出**：它们获取 LLM 的原始文本输出，并将其解析为期望的结构。

LangChain 提供了几种内置的输出解析器，包括：

- `CommaSeparatedListOutputParser`：将逗号分隔的列表解析为 Python 列表。
- `StructuredOutputParser`：根据预定义模式提取信息，返回一个字典。
- `PydanticOutputParser`：将输出解析为 Pydantic 模型，提供强类型和验证。

我们用一个简化的 `PydanticOutputParser` 场景来说明。假设您希望 LLM 从一个句子中提取一个人的姓名和年龄，并将其作为结构化对象返回。

```python
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# 1. 定义期望的数据结构
class PersonInfo(BaseModel):
    name: str = Field(description="这个人的姓名")
    age: int = Field(description="这个人的年龄")

# 2. 设置解析器
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 3. 创建一个包含格式化指令的提示词模板
prompt = PromptTemplate(
    template="从文本中提取相关信息。\n{format_instructions}\n文本：{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# 4. 实例化模型
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 5. 格式化输入
input_query = "安娜 30 岁，是一名软件工程师。"
formatted_input = prompt.format_prompt(query=input_query)

# 6. 获取模型的输出（字符串形式）
# 在实际的链中，这会自动发生，但这里展示步骤：
# output_str = model.invoke(formatted_input).content

# 7. 解析输出（示例输出字符串可能看起来像 JSON）
# example_output_str = '{"name": "安娜", "age": 30}'
# parsed_output = parser.parse(example_output_str)

# print(parsed_output)
# 输出: name='安娜' age=30
```

在此示例中，`parser.get_format_instructions()` 生成文本，告知 LLM 如何格式化其响应（通常要求与 `PersonInfo` 模式匹配的 JSON）。然后，解析器使用这种期望的格式将 LLM 的字符串输出转换为 `PersonInfo` 对象。

### 连接各个组成部分

这三个组成部分，模型、提示词 (prompt)和输出解析器，很少单独工作。LangChain 的强大之处在于将它们连接起来，通常以被称为“链”的序列形式。最基本的流程涉及接收用户输入，使用 `PromptTemplate` 对其进行格式化，发送给 `模型`，然后使用 `输出解析器` 结构化 `模型` 的响应。

> 一个基本的 LangChain 流程：输入由提示词模板格式化，由模型处理，结果由输出解析器结构化。

掌握这些主要组成部分非常重要。它们是 LangChain 更高级构造（例如复杂的链和代理）的重要支撑，我们将在下一章讨论。通过了解如何与模型对接、精心设计有效的提示词以及解析产生的输出，您就获得了构建可预测的 LLM 驱动应用程序的必要工具。

## 参考资料

- [LangChain Documentation: Core Components & LangChain Expression Language (LCEL)](https://python.langchain.com/docs/) — LangChain Development Team (Accessed 2024)
  Publisher: LangChain
  LangChain官方文档，涵盖了LangChain工作流的基本构建模块，包括模型、提示词和输出解析器，以及它们如何通过LCEL结合使用。
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); DOI: [10.48550/arXiv.2005.14165](https://doi.org/10.48550/arXiv.2005.14165)
  这篇开创性论文展示了大型语言模型的能力，尤其是它们对提示词的敏感性以及通过少量示例执行任务的能力，为提示工程奠定了基础。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Learn Prompting Community (Accessed 2024)
  一个全面的社区驱动资源，用于理解和应用提示工程技术，与有效引导LLM响应高度相关。
