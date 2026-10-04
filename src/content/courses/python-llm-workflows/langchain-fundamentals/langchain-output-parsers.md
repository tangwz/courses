---
course: "python-llm-workflows"
chapter: "langchain-fundamentals"
lesson: "langchain-output-parsers"
sourceId: 5146
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-4-langchain-fundamentals/langchain-output-parsers"
title: "解析大型语言模型输出结构"
description: "使用 LangChain 输出解析器从大型语言模型响应中提取结构化数据。"
order: 5
plots: []
sourceHash: "69df75a8bb2a8e009ffa65bdda59a37f219c1b9677d4c7267d0d84594a51e0ec"
sourceCorrections: []
---

虽然大型语言模型（LLM）擅长生成人类可读的文本，但应用程序通常需要更结构化的数据格式。一个大型语言模型可能会返回一段完美连贯的描述人物的文字，但您的应用程序可能需要将人物的姓名、职位和地点作为单独的字段。这就是 LangChain 的输出解析器发挥作用的地方。

输出解析器是旨在组织大型语言模型文本输出的类。它们主要通过两种方式工作：

1. **引导大型语言模型：** 许多解析器可以生成特定的格式指令，这些指令会附加到您的提示中。这些指令引导大型语言模型生成解析器能理解的格式的输出（例如，“将您的答案作为 JSON 对象返回，包含 'name' 和 'age' 两个键。”）。
2. **解析输出：** 一旦大型语言模型响应，解析器会接收原始文本字符串，并将其转换为所需的 Python 数据结构，例如字典、列表或自定义对象。

让我们查看 LangChain 中一些常用的输出解析器。

### SimpleJsonOutputParser

顾名思义，`SimpleJsonOutputParser` 旨在解析大型语言模型输出中的简单 JSON 对象。当您需要一个直接的字典结构时，它非常有用。

```python
# 假设 'llm' 是一个已初始化的 LangChain 大型语言模型实例
# 并且已导入 'ChatPromptTemplate' 和 'StrOutputParser'

from langchain_core.output_parsers import SimpleJsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # 示例大型语言模型提供商

# 示例：替换为您实际的大型语言模型初始化
# 确保在您的环境中设置了 OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义提示，请求 JSON 输出
prompt_template = """
从以下职位描述中提取姓名和主要技能：
{description}

将结果作为 JSON 对象返回，包含 "name" 和 "skill" 两个键。
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# 创建解析器实例
json_parser = SimpleJsonOutputParser()

# 创建链
chain = prompt | llm | json_parser

# 运行链
job_description = "我们正在招聘一名高级 Python 开发人员，精通 Web 框架和云服务。"
result = chain.invoke({"description": job_description})

print(result)
# 预期输出（可能因大型语言模型而略有不同）：
# {'name': '高级 Python 开发人员', 'skill': 'Python'}
```

此解析器期望大型语言模型输出是一个包含有效 JSON 对象的字符串。如果大型语言模型未能生成有效的 JSON，它很可能会引发解析错误。

### PydanticOutputParser

对于更复杂的数据结构和额外的验证，`PydanticOutputParser` 是一个很好的选择。它与 Pydantic 集成，Pydantic 是一个流行的用于数据验证和设置管理的 Python 库。您使用 Pydantic 模型定义所需的输出结构，解析器既负责生成格式指令，又将大型语言模型输出解析为您模型的一个实例。

首先，使用 Pydantic 定义您的数据结构：

```python
# 需要 'pip install pydantic'
from pydantic import BaseModel, Field
from typing import List

class PersonInfo(BaseModel):
    name: str = Field(description="这个人的全名")
    age: int = Field(description="这个人的年龄")
    hobbies: List[str] = Field(description="这个人的爱好列表")
```

现在，将 `PydanticOutputParser` 与此模型一起使用：

```python
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
# 假设 'llm' 是一个已初始化的 LangChain 大型语言模型实例
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# 设置解析器并注入指令到提示模板中。
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# 获取用于引导大型语言模型的格式指令
format_instructions = parser.get_format_instructions()

# 定义包含格式指令的提示模板
prompt_template_str = """
从以下文本中提取一个人的信息：
{text_input}

{format_instructions}
"""

prompt = PromptTemplate(
    template=prompt_template_str,
    input_variables=["text_input"],
    partial_variables={"format_instructions": format_instructions}
)

# 创建链
chain = prompt | llm | parser

# 输入文本
text = "Alice is 30 years old and enjoys painting, hiking, and coding."

# 运行链
person_object = chain.invoke({"text_input": text})

print(person_object)
# 预期输出：
# name='Alice' age=30 hobbies=['painting', 'hiking', 'coding']

print(f"姓名：{person_object.name}")
print(f"年龄：{person_object.age}")
print(f"爱好：{person_object.hobbies}")
# 姓名：Alice
# 年龄：30
# 爱好：['painting', 'hiking', 'coding']
```

`get_format_instructions()` 方法会生成描述所需 JSON 模式（基于 Pydantic 模型）的文本，这有助于大型语言模型正确格式化其输出。使用 Pydantic 模型提供自动验证；如果大型语言模型输出不符合 `PersonInfo` 模式（例如，为年龄提供文本而不是数字），Pydantic 将引发验证错误。

### CommaSeparatedListOutputParser

当您只需要一个项目列表时，`CommaSeparatedListOutputParser` 非常直接。它引导大型语言模型返回一个逗号分隔的列表，然后将该字符串解析为 Python 列表。

```python
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
# 假设 'llm' 是一个已初始化的 LangChain 大型语言模型实例
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# 创建解析器
list_parser = CommaSeparatedListOutputParser()

# 获取格式指令
format_instructions = list_parser.get_format_instructions()

# 定义提示
prompt_template = """
列出 5 个流行的 Python Web 框架。
{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# 创建链
chain = prompt | llm | list_parser

# 运行链
result = chain.invoke({}) # 此提示不需要特定输入

print(result)
# 预期输出（列表顺序和具体框架可能有所不同）：
# ['Django', 'Flask', 'FastAPI', 'Pyramid', 'Bottle']
```

### StructuredOutputParser

`StructuredOutputParser` 提供一种更通用的方式来定义多个输出字段，而无需 Pydantic。您定义所需的字段及其描述。与 Pydantic 解析器一样，它会生成格式指令。

```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
# 假设 'llm' 是一个已初始化的 LangChain 大型语言模型实例
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义所需的输出模式
response_schemas = [
    ResponseSchema(name="answer", description="用户问题的答案。"),
    ResponseSchema(name="source", description="用于找到答案的来源，如果可能应为网站 URL。")
]

# 创建解析器
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# 获取格式指令
format_instructions = output_parser.get_format_instructions()

# 定义提示模板
prompt_template_str = """
尽可能准确地回答用户的问题。
{format_instructions}
问题：{question}
"""

prompt = PromptTemplate(
    template=prompt_template_str,
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

# 创建链
chain = prompt | llm | output_parser

# 运行链
question = "马来西亚的首都是哪里？"
result = chain.invoke({"question": question})

print(result)
# 预期输出（来源可能因大型语言模型而异或由其估计）：
# {'answer': '马来西亚的首都是吉隆坡。', 'source': '一般知识 / 维基百科'}
```

### 在链中整合解析器

如示例所示，输出解析器通常是 LangChain 链中的最后一步。基本结构是：

`提示 -> 大型语言模型 -> 输出解析器`

提示格式化输入并包含来自解析器的任何必要的格式指令。大型语言模型生成文本响应。然后，输出解析器接收此文本并将其转换为所需的 Python 结构。

### 处理解析错误

大型语言模型并非总是完美遵循指令。有时，它们的输出可能与解析器期望的格式不匹配（例如，JSON 中缺少引号，数据类型不正确）。当发生这种情况时，输出解析器的 `parse()` 方法通常会引发异常。

在生产应用程序中，您需要错误处理。这可能涉及：

- **重试：** 实现逻辑来捕获解析错误，可能会稍微修改提示（例如，提醒大型语言模型格式），然后再次尝试大型语言模型调用。LangChain 包含 `OutputFixingParser` 等机制，它尝试通过将错误反馈给大型语言模型来自动修复格式不正确的输出。
- **日志记录：** 记录解析失败的实例，以便了解常见的失败模式。
- **默认值/回退：** 如果解析反复失败，提供一个默认的结构化输出。

选择正确的输出解析器取决于您需要提取的数据的复杂性以及您是否需要验证。它们是使大型语言模型输出在下游应用逻辑中可用的必要工具，将自由格式文本转换为可操作的结构化数据。

## 参考资料

- [LangChain Expression Language (LCEL) and Output Parsers](https://python.langchain.com/docs/modules/model_io/output_parsers/) — LangChain Community (2024)
  LangChain输出解析器的官方文档，提供详细解释、示例和在LCEL中的使用模式。
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Pydantic Community (2024)
  Pydantic的权威指南，对于定义稳健的数据模式和验证至关重要，这些模式和验证被LangChain的PydanticOutputParser所使用。
