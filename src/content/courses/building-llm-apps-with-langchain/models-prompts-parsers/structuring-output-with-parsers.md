---
course: "building-llm-apps-with-langchain"
chapter: "models-prompts-parsers"
lesson: "structuring-output-with-parsers"
sourceId: 7415
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-2-models-prompts-parsers/structuring-output-with-parsers"
title: "使用解析器结构化输出"
description: "了解如何使用输出解析器将大型语言模型的原始文本输出转换为 JSON 或 Python 对象等结构化格式。"
order: 4
plots: []
sourceHash: "4aa7153d59635f68290627274c497e8cb1f1f9a6fb8c25673f9081768d2bd7cc"
sourceCorrections: []
---

大型语言模型（LLM）擅长生成人类可读的文本，但应用程序通常需要可预测的、结构化的数据格式。一个大型语言模型可能会返回一段优美描述用户的文字，但你的应用程序可能需要这些信息以字典或自定义对象的形式呈现，包含如 `name`、`age` 和 `location` 等明确的字段。这时，输出解析器就变得必不可少。它们是标准调用序列中的最后一部分，负责将模型的原始字符串输出转换为你的代码可以直接使用的结构化格式。

我们一直在构建的基本工作流程是 $提示 \rightarrow 模型 \rightarrow 解析器$。解析器在此序列中扮演双重角色。首先，它提供有关输出应如何格式化的说明，这些说明随后被插入到发送给模型的提示中。其次，在模型生成其响应后，解析器获取原始文本并将其转换为所需的 Python 数据结构。

> 从用户输入到结构化输出的数据流。输出解析器向提示提供格式说明，然后解析大型语言模型的字符串响应。

### 使用 PydanticOutputParser 处理自定义结构

对于应用程序而言，你通常需要将输出解析为具有特定类型的自定义数据类。`PydanticOutputParser` 是一个好用的工具，因为它与流行的 Python 数据验证库 Pydantic 集成。这允许你使用 Pydantic 模型定义所需的数据结构，解析器负责生成格式说明并将输出解析为该模型的一个实例。

假设我们想从故事中提取一个虚构人物的信息。我们可以使用 Pydantic 定义我们所需的结构：

```python
from typing import List

from pydantic import BaseModel, Field

class Character(BaseModel):
    name: str = Field(description="人物的姓名。")
    attributes: List[str] = Field(description="人物的标志性特征或特质列表。")
    role: str = Field(description="人物在故事中的角色（例如：主角、反派、配角）。")
```

现在，我们可以从这个模型创建一个 `PydanticOutputParser` 实例。解析器可以基于 Pydantic 模型的 schema 自动为大型语言模型生成详细的格式说明，包括字段名称、类型和描述。

```python
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 设置解析器
parser = PydanticOutputParser(pydantic_object=Character)

# 定义提示模板
template = """
从下面的段落中提取人物信息。
将响应格式化为 JSON 对象，包含 "name"、"attributes" 和 "role" 键。

{format_instructions}

段落：
{passage}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["passage"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 初始化模型并创建链
model = ChatOpenAI(temperature=0)
chain = prompt | model | parser

# 调用链
passage_text = "埃拉拉是一位勇敢的骑士，以她坚定不移的忠诚和高超的剑术闻名。她是这个王国的英雄。"
result = chain.invoke({"passage": passage_text})

print(result)
```

运行这段代码不仅会生成一个字典，还会生成一个实际的 `Character` 对象：

```
name='Elara' attributes=['brave', 'unwavering loyalty', 'skill with a blade'] role='protagonist'
```

请留意，解析器正确提取了信息并填充了 `Character` 类的一个实例。你的应用程序现在可以以类型安全的方式访问数据，例如 `result.name`（一个字符串）和 `result.attributes`（一个字符串列表）。这种与 Pydantic 的集成使得定义和执行输出结构的过程变得清晰和可靠。

### 其他常用输出解析器

LangChain 为不同的使用场景提供了多种预构建解析器。尽管 `PydanticOutputParser` 非常灵活，但有时一个更简单的解析器就足够了。

- **JsonOutputParser:** 一个通用解析器，将大型语言模型的输出字符串转换为 Python 字典，不进行 Pydantic 的类型验证。在你只需要一个简单的 JSON 对象，且不想提前定义 Pydantic 模型时很有用。
- **CommaSeparatedListOutputParser:** 顾名思义，此解析器旨在提取逗号分隔的项目列表。它指示大型语言模型返回一个列表，例如 `item1, item2, item3`，并将其解析为 Python 列表： `['item1', 'item2', 'item3']`。
- **DatetimeOutputParser:** 当你需要大型语言模型返回日期或时间时，使用此解析器。它会指示模型预期格式（例如 `YYYY-MM-DD HH:MM:SS`），并将输出字符串解析为 Python `datetime` 对象。

通过选择合适的解析器，你可以确保语言模型的输出能立即用于应用程序的逻辑。这一步对构建可预测、可靠的系统意义重大，使得应用从简单的文本生成转向创建能够处理并根据信息采取行动的应用程序。在即将到来的实践练习中，你将应用这些技术来构建自己的结构化数据提取器。

## 参考资料

- [Output Parsers](https://www.langchain.com/docs/modules/model_io/output_parsers/) — LangChain Contributors (2024)
  Publisher: LangChain
  LangChain 输出解析器的官方文档，详细介绍了它们的目的以及用于结构化 LLM 响应的各种实现。
- [Pydantic Documentation](https://docs.pydantic.dev/) — Samuel Colvin and Pydantic Contributors (2024)
  Pydantic 的官方文档，这是一个用于数据解析、验证和序列化的 Python 库，对 PydanticOutputParser 至关重要。
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Dair.ai, Shubham Saboo, and Contributors (2024)
  关于提示工程技术的综合指南，其中包括指导大型语言模型生成结构化格式响应的策略。
- [Tool use (Function calling)](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI 关于函数调用的官方指南，展示了大型语言模型生成结构化 JSON 输出以与外部工具和 API 交互的方法。
