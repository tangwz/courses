---
course: "python-llm-workflows"
chapter: "testing-evaluating-llm-apps"
lesson: "unit-testing-llm-components"
sourceId: 5211
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/unit-testing-llm-components"
title: "单元测试组件"
description: "编写提示模板和解析器等各个部分的单元测试方法。"
order: 2
plots: []
sourceHash: "0a02f02031ae90287bb39c1040046ad41ecaea85027255afb942b4c489ff2d7b"
sourceCorrections: []
---

虽然LLM应用程序的端到端行为可能不可预测，但工作流程中的许多独立组件是确定性的，非常适合传统单元测试。对这些构成要素应用单元测试具有明显好处：它有助于隔离错误，在开发期间提供快速反馈，并确保系统中可预测的部分在与LLM交互之前正常运行。这种方法建立了可靠性，即使最终LLM输出不同。

可以把您的LLM工作流程看作一个管道。单元测试侧重于独立验证该管道内的每个不同阶段或工具。

### 测试提示模板

提示模板负责构建发送给LLM的最终提示。提示格式错误可能导致糟糕或不正确的回复。单元测试可以验证您的模板是否正确包含变量并生成预期结构。

以LangChain的`PromptTemplate`为例：

```python
from langchain.prompts import PromptTemplate

# 示例模板
template_string = "Summarize the following text about {topic}: {text}"
prompt_template = PromptTemplate(
    input_variables=["topic", "text"],
    template=template_string
)

# 使用 pytest 进行单元测试
def test_prompt_template_formatting():
    topic = "Renewable Energy"
    text_input = "Solar power is becoming increasingly popular..."
    expected_output = "Summarize the following text about Renewable Energy: Solar power is becoming increasingly popular..."

    formatted_prompt = prompt_template.format(topic=topic, text=text_input)

    assert formatted_prompt == expected_output

def test_prompt_template_missing_variable():
    # 使用 pytest 的 raises 上下文管理器示例
    import pytest
    from langchain_core.exceptions import KeyErrorOutputParser

    with pytest.raises(KeyError): # LangChain 模板在缺少变量时会引发 KeyError
         prompt_template.format(topic="Climate Change") # 缺少 'text' 变量
```

这些测试确认，模板引擎对于有效输入能按预期运行，并能适当处理缺少变量等错误，而无需调用实际的LLM。

### 测试输出解析器

输出解析器将LLM的原始文本输出转换为更结构化的格式（例如JSON、列表或自定义对象）。它们的逻辑可能复杂，涉及正则表达式或特定字符串操作。单元测试很重要，以确保它们能正确解析预期输出，并优雅地处理潜在的变体或格式错误的响应。

假设您有一个自定义解析器，或正在使用LangChain的`OutputParser`，例如`SimpleJsonOutputParser`：

````python
from langchain.output_parsers import SimpleJsonOutputParser
import json
import pytest

# 假设 SimpleJsonOutputParser 旨在提取 JSON 块
parser = SimpleJsonOutputParser()

def test_json_parser_valid_output():
    # 模拟包含 JSON 的 LLM 输出
    llm_output = 'Some introductory text.\n```json\n{"name": "Alice", "age": 30}\n```\nSome concluding text.'
    expected_parsed_output = {"name": "Alice", "age": 30}

    parsed_output = parser.parse(llm_output)

    assert parsed_output == expected_parsed_output

def test_json_parser_malformed_json():
    # 模拟包含无效 JSON 的 LLM 输出
    llm_output = 'Here is the data:\n```json\n{"name": "Bob", "age": 40,\n```\n' # 格式错误的 JSON

    # 检查解析器是否引发了适当的异常（例如 OutputParserException）
    from langchain_core.exceptions import OutputParserException
    with pytest.raises(OutputParserException):
        parser.parse(llm_output)

def test_json_parser_no_json():
     llm_output = "There seems to be no JSON data here."
     from langchain_core.exceptions import OutputParserException
     with pytest.raises(OutputParserException): # 如果未找到 JSON，则预期会报错
         parser.parse(llm_output)
````

这些测试使用示例字符串输入，模仿潜在的LLM响应，以验证解析器的逻辑，而无需任何LLM交互。您可以为各种边界情况创建测试，包括不完整的JSON、格式不同的代码块，或完全缺少预期结构的输出。

### 测试数据加载器和格式器

如果您的应用程序涉及检索增强生成（RAG），那么在索引前加载、拆分或格式化数据的组件是进行单元测试的重要对象。

- **数据加载器：** 验证它们能否正确读取不同文件类型（PDF、TXT、HTML）。测试空文件或编码异常的文件等边界情况。
- **文本拆分器：** 确保文本按照所需策略进行分块（例如，按字符数、令牌或递归方式）。测试短文本、长文本 (long context)和结构多变的文本。
- **元数据提取器：** 确认相关元数据（例如，源文件名、页码）已正确提取并与数据块关联。

```python
# 示例：测试文本拆分器
from langchain.text_splitter import CharacterTextSplitter # 示例拆分器

def test_character_splitter_basic():
    splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    text = "This is a sample text for testing the splitter functionality."
    chunks = splitter.split_text(text)

    assert len(chunks) > 1 # 预期文本会被拆分
    assert chunks[0] == "This is a sample text" # 检查第一个块的内容（近似）
    # 根据预期的块大小和重叠添加更多断言

def test_character_splitter_small_text():
    splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    text = "Short text."
    chunks = splitter.split_text(text)

    assert len(chunks) == 1 # 如果文本小于块大小，则不应拆分
    assert chunks[0] == text
```

### 测试实用函数

您为数据清理、输入验证或链/代理中特定逻辑等任务编写的任何辅助函数都应有专门的单元测试。这些通常是标准Python函数，其正确性可以轻松验证。

```python
# 示例：测试一个简单的验证函数
def is_valid_email(email: str) -> bool:
    # （用于演示的简化检查）
    return "@" in email and "." in email.split('@')[-1]

def test_email_validation():
    assert is_valid_email("test@example.com") == True
    assert is_valid_email("test.example.com") == False
    assert is_valid_email("test@domain") == False
    assert is_valid_email("") == False
```

通过全面对这些各个组件进行单元测试，您可以确保LLM应用程序的确定性部分稳固。这使得调试更简单，因为您可以更有信心认为工作流程中后期遇到的错误可能与LLM交互或组件间的集成有关，而不是构成要素中的根本缺陷。这些测试是LLM应用程序全面测试策略的重要组成部分。

## 参考资料

- [pytest documentation](https://docs.pytest.org/) — pytest-dev team (2024)
  pytest框架的官方文档，提供了在Python中编写和运行单元测试的完整指南，包括功能和实践。
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) — LangChain Team (2024)
  LangChain框架的官方文档，对于理解LLM应用开发中使用的PromptTemplates、输出解析器和文本分割器等组件很有价值。
- [Test-Driven Development: By Example](https://www.oreilly.com/library/view/test-driven-development/0321146530/) — Kent Beck (2002)
  Publisher: Addison-Wesley Professional
  一本介绍测试驱动开发（TDD）原则和实践的书籍，侧重于在编写代码之前先写测试，以增强设计、减少错误并为确定性组件提供快速反馈。
