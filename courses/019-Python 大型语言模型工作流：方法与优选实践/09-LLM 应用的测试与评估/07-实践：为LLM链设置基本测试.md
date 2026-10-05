# 实践：为LLM链设置基本测试

来源：[原文](https://apxml.com/zh/courses/python-llm-workflows/chapter-9-testing-evaluating-llm-apps/practice-basic-tests-llm-chain)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为LangChain链设置基本测试，涉及验证其结构完整性和预期流程。这种方法不同于评估LLM输出的质量，后者通常使用专门的评估方法。我们将使用流行的Python测试框架 `pytest` 和模拟技术，以便在测试期间将链与实际的LLM API调用隔离。

### 示例场景：一个简单的摘要链

设想我们有一个简单的LangChain链，设计用于接收一段文本并将其总结为三个要点。它可能包含：

1. 一个 `PromptTemplate` 来指导LLM。
2. 一个LLM模型实例（例如，来自OpenAI）。
3. 一个 `OutputParser` （也许是 `SimpleJsonOutputParser` 或自定义的）来构建要点。

以下是我们链的结构，使用LangChain表达式语言（LCEL）：

```python
# 假设已导入必要的模块：ChatOpenAI, PromptTemplate, StrOutputParser, JsonOutputParser 等。
# 假设 OPENAI_API_KEY 已在环境变量中设置

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# 简化示例 - 实际链可能使用 JsonOutputParser 以获得更好的结构
prompt_template = ChatPromptTemplate.from_template(
    "Summarize the following text into exactly three concise bullet points:\n\n{text}\n\nFormat the output as a numbered list."
)
model = ChatOpenAI(model="gpt-3.5-turbo") # 或您偏好的模型
output_parser = StrOutputParser() # 本示例的简单解析器

# 定义链
summary_chain = (
    {"text": itemgetter("text")}
    | prompt_template
    | model
    | output_parser
)

# 示例用法（不属于测试本身）
# input_text = "Large Language Models are transforming industries by enabling natural language interaction..."
# result = summary_chain.invoke({"text": input_text})
# print(result)
```

### 设置测试环境

首先，请确保您已安装 `pytest` 和 `pytest-mock`：

```bash
pip install pytest pytest-mock langchain langchain_openai python-dotenv
```

我们将把测试组织在一个单独的文件中，例如 `test_summary_chain.py`。我们还需要一种安全管理API密钥的方法；使用环境变量（以及可能通过 `python-dotenv` 加载的 `.env` 文件用于本地测试）是常见做法。

### 模拟LLM交互

在测试中直接调用LLM API会使测试变慢、成本高昂，并可能导致非确定性结果。对*链结构*进行单元和集成测试的主要思想是模拟LLM调用。我们希望验证我们的提示是否正确构建，并且链能适当地处理LLM的*预期*响应格式。

Python的内置 `unittest.mock` 库（或 `pytest-mock` 提供的 `mocker` fixture）非常适合此目的。

### 编写测试

我们来创建 `test_summary_chain.py`：

```python
import pytest
from unittest.mock import MagicMock # 用于创建模拟对象
from operator import itemgetter

# 假设您的链定义在名为 `summary_chain_module.py` 的文件中
from summary_chain_module import summary_chain, prompt_template, model, output_parser

# 测试提示模板格式化
def test_prompt_template_formatting():
    """验证提示模板是否正确插入文本。"""
    sample_text = "This is sample input text."
    expected_prompt_value = "Summarize the following text into exactly three concise bullet points:\n\nThis is sample input text.\n\nFormat the output as a numbered list."

    # 创建一个等效的 RunnablePassthrough 用于测试模板部分
    prompt_part = {"text": itemgetter("text")} | prompt_template

    result = prompt_part.invoke({"text": sample_text})

    # ChatPromptTemplate 的结果具有 'to_string()' 方法
    assert result.to_string() == expected_prompt_value

# 使用模拟LLM测试整个链
def test_summary_chain_with_mock_llm(mocker): # 使用 pytest-mock 的 'mocker' fixture
    """验证链是否正确处理模拟的LLM响应。"""
    sample_text = "This is the text to be summarized."
    mock_llm_output = "1. First point.\n2. Second point.\n3. Third point."

    # 模拟链中 'model' 实例的 'invoke' 方法
    # 我们找到 'model' 在哪里使用（在 summary_chain_module 中）并在那里进行修补。
    mock_model_invoke = mocker.patch('summary_chain_module.model.invoke') 
    mock_model_invoke.return_value = MagicMock(content=mock_llm_output) # 如果需要，模拟 AIMessage 结构，这里 StrOutputParser 直接期望字符串或带有 .content 的 AIMessage

    # 调用实际的链
    result = summary_chain.invoke({"text": sample_text})

    # 断言提示已正确传递给模拟的模型
    # 模拟函数第一次调用的第一个参数
    call_args = mock_model_invoke.call_args[0][0] 
    expected_prompt = prompt_template.invoke({"text": sample_text})
    assert call_args == expected_prompt 

    # 断言最终输出是经解析器处理的模拟LLM输出
    assert result == mock_llm_output 

# 测试输出解析器逻辑（如果它更复杂）
# 对于 StrOutputParser，没有太多可测试的，但如果使用 JsonOutputParser：
# def test_output_parser():
#     mock_llm_response_content = '{"summary": ["Point 1", "Point 2", "Point 3"]}'
#     # 假设 json_output_parser = JsonOutputParser(...) 
#     # parsed_output = json_output_parser.parse(mock_llm_response_content)
#     # assert parsed_output == {"summary": ["Point 1", "Point 2", "Point 3"]}
```

### 运行测试

导航到包含 `test_summary_chain.py` 的目录，并在终端中运行 `pytest`：

```bash
pytest
```

您应该会看到指示测试通过或失败的输出。

### 解读和后续步骤

这些测试成功验证了：

1. **提示构建：** `test_prompt_template_formatting` 确保我们的模板正确地整合了输入变量。
2. **链集成（模拟）：** `test_summary_chain_with_mock_llm` 确认输入通过提示模板流向（模拟的）模型，并且（模拟的）模型的输出由输出解析器正确处理。它检查链内的*连接*和*数据流*。

请记住，这些测试**不**评估 `mock_llm_output` 是否是*良好*的摘要。这需要前面讨论过的评估策略，例如使用评估数据集，与参考摘要进行比较（例如ROUGE分数），或采用基于LLM的评估方法。

这种做法构成了测试的一个基本层面，确保您的链在进行更复杂的质量评估之前结构是健全的。它在开发周期的早期就能捕获逻辑、解析和组件集成中的错误。

## 参考资料

- [pytest documentation](https://docs.pytest.org/en/stable/) — The pytest development team (2024)
  pytest框架的官方指南，对于理解Python测试、fixture和pytest-mock等插件至关重要。
- [unittest.mock - mock object library](https://docs.python.org/3/library/unittest.mock.html) — Python Software Foundation (2024)
  Python内置mocking库的官方文档，提供了创建mock对象以隔离测试与外部依赖的基本原理和用法。
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Team (2024)
  使用LangChain Expression Language (LCEL) 构建自定义链的官方指南，本节中使用LCEL定义了LLM链的结构。
- [Python Testing with pytest](https://pragprog.com/titles/bopytest2/python-testing-with-pytest-second-edition/) — Brian Okken (2022)
  Publisher: Pragmatic Bookshelf; Pages: 274
  一本实用指南，广泛介绍了如何使用pytest进行Python中的单元和集成测试，包括对mocking的详细解释。
- [Continuous Integration and Continuous Delivery for Machine Learning (CI/CD for ML): A Survey](https://ieeexplore.ieee.org/document/9501391) — Dmitry Sergeev, Mikhail Kovaltsev, Igor Kononenko (2021)
  Journal: IEEE Access; Publisher: IEEE; Volume: 9; Pages: 115201-115214; DOI: [10.1109/ACCESS.2021.3104938](https://doi.org/10.1109/ACCESS.2021.3104938)
  提供了机器学习中CI/CD实践的概述，其中包含测试在ML工作流中对健壮性和可靠性的作用。

---

[上一节](06-LLM%20%E4%BA%A4%E4%BA%92%E7%9A%84%E6%97%A5%E5%BF%97%E8%AE%B0%E5%BD%95%E4%B8%8E%E7%9B%91%E6%8E%A7.md) · [下一节](../10-%E9%83%A8%E7%BD%B2%E4%B8%8E%E8%BF%90%E7%BB%B4%E4%BC%98%E8%89%AF%E5%AE%9E%E8%B7%B5/01-%E6%89%93%E5%8C%85%E6%82%A8%E7%9A%84%20Python%20LLM%20%E5%BA%94%E7%94%A8%E7%A8%8B%E5%BA%8F.md)
