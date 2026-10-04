---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-testing-monitoring-maintenance"
lesson: "testing-agent-tools"
sourceId: 6615
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-6-llm-tool-testing-monitoring-maintenance/testing-agent-tools"
title: "代理工具的单元测试与集成测试"
description: "为LLM代理使用的工具制定专门的单元和集成测试策略。"
order: 1
plots: []
sourceHash: "12c8018c2cd8a5cd10a4bb4f7cdd727706deb2acc49a3ae96faa7a238523f362"
sourceCorrections: []
---

随着您为LLM代理构建工具库，确保每个工具可靠并按预期工作变得必不可少。有缺陷的工具可能导致代理行为不可预测、输出错误以及用户体验不佳。建立测试实践，特别是单元测试和集成测试，专为LLM代理工具的独特环境而设计。通过彻底测试您的工具，您为构建可靠高效的代理系统打下根基。

### 理解代理工具的测试

测试为LLM代理设计的工具，与传统软件组件相比，带来了一套略有不同的挑战。虽然单元测试和集成测试的基本原则保持不变，但与LLM的互动增加了需要考虑的层面：

- **LLM作为调用方：** 工具最终由LLM（或代表LLM行事的代理框架）调用。LLM严重依赖于工具的描述、输入模式和输出格式。测试必须确保这些方面清晰、正确，并导致工具的可预测执行。
- **结构化数据交换：** LLM通常期望结构化数据（如JSON）作为工具的输入，并生成工具可能使用或代理框架需要从工具输出中解析的结构化数据。测试这些数据契约非常重要。
- **将工具逻辑与LLM行为分离：** 区分测试工具本身与测试LLM正确*选择*或*使用*工具的能力非常重要。本节主要关注前者；即确保工具在用特定输入调用时按设计行为。

代理工具的单元测试和集成测试的主要目标是验证每个工具在隔离状态下能正确运行，并能与代理的操作环境集成。

### 单元测试：验证单个工具组件

单元测试侧重于工具中最小的可测试部分，通常是类中的单个函数或方法。目标是验证工具逻辑的每个部分都能正确工作，且独立于系统的其他部分或外部依赖。

**单元测试主要涵盖的方面：**

1. **主要功能：**

   - 验证工具使用有效输入准确执行其主要任务。例如，`计算抵押贷款支付`工具应为给定贷款参数 (parameter)返回正确的支付金额。
   - 测试可能运行不同逻辑路径的各种有效输入组合。
2. **输入验证与处理：**

   - **数据类型：** 确保工具正确处理预期数据类型，并优雅地拒绝或标记 (token)不正确类型。
   - **必选与可选参数：** 测试必选参数缺失以及可选参数提供或省略的场景。
   - **格式错误的输入：** 工具如何处理类型正确但格式不正确的输入（例如，日期参数的无效日期字符串）？
   - **边界情况：** 测试边界条件。对于数值输入，这包括零、负数（如果适用）、非常大的数字或空字符串/列表。
   - **模式一致性：** 如果您的工具期望遵循特定JSON模式的输入，请对照此模式进行验证。
3. **输出格式与结构：**

   - 确认工具以LLM或代理框架期望的精确格式和结构返回数据。这通常是JSON，但也可能是特定的字符串模式或XML。
   - 如果输出包含多个字段，请确保所有字段都存在并正确填充。
4. **错误处理与报告：**

   - 当工具遇到无法解决的问题（例如，外部服务已停止运行，无效输入阻止计算）时，它是否会抛出适当的异常或返回结构化的错误消息？
   - 返回给LLM的错误消息应足够有信息量，以便LLM能够潜在地理解问题或告知用户。例如，与通用的“错误”相比，像“无法找到指定城市的天气数据”这样的消息更实用。
5. **状态管理（对于有状态工具）：**

   - 如果您的工具在调用之间维护内部状态（例如，累积数据的工具），单元测试应验证状态初始化、转换和重置机制。
6. **模拟外部依赖：**

   - 工具经常与外部系统（如数据库或第三方API）交互。对于单元测试，这些外部依赖项应被“模拟”。模拟涉及用一个可控的替代品来替换真实的外部服务，该替代品模拟其行为。这使得测试更快、更可靠（不依赖于网络或外部服务的正常运行时间），并具有确定性。
   - Python的`unittest.mock`库（与`MagicMock`、`patch`一起）通常用于此目的。

**示例：单元测试一个简单的Python工具**

考虑一个Python工具，它获取股票价格：

```python
# stock_tool.py
import requests

class StockPriceTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.examplefinance.com/v1/stock"

    def get_price(self, ticker_symbol: str) -> dict:
        if not isinstance(ticker_symbol, str) or not ticker_symbol.isalpha():
            return {"error": "无效的股票代码格式。"}

        try:
            response = requests.get(
                f"{self.base_url}/{ticker_symbol}/price",
                params={"apikey": self.api_key}
            )
            response.raise_for_status() # 对于不良响应（4XX或5XX）抛出HTTPError
            data = response.json()
            if "price" not in data:
                return {"error": f"未找到 {ticker_symbol} 的价格数据。"}
            return {"ticker": ticker_symbol, "price": data["price"]}
        except requests.exceptions.RequestException as e:
            return {"error": f"API请求失败：{str(e)}"}
        except ValueError: # JSONDecodeError 继承自 ValueError
            return {"error": "无法解析API响应。"}
```

使用`pytest`和`unittest.mock`的单元测试可能如下所示：

```python
# test_stock_tool.py
import pytest
from unittest.mock import patch, MagicMock
from stock_tool import StockPriceTool

@pytest.fixture
def tool():
    return StockPriceTool(api_key="test_api_key")

def test_get_price_success(tool, monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"price": 150.75}
    mock_response.raise_for_status.return_value = None # 模拟没有HTTP错误

    # 使用 pytest 的 monkeypatch 来模拟 requests.get，类似于 unittest.mock.patch
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_response))

    result = tool.get_price("AAPL")
    assert result == {"ticker": "AAPL", "price": 150.75}
    requests.get.assert_called_once_with(
        "https://api.examplefinance.com/v1/stock/AAPL/price",
        params={"apikey": "test_api_key"}
    )

def test_get_price_invalid_ticker_format(tool):
    result = tool.get_price("AAPL123")
    assert result == {"error": "无效的股票代码格式。"}

def test_get_price_api_failure(tool, monkeypatch):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("API 不可用")

    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_response))

    result = tool.get_price("MSFT")
    assert "API请求失败" in result["error"]

def test_get_price_data_not_found(tool, monkeypatch):
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": "未找到股票代码"} # API未返回价格字段
    mock_response.raise_for_status.return_value = None

    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_response))

    result = tool.get_price("GOOG")
    assert result == {"error": "未找到 GOOG 的价格数据。"}
```

**单元测试的最佳实践：**

- **隔离：** 每个测试应验证一个特定方面或路径。
- **自动化：** 作为开发流程的一部分自动运行测试（例如，预提交钩子、CI/CD管道）。
- **可重复：** 测试在相同环境下每次运行时都应产生相同的结果。
- **快速：** 单元测试应快速执行以提供快速反馈。
- **描述性名称：** 测试函数名称应清楚地表明它们正在测试什么（例如，`test_tool_handles_invalid_input_gracefully`）。

> 单元测试验证单个代理工具的方法或函数在隔离状态下是否正常运行，通常使用模拟来处理API调用等外部依赖。

### 集成测试：确保工具在代理生态系统中正常工作

单元测试确认单个组件正确工作，而集成测试则验证您的工具是否与更大的代理框架或编排器正确交互。它们侧重于将您的工具连接到最终将根据LLM决策调用它的系统的“连接机制”。

**集成测试应涵盖的方面：**

1. **工具注册与发现：**

   - 代理框架能否正确加载、解析定义并识别您的工具？
   - 工具的名称、描述以及输入/输出模式是否按预期注册？
2. **框架对工具的调用：**

   - 当代理框架（或模拟LLM决策的模拟代理）决定使用您的工具时，它是否被正确调用？
   - 从框架传递给工具的参数 (parameter)是否按预期传递，包括类型转换（如果有）？
3. **参数映射与转换：**

   - LLM生成的参数结构可能与工具函数直接期望的略有不同。代理框架或包装层可能负责转换这些参数。集成测试可以验证这种映射。
4. **框架对响应的处理：**

   - 代理框架能否正确接收和解释工具的输出，无论是成功结果还是结构化错误？
   - 框架是否适当处理不同类型的工具响应（例如，JSON、字符串、错误对象）？
5. **与真实外部服务的交互（预生产/测试环境）：**

   - 与模拟外部调用的单元测试不同，集成测试可能涉及调用真实的但非生产的外部服务实例（例如，预生产API端点、测试数据库）。
   - 这有助于验证：
     - 认证和授权机制。
     - 实际API请求/响应格式。
     - 对速率限制或网络问题的基本处理（尽管广泛的压力测试通常是独立的）。
6. **模拟LLM交互：**

   - 您可以创建“模拟代理”场景。例如，给定一个模拟的用户查询，该查询*应该*导致您的工具被特定参数调用，集成测试是否确认此流程？这并非测试LLM的智能，而是从一个看似合理的类LLM指令到工具执行的路径。

**集成测试的方法：**

- **使用模拟代理/编排器进行测试：** 创建一个简化的测试框架，模仿代理在选择和调用工具中的角色。该框架将以预定义参数确定性地调用您的工具并检查结果。这为工具与框架交互提供了良好的隔离。
- **使用实际代理框架进行测试：** 使用您正在使用的实际代理框架（例如LangChain、LlamaIndex或自定义系统）。您将使用您的工具配置框架，然后触发应该导致其调用的场景。这更全面但可能需要更多设置。

**示例：集成代码片段**

假设您有一个代理框架，其中包含一个方法`run_agent_query(query: str)`。集成测试可能如下所示：

```python
# test_agent_integration.py
from my_agent_framework import AgentFramework
from stock_tool import StockPriceTool # 假设它已注册

def test_stock_tool_integration_with_agent(monkeypatch):
    # 模拟 StockPriceTool 的外部API调用，即使在集成测试中，
    # 如果您只想关注框架与工具的交互并避免外部不稳定因素。
    # 或者，让它命中测试/预生产API端点。
    mock_api_response = MagicMock()
    mock_api_response.json.return_value = {"price": 200.00}
    mock_api_response.raise_for_status.return_value = None
    monkeypatch.setattr("requests.get", MagicMock(return_value=mock_api_response))

    agent = AgentFramework()
    # 假设 StockPriceTool 以 "getStockPrice" 的名称注册到代理
    # 并且代理框架可以解析查询以调用此工具。
    # 这部分高度依赖于您具体的代理框架。

    # 此查询旨在确定性地触发股票工具
    # （通过特定关键词或测试设置中的简化NLU）
    agent_response = agent.run_agent_query("TSLA 股票的价格是多少？") 

    # 断言将检查工具是否被调用及其结果是否被处理
    # 如果直接输出不足以判断，这通常涉及检查日志或代理/框架的内部状态。
    # 为简单起见，假设 agent_response 包含结构化结果：
    assert "TSLA" in agent_response["tool_result"]["ticker"]
    assert agent_response["tool_result"]["price"] == 200.00
    # 潜在地断言 requests.get 被调用时使用了 "TSLA"
    requests.get.assert_called_with(
        "https://api.examplefinance.com/v1/stock/TSLA/price", # 或者您实际的股票代码
        params={"apikey": "test_api_key"} # 或者API密钥是如何管理的
    )
```

这个示例是简化的。与代理框架进行的实际集成测试通常需要更多设置来配置代理、其工具，并可能模拟LLM的决策过程，以可靠地触发特定工具。

> 集成测试检查工具如何与代理框架或模拟代理环境交互。这可能涉及真实的外部服务测试实例或进一步模拟，具体取决于测试的侧重点。

### 区分工具测试与LLM评估

需重申，此处讨论的单元测试和集成测试主要关注*工具本身的正确性和可靠性及其与代理连接机制的集成*。它们确保，如果LLM（或代理框架）决定用特定参数 (parameter)调用工具，该工具的行为可预测并返回格式良好的响应。

这些测试通常*不*评估：

- LLM正确理解用户查询的能力。
- LLM为任务选择*正确*工具的智慧。
- LLM为所选工具生成*正确*参数的技能。

评估这些LLM特有方面属于LLM评估和代理性能评估的更广泛范畴，这些是独立但相关的学科。经过充分测试的工具是进行有意义的LLM代理评估的先决条件。如果工具本身不可靠，就无法确定故障是由于工具还是LLM的推理 (inference)。

通过实施全面的单元测试和集成测试，您奠定了可靠工具的根基。这不仅提高了LLM代理的可靠性，还简化了调试和维护，因为您可以更有信心单个工具组件按预期运行。这种严格的测试方法是先进LLM代理系统开发中良好工程实践的标志。

## 参考资料

- The Art of Software Testing — Glenford J. Myers, Corey Sandler, Tom Badgett (2011)
  Publisher: John Wiley & Sons
  一部经典教材，涵盖软件测试的基本原理，包括单元测试和集成测试方法。
- [pytest Documentation](https://docs.pytest.org/en/stable/) — pytest-dev team (2024)
  pytest 的官方文档，pytest 是一个广泛使用的 Python 测试框架，详细介绍了其用于编写健壮的单元测试和集成测试的功能，包括 fixture 和 monkeypatching。
- [\`unittest.mock\` - mock object library](https://docs.python.org/3/library/unittest.mock.html) — The Python Development Team (2024)
  Python 标准库 `unittest.mock` 的官方文档，提供了在单元测试期间模拟和修补对象的工具，以隔离组件并管理依赖关系。
