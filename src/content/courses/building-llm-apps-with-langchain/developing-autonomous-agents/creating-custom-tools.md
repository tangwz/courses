---
course: "building-llm-apps-with-langchain"
chapter: "developing-autonomous-agents"
lesson: "creating-custom-tools"
sourceId: 7438
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents/creating-custom-tools"
title: "创建自定义工具"
description: "学习如何定义您自己的自定义工具，赋予代理新功能，例如与特定API进行交互。"
order: 4
plots: []
sourceHash: "79a26e94b29a8721e27fce3bdf77fc232aecda218a92d07e049113c74c966c22"
sourceCorrections: []
---

虽然标准工具为许多任务提供了坚实基础，但当代理能够与您的特定环境进行交互时，它们的真正作用才会显现出来。许多应用程序要求代理连接到专有数据库、内部服务或专用第三方API。创建自定义工具使这成为可能。

自定义工具本质上是一个Python函数，附带一个LLM能够理解的清晰说明。代理的效力几乎完全由您如何定义这些工具来决定。

### 工具的结构

LangChain工具的核心由三个主要组成部分构成：

- **name** (str)：工具的唯一标识符。LLM使用此名称来指定它想要使用的工具。
- **description** (str)：对工具功能的详细说明。这是最重要的组成部分。LLM的推理 (inference)引擎完全依赖于此说明来判断何时该工具合适以及它预期什么输入。一个精心编写的说明决定了代理的成败。
- **func** (callable)：当代理调用该工具时执行的实际Python函数。

您可以使用 `langchain.tools` 中的 `Tool` 类直接构造一个工具。让我们创建一个简单的工具，根据给定半径计算圆的周长。

```python
import math
from langchain.tools import Tool

def calculate_circumference(radius: str) -> str:
    """根据给定半径计算圆的周长。"""
    try:
        # 代理以字符串形式提供输入，因此我们必须进行类型转换。
        radius_float = float(radius)
        circumference = 2 * math.pi * radius_float
        return f"周长为 {circumference:.2f}。"
    except ValueError:
        return "输入无效。请为半径提供一个有效数字。"

circumference_tool = Tool(
    name="CircleCircumferenceCalculator",
    description="使用此工具计算圆的周长。输入应为圆的半径，一个数字。",
    func=calculate_circumference,
)

print(circumference_tool.invoke("10"))
```

在此示例中，`description` 清晰地告知LLM工具的用途及其预期输入格式。函数本身包含基本错误处理，这是您应始终遵循的做法。

### 使用 @tool 装饰器简化操作

编写 `Tool` 类定义可能会变得重复。LangChain提供了一种更符合Python风格且方便的方式，使用 `@tool` 装饰器来创建工具。此装饰器自动根据函数名推断工具的 `name`，并且重要的是，使用函数的文档字符串作为其 `description`。

这种方法不仅减少了样板代码，还鼓励良好的文档编写习惯。让我们重构我们的周长计算器。

```python
from langchain.tools import tool

@tool
def circle_circumference_calculator(radius: str) -> str:
    """
    计算圆的周长。
    当你需要根据给定半径求圆的周长时，请使用此工具。
    此工具的输入应为表示半径数值的字符串。
    """
    try:
        radius_float = float(radius)
        circumference = 2 * math.pi * radius_float
        return f"周长为 {circumference:.2f}。"
    except ValueError:
        return "输入无效。请为半径提供一个有效数字。"

# 现在该工具是一个可调用对象
print(circle_circumference_calculator.name)
print(circle_circumference_calculator.description)
print(circle_circumference_calculator.invoke("10"))
```

输出显示装饰器正确地将函数名和文档字符串分配给了工具的属性。对于大多数自定义工具，装饰器是推荐的方法。

下图展示了代理如何使用工具的描述来决定采取何种行动。

> 代理的决策循环。LLM查看可用工具的描述，根据用户的查询及其内部推理 (inference)过程选择合适的行动。

### 定义结构化工具输入

默认情况下，使用 `@tool` 装饰器或 `Tool` 类创建的工具接受单个字符串参数 (parameter)。然而，许多函数和API调用需要多个结构化参数。您可以使用Pydantic `BaseModel` 为工具定义结构化输入模式。

这为LLM提供了它需要生成的参数的清晰格式，从而提高了可靠性。让我们为“用户资料API”创建一个工具，该API需要 `user_id` 和一个可选的 `include_history` 标志。

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class UserProfileInput(BaseModel):
    """用户资料工具的输入模型。"""
    user_id: int = Field(description="用户的唯一标识符。")
    include_history: bool = Field(default=False, description="是否包含用户的订单历史。")

@tool(args_schema=UserProfileInput)
def get_user_profile(user_id: int, include_history: bool = False) -> str:
    """
    获取用户的个人资料信息。
    用它来获取特定用户ID的详细资料。
    您可以选择包含他们的订单历史。
    """
    # 在实际应用中，这将查询数据库或API。
    profile = f"用户 {user_id} 的资料: 姓名 - Alex, 自2022年起成为会员。"
    if include_history:
        history = " 订单历史: [订单 #123, 订单 #456]。"
        return profile + history
    return profile

# 代理现在知道如何构造输入
print(get_user_profile.args)
```

通过提供 `args_schema`，您准确地指导代理如何格式化其请求到工具。当LLM决定使用此工具时，它将尝试生成一个匹配 `UserProfileInput` 模式的JSON对象。

### 一个实用示例：天气API工具

让我们构建一个更实用的工具，从外部API获取真实数据。我们将创建一个工具，使用OpenWeatherMap API获取给定城市的当前天气。

首先，确保您拥有OpenWeatherMap的API密钥，并且 `requests` 库已安装（`pip install requests`）。

```python
import os
import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

# 将您的API密钥设置为环境变量
# os.environ["OPENWEATHERMAP_API_KEY"] = "your_api_key_here"

class WeatherInput(BaseModel):
    """GetCurrentWeather 工具的输入。"""
    city: str = Field(description="要获取天气的城市名称。")

@tool(args_schema=WeatherInput)
def get_current_weather(city: str) -> str:
    """
    获取指定城市的当前天气。
    使用此工具查询任何城市的温度和天气状况。
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    if not api_key:
        return "错误：OPENWEATHERMAP_API_KEY 环境变量未设置。"

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 对错误的响应（4xx或5xx）抛出HTTPError
        data = response.json()

        temperature = data["main"]["temp"]
        description = data["weather"][0]["description"]

        return f"当前 {city} 的天气是 {temperature}°C，{description}。"
    except requests.exceptions.HTTPError as http_err:
        return f"发生HTTP错误：城市未找到或API错误。"
    except Exception as e:
        return f"发生错误：{e}"

# 直接运行工具的示例
print(get_current_weather.invoke({"city": "London"}))
```

该工具可靠：它具有结构化输入、清晰的描述、调用外部服务，并处理潜在错误，例如缺少API密钥或无效城市名。这就是您应该为代理构建的工具的质量。当集成到代理中时，它现在可以通过推理 (inference)出需要使用 `get_current_weather` 工具并传入参数 (parameter) `city="Tokyo"` 来回答“东京天气如何？”之类的问题。

## 参考资料

- [Tools](https://python.langchain.com/docs/how_to/tool_calling/) — LangChain (2024)
  Publisher: LangChain
  LangChain框架中创建和使用自定义工具的官方文档，包括结构化输入方案和`@tool`装饰器。
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Samuel Colvin and Pydantic Contributors (2024)
  Pydantic的指南，用于为LangChain工具定义结构化输入方案（`args_schema`），确保类型验证和LLM的清晰度。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  关于GPT等大型语言模型如何设计用于调用带有结构化参数的外部函数的介绍，提供了LangChain工具集成的底层机制。
