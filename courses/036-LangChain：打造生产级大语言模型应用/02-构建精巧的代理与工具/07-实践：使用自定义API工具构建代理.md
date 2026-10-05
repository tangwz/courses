# 实践：使用自定义API工具构建代理

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-2-sophisticated-agents-tools/practice-agent-custom-tools)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建一个LangChain代理需要为其配备自定义工具，这些工具可以与外部数据源交互或执行特定的计算。这种方法模拟了常见的生产场景，在这些场景中，代理需要超出LLM自身知识的专业能力。

我们的目标是创建一个代理，它能够回答有关当前天气的问题并估算地点间的驾驶时间。这需要通过自定义工具，让代理能够使用两种不同的功能。

### 场景定义

设想您需要一个助手，能够回答以下问题：

- “东京目前的温度是多少？”
- “从旧金山开车到洛杉矶，实际需要多长时间？”
- “告诉我巴黎的天气预报，并估算从巴黎到里昂的驾驶时间。”

为处理这些请求，代理需要：

1. 一个工具，用于获取指定城市的当前天气信息。
2. 一个工具，用于估算两个城市之间的驾驶时间。

我们将把这些功能作为自定义LangChain工具实现，并将其整合到一个工具调用代理中。

### 步骤1：实现天气工具

首先，我们创建一个获取天气数据的工具。对于实际应用，您可能会使用OpenWeatherMap、WeatherAPI或类似的服务提供商。这通常需要获取一个API密钥。为简化本实践部分，我们将定义一个函数来返回模拟天气数据。不过，我们将以调用真实API的方式来构建它。

```python
# 前提条件：请确保已安装langchain、langchain-openai和langchain-core
# pip install langchain langchain-openai langchain-core python-dotenv

import os
import random
from dotenv import load_dotenv
from langchain_core.tools import BaseTool, Tool
from typing import Type, Optional
from pydantic import BaseModel, Field

# 加载环境变量（可选，用于真实API的API密钥）
load_dotenv()

# --- 天气工具实现 ---

class WeatherInput(BaseModel):
    """天气工具的输入模式。"""
    location: str = Field(description="需要获取天气的城市名称。")

def get_current_weather(location: str) -> str:
    """
    模拟获取某个地点的当前天气。
    在实际应用中，这将调用外部天气API。
    """
    print(f"---> Calling Weather Tool for: {location}")
    # 模拟API调用
    try:
        # 模拟数据生成
        temp_celsius = random.uniform(5.0, 35.0)
        conditions = random.choice(["晴朗", "多云", "有雨", "有风", "下雪（不太可能！）"])
        humidity = random.randint(30, 90)
        return f"The current weather in {location} is {temp_celsius:.1f}°C, {conditions}, with {humidity}% humidity."
    except Exception as e:
        return f"Error fetching weather for {location}: {e}"

# 选项1：使用Tool装饰器（适用于基本功能，更简单）
# from langchain_core.tools import tool
# @tool("weather_checker", args_schema=WeatherInput)
# def weather_tool(location: str) -> str:
#     """用于查找特定城市当前天气状况的工具。"""
#     return get_current_weather(location)

# 选项2：继承BaseTool类（控制力更强，更适合复杂逻辑/状态）
class WeatherTool(BaseTool):
    name: str = "天气查询器"
    description: str = "用于查找特定城市当前天气状况的工具。输入应为城市名称。"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str) -> str:
        """使用该工具。"""
        return get_current_weather(location)

    async def _arun(self, location: str) -> str:
        """异步使用该工具。"""
        # 对于这个简单的模拟函数，异步并非严格必要，
        # 但它演示了真实异步API调用的模式。
        # 在实际场景中，您会使用异步HTTP客户端（例如aiohttp）。
        return self._run(location) # 模拟异步调用

weather_tool = WeatherTool()

# 直接测试工具（可选）
# print(weather_tool.invoke({"location": "London"}))
# print(weather_tool.invoke("Paris")) # 如果args_schema允许，也接受直接字符串输入
```

关于此工具的要点：

- **`WeatherInput` 模式:** 我们定义了一个Pydantic模型`WeatherInput`来指定预期输入（`location`）。这有助于LangChain验证输入，并为LLM工具调用API提供结构。
- **`get_current_weather` 函数:** 这是核心逻辑。它目前使用随机数据，但模仿了API调用处理程序的结构，包括基本的错误处理。`print`语句有助于跟踪工具的执行。
- **`WeatherTool` 类:** 我们从`langchain_core`继承`BaseTool`以进行明确控制。
  - `name`：工具的简洁标识符。
  - `description`：对代理来说非常必要。LLM使用此描述来决定*何时*使用该工具以及*提供什么输入*。务必使其清晰且信息丰富。
  - `args_schema`：链接到我们的Pydantic输入模型。
  - `_run`：同步执行方法。
  - `_arun`：异步执行方法。

### 步骤2：实现驾驶时间工具

接下来，我们需要一个工具来估算驾驶时间。同样，实现中可能会使用Google Maps Distance Matrix或OSRM等API。我们将通过一个简单的计算来模拟它。

```python
# --- 驾驶时间工具实现 ---

class DrivingTimeInput(BaseModel):
    """驾驶时间工具的输入模式。"""
    origin: str = Field(description="起始城市或地点。")
    destination: str = Field(description="目的城市或地点。")

def estimate_driving_time(origin: str, destination: str) -> str:
    """
    模拟估算两个地点之间的驾驶时间。
    为简化起见，假设一个固定的平均速度。
    """
    print(f"---> Calling Driving Time Tool for: {origin} to {destination}")
    # 基于城市名称长度的极简距离模拟
    # （在生产环境中请替换为真实的距离计算或API调用）
    simulated_distance_km = abs(len(origin) - len(destination)) * 50 + random.randint(50, 500)
    average_speed_kph = 80

    if simulated_distance_km == 0: # 避免起点/终点相同时除以零
        return f"Origin and destination ({origin}) are the same."

    time_hours = simulated_distance_km / average_speed_kph
    hours = int(time_hours)
    minutes = int((time_hours - hours) * 60)

    return f"The estimated driving time from {origin} to {destination} is approximately {hours} hours and {minutes} minutes ({simulated_distance_km} km)."

class DrivingTimeTool(BaseTool):
    name: str = "驾驶时间估算器"
    description: str = ("用于估算两个城市之间驾驶时间的工具。"
                      "输入应为起始城市和目的城市。")
    args_schema: Type[BaseModel] = DrivingTimeInput

    def _run(self, origin: str, destination: str) -> str:
        """使用该工具。"""
        return estimate_driving_time(origin, destination)

    async def _arun(self, origin: str, destination: str) -> str:
        """异步使用该工具。"""
        # 模拟异步调用以作演示
        return self._run(origin, destination)

driving_tool = DrivingTimeTool()

# 直接测试工具（可选）
# print(driving_tool.invoke({"origin": "Paris", "destination": "Berlin"}))
```

此工具遵循与天气工具相同的模式：一个输入模式（`DrivingTimeInput`）、一个核心逻辑函数（`estimate_driving_time`）和一个`BaseTool`子类（`DrivingTimeTool`）。

### 步骤3：创建和配置代理

既然我们有了自定义工具，现在将它们整合到一个代理中。我们将使用**工具调用代理**。这是GPT-3.5和GPT-4等模型的现代标准，它采用模型原生的API进行函数调用，而不是依赖脆弱的文本解析（如旧的ReAct模式）。

```python
# --- 代理设置 ---
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 确保在您的环境变量或.env文件中设置了OPENAI_API_KEY
# os.environ["OPENAI_API_KEY"] = "your_api_key"
if not os.getenv("OPENAI_API_KEY"):
    print("警告：OPENAI_API_KEY未设置。代理执行很可能会失败。")

# 1. 初始化LLM
# 工具调用需要支持此功能的聊天模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 定义工具列表
tools = [weather_tool, driving_tool]

# 3. 获取提示模板
# 获取一个为工具调用代理优化的预定义提示
# 您可以在LangChain Hub上查看其他提示
prompt = hub.pull("hwchase17/openai-tools-agent")

# 4. 创建工具调用代理
# 这将LLM、工具和提示绑定在一起，借助OpenAI工具API。
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. 创建代理执行器
# 这将运行代理循环
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # 设置为True以查看代理的工具使用情况
    max_iterations=5 # 防止潜在的无限循环
)

print("代理执行器创建成功。")
```

让我们分析一下代理的创建过程：

1. **LLM 初始化:** 我们实例化`ChatOpenAI`。温度设置为0，以便为工具使用提供更确定的响应。
2. **工具列表:** 我们将自定义的`weather_tool`和`driving_tool`实例收集到一个列表中。
3. **提示模板:** 我们从LangChain Hub获取`hwchase17/openai-tools-agent`。此提示专为处理工具调用模型所需的系统指令而设计。
4. **`create_tool_calling_agent`:** 此函数构建代理逻辑。与需要复杂文本解析指令的传统代理不同，此代理在内部使用`bind_tools`方法将我们的工具定义直接附加到API调用中。
5. **`AgentExecutor`:** 这是代理的运行时环境。它管理循环：将输入发送到LLM，执行LLM请求的工具，并将输出反馈给LLM。

### 步骤4：运行代理并观察行为

代理执行器准备就绪后，我们用不同的查询来测试它。

```python
# --- 运行代理 ---

print("\n--- 运行简单天气查询 ---")
response1 = agent_executor.invoke({
    "input": "What's the weather like right now in Toronto?"
})
print("\n最终答案:", response1['output'])

print("\n--- 运行简单驾驶时间查询 ---")
response2 = agent_executor.invoke({
    "input": "How long does it take to drive from Berlin to Munich?"
})
print("\n最终答案:", response2['output'])

print("\n--- 运行多工具查询 ---")
response3 = agent_executor.invoke({
    "input": "Can you tell me the current weather in Rome and also how long it might take to drive there from Naples?"
})
print("\n最终答案:", response3['output'])

# 代理可以直接回答的查询示例（如果可能）
# print("\n--- 运行非工具查询 ---")
# response4 = agent_executor.invoke({
#     "input": "What is the capital of France?"
# })
# print("\n最终答案:", response4['output'])
```

当`verbose=True`时观察输出。您将看到与旧ReAct代理不同的模式：

1. **调用:** 您会看到代理直接调用工具，而不是“思考”痕迹。例如：`Invoking: Weather Checker with {'location': 'Toronto'}`。
2. **结果:** 执行器捕获`weather_tool`的输出并进行记录。
3. **(必要时重复):** 对于多工具查询，代理可能会在收到天气数据后立即调用驾驶工具。
4. **最终答案:** LLM将工具输出综合成自然语言响应。

请密切注意代理如何使用`BaseTool`子类中定义的*精确*名称和输入模式。工具描述的质量对代理选择正确工具的能力非常必要。

### 总结与后续步骤

本实践练习演示了创建具有自定义能力的LangChain代理的基本工作流程：

1. **明确需求:** 确定代理必须执行的特定任务。
2. **实现工具:** 使用`BaseTool`或`@tool`装饰器为每种能力创建函数或类。请特别注意`name`、`description`和`args_schema`。
3. **配置代理:** 为现代LLM使用`create_tool_calling_agent`，以确保可靠的工具使用，避免解析错误。
4. **实例化执行器:** 创建`AgentExecutor`来管理运行时循环。
5. **测试和观察:** 运行查询并使用`verbose=True`来验证代理是否选择了正确的工具。

从这里，您可以进一步查看：

- **更复杂的工具:** 整合与数据库或专有API交互的工具。
- **LangGraph:** 对于需要复杂状态管理、循环或人机协作工作流程的生产应用，可以考虑从`AgentExecutor`迁移到**LangGraph**，它提供了一个更易于控制的基于图的执行环境。
- **异步执行:** 通过确保您的工具和代理执行器为I/O密集型任务有效使用异步操作（`_arun`方法）来优化性能。

## 参考资料

- [LangChain Documentation: Tools](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF1fmBZzfhX98wFrGNeABeUderWA6TrZX6r64JDWHbWPuGnTTrSU1Q0mj7AqxGvntkE6W1QGtutXrIKykklu9SbpFOpa4qCU-DR7oL_JOcCHgFa31Xs6Q5Fm9m1pN8gHwmtTjRwyOwvRyR_cBk=) — LangChain (2024)
  LangChain框架中实现自定义工具的官方文档，包括BaseTool和Pydantic作为输入模式等概念。
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izzet Akman, Penporn Koanantakool, Chen Sun, Kai Wei, Jaehoon Lee, Igor Mordatch, Jordan Soares, Yuan Cao, Tarun Gulati, Tom Baker, Sheng Wang, Anna Trella, Vincenz Schwab, Kathleen McKeown, Jean-Francois Rajotte, Harsha Surana, Christina Taylor, Agostina Chiachi, Phoenix Williams, Michael Gu, Jeannie Kim, Joshua Lee, Koray Kavukcuoglu, Quoc V. Le, Ed H. Chi, and Jason Wei (2022)
  Journal: arXiv preprint arXiv:2210.03629; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍ReAct框架的原始研究论文，该框架结合了推理和行动，使语言模型能够执行复杂任务，与本节中使用的代理类型直接相关。
- [Pydantic v1 Documentation](https://docs.pydantic.dev/1.10/) — Samuel Colvin and Pydantic contributors (2024)
  Pydantic v1的官方文档，解释了如何使用Python类型提示进行数据验证和设置管理，用于定义LangChain工具的结构化输入模式。
- [A Survey on Large Language Model Based Agents](https://arxiv.org/abs/2309.07864) — Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Tao Gui (2023)
  Journal: arXiv preprint arXiv:2309.07864; DOI: [10.48550/arXiv.2309.07864](https://doi.org/10.48550/arXiv.2309.07864)
  一项学术调查，全面概述了基于大型语言模型的代理，涵盖了它们的架构、应用和研究方向。
- [OpenAI Platform: Tool use](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI
  OpenAI关于使大型语言模型定义和使用外部工具的官方指南，这代表了许多代理系统的底层机制。

---

[上一节](06-%E6%99%BA%E8%83%BD%E4%BD%93%E6%89%A7%E8%A1%8C%E8%BF%BD%E8%B8%AA%E4%B8%8E%E5%88%86%E6%9E%90.md) · [下一节](../03-%E8%BF%9B%E9%98%B6%E8%AE%B0%E5%BF%86%E7%AE%A1%E7%90%86%E6%96%B9%E6%B3%95/01-%E9%AB%98%E7%BA%A7%E8%AE%B0%E5%BF%86%E7%B1%BB%E5%9E%8B%E6%AF%94%E8%BE%83.md)
