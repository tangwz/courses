---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-selection-orchestration"
lesson: "practice-multi-tool-agent-orchestration"
sourceId: 6583
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-3-llm-tool-selection-orchestration/practice-multi-tool-agent-orchestration"
title: "实践：协调多工具代理"
description: "一次实际操作课程，内容是构建一个能够协调运用多个工具来完成任务的代理。"
order: 7
plots: []
sourceHash: "5481d5e6453b9dedeb6df48d6ba63f5f261fde66b417ab02a51f575801f20764"
sourceCorrections: []
---

我们已经审视了一个代理如何选择工具以及按顺序使用它们的原理，现在是时候将这些想法付诸实践了。在这个动手练习中，您将构建一个简化的代理，它协调使用多个工具来完成用户请求。我们将着重于一个常见情景：规划一次简单的外出。这会涉及获取信息、根据这些信息提出建议，然后寻找一个相关的便利设施。

我们的目标是说明代理如何执行一系列工具调用，一个工具的输出通常会成为下一个工具的输入。这显示了本章前面讨论的多步骤执行和依赖管理的核心原理。

### 情景：出行规划代理

想象一个用户希望获得本地活动规划方面的帮助。他们可能会问：“这周六在旧金山有什么好的活动，附近哪里可以吃饭？” 为了回答这个问题，我们的代理需要：

1. 确定指定地点和日期的天气。
2. 建议一个适合那种天气的活动。
3. 在建议的活动地点附近找到一家餐厅。

这显然需要一系列操作，因此它非常适合多工具协调。

### 定义工具

定义三个Python函数，每个函数代表一个不同的工具。在实际应用中，工具通常与外部API交互，但在此处使用简化的实现，以突出协调。

1. `get_weather_forecast`：接收地点和日期，返回包含天气状况和温度的字典。
2. `suggest_activity`：接收天气状况并建议一个活动。
3. `find_restaurant`：接收城市和活动描述（例如“公园附近”），返回一个餐厅建议。

以下是这些工具的Python实现。请注意它们的输入参数 (parameter)和返回值的结构。清晰的输入/输出模式对工具的可靠使用很重要，如第1章中讨论过的那样。

```python
# tool_definitions.py

def get_weather_forecast(location: str, date: str) -> dict:
    """
    模拟获取天气预报。
    在真实的工具中，这会调用天气API。
    """
    print(f"工具调用：get_weather_forecast(location='{location}', date='{date}')")
    if location.lower() == "san francisco":
        if "saturday" in date.lower() or "sunday" in date.lower(): # 模拟周末天气
            return {"city": location, "condition": "Sunny", "temperature_celsius": 22}
        else:
            return {"city": location, "condition": "Foggy", "temperature_celsius": 18}
    elif location.lower() == "london":
        return {"city": location, "condition": "Cloudy", "temperature_celsius": 15}
    else:
        return {"city": location, "condition": "Partly Cloudy", "temperature_celsius": 20}

def suggest_activity(weather_condition: str) -> dict:
    """
    根据天气状况建议一个活动。
    """
    print(f"工具调用：suggest_activity(weather_condition='{weather_condition}')")
    if weather_condition == "Sunny":
        return {"activity_type": "Outdoor", "suggestion": "Go for a hike in a nearby park"}
    elif weather_condition == "Cloudy" or weather_condition == "Partly Cloudy":
        return {"activity_type": "Indoor/Outdoor", "suggestion": "Visit a local museum or a covered market"}
    elif weather_condition == "Foggy":
        return {"activity_type": "Indoor", "suggestion": "Explore an art gallery or enjoy a cozy cafe"}
    else: # 其他状况的默认值
        return {"activity_type": "Flexible", "suggestion": "Check out local events"}

def find_restaurant(city: str, activity_type: str, preference: str = "near activity") -> dict:
    """
    根据城市、活动类型和偏好模拟寻找餐厅。
    'preference' 可用于寻找“活动附近”的地点或特定菜系。
    """
    print(f"工具调用：find_restaurant(city='{city}', activity_type='{activity_type}', preference='{preference}')")
    if city.lower() == "san francisco":
        if activity_type == "Outdoor":
            return {"name": "The Green Eatery", "cuisine": "Californian", "vicinity": "near Golden Gate Park"}
        elif activity_type == "Indoor":
            return {"name": "City Lights Cafe", "cuisine": "Italian", "vicinity": "Downtown"}
        else:
            return {"name": "The Bay Diner", "cuisine": "American", "vicinity": "Fisherman's Wharf"}
    elif city.lower() == "london":
        return {"name": "The Thames Bistro", "cuisine": "British", "vicinity": "near South Bank"}
    else:
        return {"name": "Local Gem Restaurant", "cuisine": "Varied", "vicinity": "city center"}
```

### 协调工具调用

定义好工具后，我们现在可以概述代理的思路了。代理将接收初始查询，然后按特定顺序调用工具，将一个工具的输出用作下一个工具的输入。

以下是计划的工具执行流程的可视化展现：

> 该图说明了信息的流动。用户的请求启动了该过程。`get_weather_forecast` 的输出（例如“晴朗”）被 `suggest_activity` 使用。`get_weather_forecast`（用于城市信息）和 `suggest_activity`（用于活动类型）都为 `find_restaurant` 提供输入。最后，代理整合所有收集到的信息来生成一个回复。

现在，让我们在一个模拟代理的Python函数中实现这种协调思路。

```python
# agent_orchestrator.py
from tool_definitions import get_weather_forecast, suggest_activity, find_restaurant

def trip_planner_agent(query_location: str, query_date: str):
    """
    根据用户查询协调工具调用来规划行程。
    """
    print(f"\n代理：收到规划 {query_location} 在 {query_date} 的请求。")

    # 第1步：获取天气预报
    print("代理：第1步 - 获取天气预报...")
    weather_info = get_weather_forecast(location=query_location, date=query_date)
    if not weather_info:
        print("代理：无法获取天气信息。中止计划。")
        return "抱歉，我无法获取您请求的天气信息。"

    print(f"代理：{weather_info.get('city', 'N/A')} 的天气是 {weather_info.get('condition', 'N/A')}，气温 {weather_info.get('temperature_celsius', 'N/A')}°C。")

    # 第2步：根据天气建议一个活动
    # 此步骤依赖于get_weather_forecast的输出
    print("\n代理：第2步 - 建议一个活动...")
    activity_info = suggest_activity(weather_condition=weather_info.get('condition'))
    if not activity_info:
        print("代理：无法建议活动。中止计划。")
        return "抱歉，我无法根据天气提出活动建议。"

    print(f"代理：建议的活动类型：{activity_info.get('activity_type', 'N/A')}，建议：{activity_info.get('suggestion', 'N/A')}。")

    # 第3步：寻找餐厅
    # 此步骤依赖于城市（来自weather_info）和活动类型（来自activity_info）
    print("\n代理：第3步 - 寻找餐厅...")
    restaurant_info = find_restaurant(city=weather_info.get('city'), activity_type=activity_info.get('activity_type'))
    if not restaurant_info:
        print("代理：未能找到餐厅。继续部分计划。")
        # 我们可以选择返回部分计划或失败。在这里，我们选择继续。
        restaurant_suggestion_text = "目前未能找到具体的餐厅建议。"
    else:
        restaurant_suggestion_text = (
            f"用餐方面，您可以尝试 {restaurant_info.get('name', '一个当地餐厅')} "
            f"（{restaurant_info.get('cuisine', '多种菜系')}） "
            f"位于 {restaurant_info.get('vicinity', '附近')}。"
        )
        print(f"代理：找到餐厅：{restaurant_info.get('name', 'N/A')}。")

    # 第4步：为用户组织最终回复
    print("\n代理：第4步 - 组织最终计划...")
    final_plan = (
        f"好的，这是您 {query_location} 在 {query_date} 的计划：\n"
        f"- 天气预计为 {weather_info.get('condition')}，气温约为 {weather_info.get('temperature_celsius')}°C。\n"
        f"- 基于此，我建议您：{activity_info.get('suggestion')}。\n"
        f"- {restaurant_suggestion_text}"
    )

    print("\n代理：生成最终计划：")
    print(final_plan)
    return final_plan

# 运行我们的代理
if __name__ == "__main__":
    user_location = "San Francisco"
    user_date = "Saturday"
    plan = trip_planner_agent(query_location=user_location, query_date=user_date)

    print("\n--- 另一个地点的例子 ---")
    user_location_2 = "London"
    user_date_2 = "next Tuesday"
    plan_2 = trip_planner_agent(query_location=user_location_2, query_date=user_date_2)
```

当您运行 `agent_orchestrator.py` 时，您将看到输出日志，显示每个工具的调用以及代理在每个步骤的内部思考过程：

```text
代理：收到规划旧金山在周六的请求。
代理：第1步 - 获取天气预报...
工具调用：get_weather_forecast(location='San Francisco', date='Saturday')
代理：旧金山的天气是晴朗，22°C。

代理：第2步 - 建议一个活动...
工具调用：suggest_activity(weather_condition='Sunny')
代理：建议的活动类型：户外，建议：在附近的公园远足。

代理：第3步 - 寻找餐厅...
工具调用：find_restaurant(city='San Francisco', activity_type='Outdoor', preference='near activity')
代理：找到餐厅：The Green Eatery。

代理：第4步 - 组织最终计划...

代理：生成最终计划：
好的，这是您旧金山周六的计划：
- 天气预计为晴朗，气温约为22°C。
- 基于此，我建议您：在附近的公园远足。
- 用餐方面，您可以尝试 The Green Eatery（加州菜），位于金门公园附近。

--- 另一个地点的例子 ---
代理：收到规划伦敦在下周二的请求。
代理：第1步 - 获取天气预报...
工具调用：get_weather_forecast(location='London', date='next Tuesday')
代理：伦敦的天气是多云，15°C。

代理：第2步 - 建议一个活动...
工具调用：suggest_activity(weather_condition='Cloudy')
代理：建议的活动类型：室内/室外，建议：参观当地博物馆或有顶市场。

代理：第3步 - 寻找餐厅...
工具调用：find_restaurant(city='London', activity_type='Indoor/Outdoor', preference='near activity')
代理：找到餐厅：The Thames Bistro。

代理：第4步 - 组织最终计划...

代理：生成最终计划：
好的，这是您伦敦下周二的计划：
- 天气预计为多云，气温约为15°C。
- 基于此，我建议您：参观当地博物馆或有顶市场。
- 用餐方面，您可以尝试 The Thames Bistro（英式菜），位于南岸附近。
```

### 分析协调

这个动手练习显示了工具协调的几个重要方面：

- **多步骤执行：** 整个任务（“规划行程”）被分解为一系列更小、可管理的步骤，每个步骤由一个特定工具处理。
- **依赖管理：** 执行流程明确地管理了依赖关系。`suggest_activity` 工具需要来自 `get_weather_forecast` 的 `condition`。`find_restaurant` 工具使用了来自 `get_weather_forecast` 的 `city` 和来自 `suggest_activity` 的 `activity_type`。输入和输出的这种链式连接是复杂工具使用的基本要求。
- **工具选择（简化）：** 在这个例子中，工具序列被硬编码到我们的 `trip_planner_agent` 函数中。在更高级的代理中，大型语言模型会根据对话历史、当前目标和可用工具的描述（如“代理驱动的工具选择机制”中讨论的）动态决定下一步调用哪个工具。我们这种明确的序列安排旨在清晰地说明协调流程。
- **数据转换/聚合：** 代理并非盲目传递数据。它会从工具输出中提取特定的信息片段（例如，`weather_info.get('condition')`），用作后续工具的输入，或用于构建最终回复。

### 进一步考虑

虽然这个例子是简化的，但它为更复杂的协调打下了基础。以下是针对更高级情景可以考虑的几点：

- **错误处理：** 我们的代理对缺失的工具输出有基本检查。代理需要更完善的错误处理，例如重试工具调用、尝试替代工具，或者当工具失败或返回意外结果时向用户请求澄清（请参阅“工具链中故障的恢复”）。
- **条件逻辑：** 我们可以引入条件路径。例如，如果天气是“下雨”，代理可能会跳过建议户外活动，或者明确使用专门用于寻找室内活动的工具。这与“条件工具执行逻辑”有关。
- **并行执行：** 对于需要多个独立信息的任务，代理可能会并行执行一些工具调用以节省时间（请参阅“实现顺序和并行工具使用”）。例如，如果我们同时需要查找某个地点的开放时间和预订交通，如果它们之间没有依赖，这些操作可能并行执行。

这个动手练习让您实际了解了代理如何协调多个工具以完成复杂目标。通过定义清晰的工具并协调它们的执行，您可以显著提升大型语言模型代理的能力。随着您构建更完善的代理，顺序执行、依赖管理以及（最终）动态工具选择的原理将是其设计的核心。尝试修改现有工具，或添加新工具（例如，`get_event_listings` 工具），并思考如何将其整合到代理的协调思路中。

## 参考资料

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/) — LangChain (2024)
  Publisher: LangChain
  LangChain代理模块的官方文档，为构建利用工具并管理其执行流的代理提供了实用模式。
- [A Survey on Large Language Model based Autonomous Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEBcCYtTIvWLtEpJbZf1aQ0TNfSgBcvPJ7ntr0V-5Oeps56I36HF3FgFazi6boRAxrl3xJ4dpxf8fmfnqVZIBh4lI2TI98_8qYRBGZOUpSSFr1IKwB69ZwPSg==) — Lei Wang, Chen Ma, Xueyang Feng, Zhiyuan Liu, Maosong Sun, Lei Hou (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  提供了基于大型语言模型的自主代理的综述，涵盖其结构、工具交互和多步推理能力。
