---
course: "building-advanced-llm-agent-tools"
chapter: "llm-tool-selection-orchestration"
lesson: "tool-dependency-management"
sourceId: 6575
sourceUrl: "https://apxml.com/zh/courses/building-advanced-llm-agent-tools/chapter-3-llm-tool-selection-orchestration/tool-dependency-management"
title: "管理工具调用间的依赖关系"
description: "了解如何在序列中管理一个工具的输出成为另一个工具输入的情况。"
order: 3
plots: []
sourceHash: "f06efa96598cfe20766c03aa9ea359008a8f38599ce8df0a2901a8c2a9ac7aa4"
sourceCorrections: []
---

当智能体需要完成一项任务，而单个工具不足以处理时，它通常会将多个工具串联起来使用。在这些多步骤的流程中，一个工具的输出常常会成为后续工具所需的信息。这种信息流动会产生依赖关系：工具B在工具A提供所需数据之前无法执行其功能。有效管理这些依赖关系对于构建能够执行复杂计划的精巧智能体来说是基础。

设想一个智能体负责规划周末行程。它可能首先使用 `get_flight_prices` 工具。该工具的输出，比如最便宜的航班选项及其日期和时间，随后成为 `book_hotel` 工具的重要输入，该工具需要知道抵达和离开日期才能找到合适的住宿。如果没有航班详情，酒店预订工具就无法进行。这是一种常见的模式：一个步骤的成功执行能使下一个步骤得以进行。

### 识别和传递依赖数据

智能体，或您设计的底层编排逻辑，需要一种方式来理解和管理这些数据交接。通常有两种方式处理这些依赖关系：

1. **智能体驱动的数据流：** LLM本身作为其推理 (inference)过程的一部分，可以确定 `tool_A_output` 的输出应作为 `tool_B` 的 `parameter_x` 使用。这在很大程度上依赖于编写良好的工具描述（如第1章所述），这些描述清楚地指明了工具的输出和所需的输入。LLM会根据工具A的结果，生成带有必要数据映射的工具B调用。
2. **编排器管理的数据流：** 在更结构化的智能体框架或定制编排器中，您可能明确定义数据如何在工具之间流动。编排器执行工具A，捕获其输出，然后在工具B需要运行时，以程序方式将输出的相关部分传递给工具B。

无论LLM还是编排器主要管理数据流，数据传递的机制通常涉及以下方法之一：

#### 直接输出到输入映射

这是最直接的方法。智能体（或编排器）获取前一个工具的直接输出，并将其作为参数 (parameter)传递给下一个工具。

例如，如果 `tool_A` 返回一个JSON对象，如 `{"user_id": "123", "email": "user@example.com"}`，而 `tool_B` 需要一个 `user_identifier`，系统会将 `tool_A_output.user_id` 映射到 `tool_B` 的 `user_identifier` 参数。

```python
# 演示性的Python类伪代码
user_data_from_tool_A = agent.execute_tool("fetch_user_profile", user_name="Alice")
# user_data_from_tool_A 可能是：{"id": "u456", "preferences": ["music", "hiking"]}

if user_data_from_tool_A and user_data_from_tool_A.get("id"):
    recommendations = agent.execute_tool(
        "get_recommendations",
        user_id=user_data_from_tool_A["id"],
        categories=user_data_from_tool_A.get("preferences", [])
    )
    # 处理推荐结果
else:
    # 处理user_id缺失或tool_A执行失败的情况
    print("Could not retrieve user ID to get recommendations.")
```

在此代码片段中，`user_data_from_tool_A` 中的 `id` 字段直接用作 `get_recommendations` 的 `user_id` 参数。

#### 使用共享上下文 (context)或暂存区

对于更复杂的序列，或者当多个先前的工具输出共同构成后续工具的输入时，共享上下文（有时称为“暂存区”或“内存”）会非常有效。每个工具可以将其结果写入这个共享空间中明确定义的位置。随后的工具可以从该上下文中读取，以获取其所需的输入。

考虑一个帮助用户分析销售数据的智能体：

1. `Tool_LoadData`：将CSV中的销售数据加载到上下文中，作为 `context["sales_data"]`。
2. `Tool_FilterData`：获取 `context["sales_data"]`，应用过滤器（例如，针对特定区域），并将结果写入 `context["filtered_sales_data"]`。
3. `Tool_CalculateTotal`：读取 `context["filtered_sales_data"]` 并计算总数，将其写入 `context["total_sales_for_region"]`。
4. `Tool_GenerateReport`：读取 `context["total_sales_for_region"]` 和 `context["filtered_sales_data"]` 以生成摘要。

```python
# 演示性的上下文使用
agent_context = {}

# 步骤 1：获取用户位置
location_data = agent.execute_tool("get_user_location") # 例如，{"city": "London", "country": "UK"}
if location_data and location_data.get("city"):
    agent_context["user_city"] = location_data["city"]
else:
    # 处理获取位置失败的情况
    agent_context["user_city"] = "default_city" # 备用方案或错误处理

# 步骤 2：根据上下文中的位置获取天气
if "user_city" in agent_context:
    weather_report = agent.execute_tool("get_weather_forecast", city=agent_context["user_city"])
    # weather_report 可能是：{"temperature_celsius": 15, "condition": "Cloudy"}
    if weather_report:
        agent_context["current_temp_celsius"] = weather_report.get("temperature_celsius")
        agent_context["current_condition"] = weather_report.get("condition")

# 步骤 3：根据上下文中的天气建议活动
if "current_temp_celsius" in agent_context and "current_condition" in agent_context:
    activity_suggestion = agent.execute_tool(
        "suggest_activity",
        temperature=agent_context["current_temp_celsius"],
        weather_condition=agent_context["current_condition"]
    )
    print(f"Suggested activity: {activity_suggestion}")
```

虽然这种方式灵活，但使用共享上下文需要仔细管理，以避免键名冲突，并确保数据不会被意外覆盖或变得过时。

#### 工具间的数据转换

有时，一个工具的输出格式与下一个工具所需的输入格式不完全一致。例如，`Tool_A` 可能输出摄氏温度，但 `Tool_B` 期望华氏温度。或者 `Tool_A` 返回一个复杂对象，而 `Tool_B` 只需其中的一个字段。

在这种情况下，需要一个转换步骤。这种转换可以是：

- **由LLM执行：** 如果LLM在编排调用，它可能会被指示（或自行推断）重新格式化或提取数据。例如，“从天气工具的输出中获取 'temp\_c' 字段，将其转换为华氏温度，并将其用作服装建议工具的 'temperature' 输入。”
- **由编排器处理：** 您的智能体编排代码可以包含小型实用函数或步骤来执行这些转换。这对于精确的数值转换或结构改变通常更可靠。
- **内置于消费工具中：** 工具B可以设计为接受多种格式或在内部执行常见转换，但这会使工具B更复杂。

目标是确保传递给工具的数据结构和格式能够被工具可靠地处理。您的工具的清晰输入和输出模式，如第1章所述，能大大简化这一点。

### 可视化工具链中的数据流

当您可以可视化数据流时，理解依赖关系会变得更容易。对于一系列工具，您可以将其视为一个有向图，其中节点是工具，边表示数据传递。

> 一个简单的数据流图，显示 `ToolA` 向 `ToolB` 提供 `user_id`，然后 `ToolB` 向 `ToolC` 提供 `order_list`。

这种可视化表示有助于设计和调试复杂的工具交互，确保链中的每个工具都从其前序工具接收到所需的输入。

### 处理依赖调用中的故障

管理依赖关系的一个重要方面是，决定如果链中的前一个工具失败或未返回预期数据时该怎么做。如果 `Tool_A` 失败，`Tool_B`（它依赖于 `Tool_A` 的输出）就无法按计划进行。

处理此类故障的策略包括：

- **重试失败的工具：** 也许故障是暂时的。
- **使用备用值或默认值：** 如果适用于任务。
- **调用替代工具：** 如果有其他工具可以提供类似数据。
- **终止当前子任务：** 并向智能体或用户报告失败。
- **重新规划：** 智能体可能需要设计一个新的工具调用序列。

我们将在本章后面的“工具链中从故障中恢复”部分更详细地讨论错误恢复。现在，认识到依赖管理策略必须考虑到潜在的上游故障是很重要的。

### 依赖管理的好实践

在设计使用工具序列的智能体时，请考虑以下实践：

- **清晰的工具签名：** 确保您的工具定义（描述、输入参数 (parameter)、输出模式）精确。这使得LLM和任何编排代码都能更容易地理解生成和消费的数据。
- **标准化数据格式：** 工具输出首选JSON等常见且易于解析的数据格式。这简化了数据交换和转换。
- **隔离转换逻辑：** 如果需要数据转换，请尝试隔离此逻辑。它可以是编排器中的一个小型专用函数，或是对LLM的特定指令，而不是用过多的输入灵活性来增加主要工具的负担。
- **减少复杂相互依赖：** 如果您发现依赖关系过于复杂交织，这可能表明您的工具粒度过细，或者任务分解可以改进。在可能的情况下，目标是实现清晰、大多线性的流程，或者包含良好的分支。
- **显式与隐式依赖：** 注意依赖关系是在编排逻辑中显式声明，还是由LLM隐式处理。显式声明通常更可靠且易于调试，而LLM处理的依赖关系提供更大的灵活性，但可能更难预测。

通过深思熟虑地管理数据如何在工具之间流动，您可以使您的LLM智能体能够执行更精巧、多步骤的任务，从简单的工具调用转向协调的工作流程。这种将输出连接到输入的能力使得智能体能够在前一个结果的基础上构建，并实现更重要的目标。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: ICLR 2023
  介绍了LLM代理的ReAct范式，通过共享暂存器迭代地推理、通过工具行动和观察，有效管理顺序工具调用和数据流。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG7Y8y1_fQg3YQhuKau9-lnC7M1Ycbvqa-s6Hc84zve_3w_T0suvLm7A_L0n1rLLEE9CZuCX29wo35IwaaCbcnYlsXv4rDiEqxUr-1CTl001yW-ew2bYXXAQF3dPl8NaD9wnrYOOAQ=) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442
  详细介绍了一种具有长期记忆和规划的代理架构，阐明了复杂多步骤行为是如何编排的，以及代理的内部状态（上下文）如何通过交互和工具使用而演变。
- [Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQFfd8S-Z46DQ_1enKWzl0UbdBqYEh7OyEB8i3r2CW8DPBrrjC2xwgUWMeTfEuNhgupDQgPVaDqp0RuAAuGixyewMU0wZJvLXA1AcocSQe0DsaD72-e-m2xqYn5QKPz2OkovG_9tdaodC3oXGWKMBgOWu9ejbw==) — LangChain documentation team (2024)
  Publisher: LangChain
  提供了构建链式工具LLM代理的实际示例和架构模式，在一个流行框架中展示了编排器管理的数据流、共享上下文和直接输出到输入的映射。
