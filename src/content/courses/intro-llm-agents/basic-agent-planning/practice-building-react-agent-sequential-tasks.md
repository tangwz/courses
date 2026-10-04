---
course: "intro-llm-agents"
chapter: "basic-agent-planning"
lesson: "practice-building-react-agent-sequential-tasks"
sourceId: 6609
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-5-basic-agent-planning/practice-building-react-agent-sequential-tasks"
title: "实践：构建一个处理顺序任务的 ReAct 智能体"
description: "一次关于使用 ReAct 框架实现智能体以处理多步任务的实践环节。"
order: 8
plots: []
sourceHash: "9052f068c5cf8be7f0856ce49575693f715888b2eaeafb8fbfe598dc2093fc49"
sourceCorrections: []
---

一个亲手操作的例子将演示如何构建一个采用 ReAct（思考与行动）模式的智能体。我们将引导你创建一个能够执行一系列行动以达成一个特定目标的智能体。ReAct 方法是指智能体循环进行思考（对任务进行推理 (inference)）、行动（决定一个行动，通常涉及一个工具）和观察（处理该行动的结果）。

本次实践练习的目标是构建一个能够做到以下几点的智能体：

1. 查找指定城市的当前天气。
2. 根据天气情况，推荐一项合适的户外活动。

该任务至少需要两个步骤并使用一个（模拟）工具，使其成为 ReAct 框架的一个合适范例。

### 准备工作：智能体目标和工具

首先，让我们明确定义智能体的整体目标：
`"查找英国伦敦当前的当地天气。然后，推荐一项适合该天气的户外活动。"`

为完成此任务，我们的智能体将需要一个工具来获取天气信息。为了简化本次初学者练习，我们将定义一个 Python 函数来模拟这个工具。在实际场景中，这个函数可能会调用外部天气 API。

```python
# 一个简单的模拟工具
def get_weather(city: str) -> str:
    """模拟为城市获取天气信息。"""
    print(f"TOOL_CALL: get_weather(city='{city}')")
    if city.lower() == "london, uk":
        # 我们可以让天气在不同运行中变化，或为简化起见固定不变。
        # 为此例中可预测的输出，目前固定。
        return "英国伦敦的天气是 15°C 晴朗。"
    elif city.lower() == "paris, fr":
        return "法国巴黎的天气是 18°C 局部多云。"
    else:
        return f"抱歉，我没有 {city} 的天气信息。"

# 我们还需要一种方式让智能体表示它已完成任务。
# 这不是一个工具，而是一种特殊的行动格式。
# Action: Finish[final_answer_string]
```

### ReAct 循环：思考、行动、观察

我们的 ReAct 智能体的核心是一个循环。在这个循环的每次迭代中，智能体将：

1. **思考（推理 (inference)）：** LLM 处理当前目标、任何之前步骤（历史记录）和最新观察结果。然后，它形成一个关于下一步做什么的想法。
2. **行动：** 基于其想法，LLM 决定一个行动。这可以是使用一个工具（如 `get_weather`），或者如果任务已完成，使用特殊的 `Finish` 行动。
3. **观察：** 智能体（我们的 Python 代码）执行该行动。如果它是一个工具调用，工具的输出将成为观察结果。如果它是 `Finish`，循环终止。

让我们可视化这个核心循环：

> ReAct 循环：智能体使用当前状态（包括最新观察结果）来提示 LLM，LLM 随后生成一个思考和一个行动。该行动被执行，产生一个新的观察结果，循环重复。

### 编写 LLM 提示词 (prompt)

提示词是我们指导 LLM 在 ReAct 框架内行为的方式。它需要告诉 LLM：

- 其总体目标。
- 可用的工具以及如何使用它们（它们的“签名”或格式）。
- 思考和行动的格式（例如，`Thought: ... Action: ...`）。
- 如何表示完成（例如，`Action: Finish[...]`）。
- 之前思考-行动-观察步骤的历史记录。
- 最新观察结果。

这里是我们的提示词模板：

```python
# 这是一个 Python 字符串模板，本身不可执行
prompt_template = """
你是一个乐于助人的助手。你的目标是：{goal}

你可以使用以下工具：
- get_weather[city_name]: 返回指定城市的当前天气。

当你掌握所有信息时，请使用以下格式提供最终答案：
Action: Finish[your_final_answer_string]

请始终遵循此循环：
Thought: （你对当前状态、所学内容以及为达成目标下一步做什么的推理）
Action: （要采取的行动，可以是使用工具或 Finish）

这是你目前工作的历史记录（思考/行动/观察三元组）：
{history}

当前观察结果: {observation}
Thought:"""
```

`{goal}`、`{history}` 和 `{observation}` 占位符将在循环的每个步骤中填充。LLM 预计会生成以 `Thought:` 开头的文本。

### 实现智能体循环（简化的 Python 草图）

让我们勾勒出 Python 逻辑。假设你有一个函数 `call_llm(prompt_text)`，它将提示词 (prompt)发送给 LLM 并返回其响应。

```python
# --- 智能体设置 ---
agent_goal = "查找英国伦敦当前的当地天气。然后，推荐一项适合该天气的户外活动。"
history_log = []  # 用于存储 (思考, 行动字符串, 观察结果) 元组
current_observation = "暂无观察结果。让我们开始。"
max_steps = 5  # 防止无限循环

# --- 模拟 LLM 调用 ---
# 在实际应用中，这将是对 LLM 服务的 API 调用。
# 为此例，我们将模拟前几个步骤的 LLM 响应
# 以说明流程，无需真实的 LLM。

def mock_llm_call(prompt_text):
    # 这个模拟函数将根据提示词内容模拟 LLM 的行为。
    # 为此例，它被高度简化。
    print("\n--- LLM 提示词 ---")
    print(prompt_text)
    print("--- LLM 提示词结束 ---\n")

    if "get_weather[London, UK]" not in prompt_text and "Thought:" in prompt_text.split("Current Observation:")[-1]:
        # 第一步，LLM 应该决定获取天气
        return "我需要查找英国伦敦的天气。我应该使用 get_weather 工具。\nAction: get_weather[London, UK]"
    elif "Observation: The weather in London, UK is 15\u00b0C and sunny." in prompt_text:
        # 第二步，LLM 已获取天气，需要推荐活动
        return "天气是 15°C 晴朗。这很宜人。一个好的户外活动将是在公园里散步。\nAction: Finish[英国伦敦的天气是 15°C 晴朗。一项合适的户外活动是在公园里散步。]"
    else:
        # 此模拟中意外提示词的备用方案
        return "Thought: 我不确定如何继续。\nAction: Finish[未能完成任务。]"

# --- 解析 LLM 输出的辅助函数 ---
def parse_llm_response(response_text):
    thought = ""
    action_str = ""
    if "Thought:" in response_text:
        thought = response_text.split("Thought:")[1].split("Action:")[0].strip()
    if "Action:" in response_text:
        action_str = response_text.split("Action:")[1].strip()
    return thought, action_str

# --- ReAct 循环 ---
for step in range(max_steps):
    print(f"\n--- 步骤 {step + 1} ---")

    # 1. 构建提示词
    history_str = "\n".join([f"Thought: {t}\nAction: {a}\nObservation: {o}" for t, a, o in history_log])

    current_prompt = prompt_template.format(
        goal=agent_goal,
        history=history_str if history_log else "暂无历史记录。",
        observation=current_observation
    )

    # 2. 从 LLM 获取思考和行动
    llm_response_text = mock_llm_call(current_prompt) # 使用 mock_llm_call 或你真实的 LLM
    thought, action_str = parse_llm_response(llm_response_text)

    print(f"LLM 思考: {thought}")
    print(f"LLM 行动: {action_str}")

    # 3. 执行行动并获取观察结果
    if not action_str:
        print("错误: LLM 未提供行动。")
        current_observation = "错误: LLM 未提供行动。"
        history_log.append((thought, "无行动", current_observation))
        continue # 或 break

    if action_str.startswith("Finish["):
        final_answer = action_str[len("Finish["):-1]
        print(f"\n最终答案: {final_answer}")
        break # 退出循环
    elif action_str.startswith("get_weather["):
        city_name = action_str[len("get_weather["):-1]
        current_observation = get_weather(city_name)
    else:
        print(f"错误: 未知行动: {action_str}")
        current_observation = f"错误: 未知行动 '{action_str}'。请使用可用工具或 Finish。"

    print(f"观察结果: {current_observation}")
    history_log.append((thought, action_str, current_observation))

    if step == max_steps - 1:
        print("\n已达到最大步数。智能体未能完成任务。")
```

### 逐步示例执行

让我们追踪智能体（使用 `mock_llm_call`）的运行方式：

**步骤 1：**

- **初始状态：** `goal` 已设置，`history` 为空，`observation` 是“暂无观察结果...”。
- **提示词 (prompt)构建：** 提示词是根据目标、工具描述和初始观察结果构建的。
- **LLM 调用（模拟）：**
  - 输入提示词（简化）：

    ```
    你是一个乐于助人的助手。你的目标是：查找英国伦敦当前的当地天气...
    ...
    当前观察结果: 暂无观察结果。让我们开始。
    Thought:
    ```
  - 模拟 LLM 输出：`我需要查找英国伦敦的天气。我应该使用 get_weather 工具。\nAction: get_weather[London, UK]`
- **解析输出：**
  - `Thought:` 我需要查找英国伦敦的天气。我应该使用 get\_weather 工具。
  - `Action:` `get_weather[London, UK]`
- **执行行动：** `get_weather("London, UK")` 函数被调用。
  - `TOOL_CALL: get_weather(city='London, UK')`（由我们的工具打印）
- **新观察结果：** `英国伦敦的天气是 15°C 晴朗。`
- **更新历史：** （思考、行动、观察）三元组被添加到 `history_log` 中。

**步骤 2：**

- **状态：** `goal` 相同，`history` 包含步骤 1 的三元组，`observation` 是“英国伦敦的天气是 15°C 晴朗。”
- **提示词构建：** 提示词现在包含步骤 1 的历史记录和新的观察结果。
- **LLM 调用（模拟）：**
  - 输入提示词（简化）：

    ```
    你是一个乐于助人的助手。你的目标是：查找英国伦敦当前的当地天气...
    ...
    历史：
    Thought: 我需要查找英国伦敦的天气...
    Action: get_weather[London, UK]
    Observation: 英国伦敦的天气是 15°C 晴朗。
    当前观察结果: 英国伦敦的天气是 15°C 晴朗。
    Thought:
    ```
  - 模拟 LLM 输出：`天气是 15°C 晴朗。这很宜人。一个好的户外活动将是在公园里散步。\nAction: Finish[英国伦敦的天气是 15°C 晴朗。一项合适的户外活动是在公园里散步。]`
- **解析输出：**
  - `Thought:` 天气是 15°C 晴朗。这很宜人。一个好的户外活动将是在公园里散步。
  - `Action:` `Finish[英国伦敦的天气是 15°C 晴朗。一项合适的户外活动是在公园里散步。]`
- **执行行动：** 该行动以 `Finish[` 开头。
- **结果：** 智能体打印最终答案，循环终止。
  `最终答案: 英国伦敦的天气是 15°C 晴朗。一项合适的户外活动是在公园里散步。`

这个逐步说明说明了 ReAct 智能体如何使用 LLM 的推理 (inference)（思考）来决定一个行动，执行该行动，获得一个观察结果，然后将其反馈给 LLM，以继续该过程直到达成目标。

### 构建你自己的 ReAct 智能体要考虑的方面

- **LLM 选择：** “思考”和“行动”步骤的质量在很大程度上取决于 LLM 的能力。
- **提示词 (prompt)工程很重要：** 提示词的结构和清晰度显著影响智能体的表现。你可能需要尝试并优化你的提示词。
- **工具设计：** 工具应明确定义，具有清晰的输入参数 (parameter)和可预测的输出格式。LLM 需要根据提示词中的描述来理解如何使用它们。
- **LLM 输出解析：** 你需要对 `Thought:` 和 `Action:` 部分进行解析。正则表达式或更结构化的输出格式（例如 JSON，如果你的 LLM 在聊天模式下很好地支持）会很有用。
- **错误处理：** 如果 LLM 生成了无效行动怎么办？或者工具失败了怎么办？你的智能体循环应该有基本的错误处理（例如，将错误消息作为观察结果反馈回去）。
- **历史记录管理：** 对于长时间对话或许多步骤，历史记录会变得非常长。这可能会超出 LLM 的上下文 (context)窗口。更高级的智能体使用历史记录摘要技术，这超出了本介绍性示例的范围。

这次亲手操作的练习提供了构建 ReAct 智能体的基本认识。通过尝试不同的任务、工具和优化你的提示词，你可以创建更复杂的智能体，能够解决日益复杂的顺序问题。请记住，构建智能体通常是一个设计、实现和测试的迭代过程。

## 参考资料

- [A Survey on Large Language Model Based Autonomous Agents](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHawfE-qtNIR9u2fccuGcmiEIQ-85RVZddk-V7fwFJk7durgBfMi9GNR0W1v_kuf4FmY3L4wN_mzIs4a1U2FYzAX7g4BtX4Zj0oJDV8q1bbupeSmgkCoa6TA1E=) — Lei Wang, Chen Ma, Xueyang Feng, Zhiyuan Liu, Maosong Sun, Lei Hou (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  全面概述了基于大型语言模型的自主智能体，涵盖其设计、能力和应用，包括规划和工具使用。
