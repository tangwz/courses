# 实现 ReAct 智能体

来源：[原文](https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-2-advanced-agent-architectures-reasoning/implementing-react-agents)

[返回章节目录](README.md) · [返回课程目录](../README.md)

ReAct 框架将推理 (inference)（`Thought`）与环境交互（`Action`）结合起来，以使智能体能够执行任务。实现此类智能体需要精心设计提示、解析大型语言模型的结构化输出、执行动作以及管理迭代过程。

### ReAct 运行周期

ReAct 智能体其核心在于一个循环中运作，逐步处理信息。每个步骤通常包含生成一个思考、根据该思考提出一个动作、执行该动作，并纳入由此产生的观察结果。

> ReAct 智能体内部的迭代流程，展示了从提示大型语言模型到解析其输出、执行动作、观察结果，以及更新上下文 (context)以进行下一个周期的过程。

### 结构化思考与动作的提示设计

ReAct 的有效性很大程度上取决于如何设计提示，以引导大型语言模型按所需的 `Thought: ... Action: ...` 格式生成输出。这通常通过少样本提示实现，即提示中包含成功推理 (inference)路径的示例。

一个典型的 ReAct 提示结构可能如下所示：

```text
You are an expert assistant designed to solve complex tasks by reasoning step-by-step and interacting with available tools.

Available Tools:
[
  {
    "name": "Search",
    "description": "执行网络搜索以查找最新信息。",
    "arguments": {"query": "搜索查询字符串。"}
  },
  {
    "name": "Calculator",
    "description": "计算数学表达式。",
    "arguments": {"expression": "要评估的数学表达式。"}
  },
  # ... 可能会有更多工具
]

Follow this format strictly:

Question: 用户的输入问题。
Thought: 你对当前状态、目标以及下一步要采取的动作的推理。
Action: 要采取的动作。从可用工具列表中选择一个工具或使用 'Final Answer'。格式为 JSON 对象：{"action": "ToolName", "action_input": {"arg1": "value1", ...}} 或 {"action": "Final Answer", "action_input": "你的最终答案在这里。"}

--- 之前步骤 ---
Thought: 之前思考 1...
Action: {"action": "PreviousAction1", "action_input": {...}}
Observation: 之前观察结果 1...

Thought: 之前思考 2...
Action: {"action": "PreviousAction2", "action_input": {...}}
Observation: 之前观察结果 2...

--- 当前步骤 ---
Question: {current_user_question}
Thought:
```

**提示设计考量：**

- **指令清晰度：** 必须明确说明角色、可用工具和所需的输出格式。
- **工具描述：** 工具描述应简洁但信息充足，以便大型语言模型理解其用途和所需参数 (parameter)。在提示中使用 JSON 或 YAML 等结构化格式定义工具很常见。
- **少样本示例（可选但建议）：** 在初始提示中包含 1-3 个成功的 `思考 -> 动作 -> 观察结果` 序列示例，能显著提升大型语言模型遵循格式和推理模式的能力。
- **历史管理：** `--- 之前步骤 ---` 部分很重要。它提供了正在进行的推理过程的上下文 (context)。随着交互的进行，此历史记录会增加。管理上下文长度的策略，例如总结早期步骤或使用滑动窗口，对于长时间任务变得很重要（在第 3 章中会更详细地介绍）。
- **停止序列：** 为大型语言模型生成定义停止序列（例如 `Observation:`）有助于确保模型在生成 `动作` 后暂停，从而允许你的代码在提示下一步之前执行该动作。

### 解析大型语言模型输出

一旦大型语言模型生成其响应，就需要进行解析以提取 `思考` 和 `动作` 组件。通常使用正则表达式或字符串处理技术。对于格式为 JSON 的动作（如示例提示中所示），会使用 JSON 解析器。

```python
# 简化的类似 Python 的解析伪代码
llm_output = llm.generate(prompt) # 假设 llm_output 包含 "Thought: ... \nAction: ..."

thought = parse_thought(llm_output) # 提取 "Thought:" 之后和 "Action:" 之前的文本
action_json_str = parse_action_string(llm_output) # 提取 "Action:" 之后的文本

try:
    action_data = json.loads(action_json_str)
    action_name = action_data.get("action")
    action_input = action_data.get("action_input")
except json.JSONDecodeError:
    # 处理大型语言模型未生成有效 JSON 的情况
    handle_parsing_error()
    # 可能会要求大型语言模型重新格式化或尝试一个默认动作
```

解析过程中的错误处理很重要。大型语言模型有时可能无法严格遵守格式。你的实现需要处理格式不正确输出的策略，也许可以通过包含特定错误反馈的重新提示，或尝试一个默认的恢复动作。

### 动作执行与观察结果生成

解析后的 `action_name` 和 `action_input` 决定接下来会发生什么。

1. **识别工具：** 将 `action_name` 与系统中定义的可用工具/函数之一进行匹配。
2. **验证输入：** 检查 `action_input` 是否包含所选工具所需的参数 (parameter)。
3. **执行：** 使用提供的参数调用相应的函数或 API。
4. **捕获结果：** 工具执行的输出、成功/失败状态或错误消息将成为 `观察结果`。

```python
# 简化的类似 Python 的执行伪代码
available_tools = {
    "Search": search_function,
    "Calculator": calculator_function
}

if action_name == "Final Answer":
    # 任务已完成
    final_response = action_input
    mark_task_done()
elif action_name in available_tools:
    tool_function = available_tools[action_name]
    try:
        # 根据工具的预期参数验证 action_input
        validate_tool_input(action_name, action_input)
        observation = tool_function(**action_input) # 执行工具
    except Exception as e:
        # 将执行错误捕获为观察结果
        observation = f"Error executing {action_name}: {str(e)}"
else:
    # 处理大型语言模型虚构工具名称的情况
    observation = f"Error: Unknown action '{action_name}'. Please choose from available tools."

# 观察结果现在已准备好添加到历史中，以用于下一个提示
```

`观察结果` 应当信息丰富。如果搜索工具返回结果，观察结果可能是片段。如果计算器计算出一个值，该值就是观察结果。关键在于，如果工具失败，错误消息本身就会成为观察结果，使得智能体在下一个 `思考` 步骤中能够对失败进行推理 (inference)（例如，“思考：搜索失败因为查询格式不正确。我应该尝试重新措辞查询。”）。

### 循环管理与终止

核心循环通过将 `思考`、`动作` 和 `观察结果` 追加到提示的历史部分并生成下一步来继续。终止条件是必要的：

- **最终答案：** 主要的成功条件是当智能体输出一个类似 `{"action": "Final Answer", "action_input": "..."}` 的动作时。
- **最大迭代次数：** 设置步数限制以防止无限循环，特别是当智能体卡住时。
- **时间限制：** 为整个任务设置超时。
- **错误阈值：** 如果连续发生过多错误，则停止执行。

实现 ReAct 需要协调这些组件：精心设计的提示、解析、可靠的工具执行以及通过迭代过程进行的清晰状态管理。接下来的部分将审视其他架构，但这里看到的结构化提示、解析和动作执行的原则是许多智能体系统的重要组成部分。本章后面的实践环节将涉及构建一个包含这些元素的 ReAct 智能体。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍ReAct框架的原始研究论文，详细阐述了其概念设计以及在结合语言模型智能体的推理和行动方面的实际应用。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: arXiv preprint arXiv:2201.11903; DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  这篇基础性论文介绍了思维链提示技术，该技术指导LLM生成一系列中间推理步骤。此技术是ReAct框架中“思想”组件的直接前身和核心元素。

---

[上一节](01-ReAct%20%E6%A1%86%E6%9E%B6%EF%BC%9A%E6%8E%A8%E7%90%86%E4%B8%8E%E8%A1%8C%E5%8A%A8%E7%9A%84%E6%95%B4%E5%90%88.md) · [下一节](03-%E8%87%AA%E9%97%AE%EF%BC%9A%E9%80%9A%E8%BF%87%E8%BF%AD%E4%BB%A3%E6%8F%90%E9%97%AE%E6%8F%90%E5%8D%87%E4%BA%8B%E5%AE%9E%E5%87%86%E7%A1%AE%E6%80%A7.md)
