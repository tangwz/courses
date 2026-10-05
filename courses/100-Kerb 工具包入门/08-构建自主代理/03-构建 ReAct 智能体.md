# 构建 ReAct 智能体

来源：[原文](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents/building-a-react-agent)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建一个自主智能体可展现 ReAct 模式。`ReActAgent` 类为创建此类智能体提供了开箱即用的思考-行动-观察循环实现。

`ReActAgent` 协调管理整个过程。其核心是，它需要一个推理 (inference)引擎，即一个 LLM，来产生思考并决定行动。您将此引擎作为可调用函数提供。我们来看一下您将用来制作智能体的主要构成部分。

首先，您需要从 `patterns` 子模块导入 `ReActAgent` 类。

```python
from kerb.agent.patterns import ReActAgent
```

当您创建 `ReActAgent` 实例时，您会通过以下几个参数 (parameter)配置其行为：

- `name`: 一个字符串，用于标识您的智能体，这对于日志记录和调试很有用。
- `llm_func`: 一个 Python 函数或任何可调用对象，它接受一个字符串提示并返回 LLM 的字符串响应。此函数充当智能体的“大脑”，产生思考和决定行动。
- `tools`: 一个 `Tool` 对象列表，智能体可以用它们来执行行动。我们在此处简要提及，并在下一节中详细介绍。
- `max_iterations`: 一项必要的安全措施，它设定了智能体在停止前可以执行的思考-行动-观察循环的最大次数。这可以防止智能体陷入无限循环并消耗过多资源。

### 您的第一个智能体实现

我们从一个不使用任何外部工具的简单智能体开始。它唯一的“行动”是提供最终答案。这有助于阐明核心推理 (inference)循环。

首先，我们需要定义 `llm_func`，它将充当智能体的推理引擎。在此示例中，我们将使用一个模拟函数来模拟 LLM 的行为。在实际应用中，此函数将向 OpenAI 或 Anthropic 等供应商进行 API 调用。

LLM 的输出必须遵循 `ReActAgent` 可以解析的特定格式。两个主要组成部分是 `Thought`（思考）和 `Final Answer`（最终答案）。

- `Thought`: 这是智能体阐述其推理过程的地方。它应描述智能体的计划或思考。
- `Final Answer`: 这表明智能体已完成其任务并提供最终输出。

这是一个模拟 LLM 函数，展现了此结构。

```python
def simple_llm(prompt: str) -> str:
    """一个提供最终答案的模拟 LLM。"""
    if "weather" in prompt.lower():
        return "Thought: I need to provide weather information.\nFinal Answer: The weather is sunny."
    else:
        return "Thought: I will process this request.\nFinal Answer: I have processed your request."
```

定义好推理引擎后，我们现在可以实例化并运行智能体。我们将创建一个名为“BasicAgent”的智能体，传入我们的 `simple_llm` 函数，并设置 5 次迭代的限制。

```python
#
from kerb.agent.patterns import ReActAgent
from kerb.agent import AgentResult

# 上面定义的 simple_llm 函数
def simple_llm(prompt: str) -> str:
    if "weather" in prompt.lower():
        return "Thought: I need to provide weather information.\nFinal Answer: The weather is sunny."
    else:
        return "Thought: I will process this request.\nFinal Answer: I have processed your request."

# 创建 ReAct 智能体实例
agent = ReActAgent(
    name="BasicAgent",
    llm_func=simple_llm,
    max_iterations=5
)

# 定义目标并运行智能体
goal = "What is the weather like today?"
result = agent.run(goal)

# 显示最终输出
print(f"Goal: {goal}")
print(f"Final Answer: {result.output}")
```

`agent.run()` 方法会执行智能体的循环，直到它产生 `Final Answer` 或达到 `max_iterations` 上限。该方法返回一个 `AgentResult` 对象，其中包含最终输出、执行状态以及所有已执行步骤的详细列表。

### 观察智能体的推理 (inference)

最终输出很有用，但智能体的真正能力在于其逐步推理的能力。您可以通过检查 `AgentResult` 的 `steps` 属性来检查此过程。列表中的每个步骤都代表思考-行动-观察循环的一次循环。

让我们扩展我们之前的示例，打印出智能体的思考过程。

```python
# ... (来自上一个示例的智能体设置和运行) ...

print("\nEXECUTION STEPS:")
for i, step in enumerate(result.steps, 1):
    print(f"\nStep {i}:")
    if step.thought:
        print(f"  Thought: {step.thought}")
    if step.action:
        print(f"  Action: {step.action}")
    if step.observation:
        print(f"  Observation: {step.observation}")
```

对于我们的简单智能体，循环只运行一次。LLM 产生一个思考并立即提供最终答案，因此 `action` 和 `observation` 字段将为空。这展现了最简单形式的智能体行为：思考，然后得出结论。

### 制作结合工具的智能体

当智能体可以通过工具与外部环境交互时，它们会显著更强大。工具就是一个简单的函数，智能体可以决定调用它来收集信息或执行行动。

要让智能体使用工具，需要两点：

1. 在初始化期间，必须向智能体传入可用 `Tool` 对象的列表。
2. 当 `llm_func` 决定需要一个工具时，它必须被提示以 `Thought`、`Action` 和 `Action Input` 的格式生成输出。

- `Action`: 要使用的工具名称（例如，`calculate`）。
- `Action Input`: 传入工具函数的参数 (parameter)（例如，`15 * 7`）。

我们来创建一个新的模拟 LLM，它模拟决定使用一个工具。

```python
def mock_llm_react(prompt: str) -> str:
    """以 ReAct 格式响应并包含行动的模拟 LLM。"""
    # 第一轮：LLM 决定使用 'calculate' 工具
    if "calculate" in prompt.lower() or "15 * 7" in prompt.lower():
        return """Thought: I need to calculate 15 multiplied by 7. I will use the calculator tool.
Action: calculate
Action Input: 15 * 7"""

    # 第二轮：获得工具结果（观察）后
    elif "105" in prompt:
        return """Thought: I have the result of the calculation. Now I can provide the final answer.
Final Answer: The result of 15 * 7 is 105."""

    else:
        return "Thought: I am unsure how to proceed.\nFinal Answer: I cannot solve this problem."
```

接下来，我们定义工具本身。我们将创建一个简单的计算器函数并将其封装在一个 `Tool` 对象中。工具的 `name` 必须完全匹配 LLM 在 `Action` 字段中使用的名称。我们将在下一节中详细介绍工具创建。

```python
from kerb.agent import Tool

def calculate(expression: str) -> str:
    """一个简单的计算器工具。"""
    try:
        # 警告: eval() 可能不安全。请在生产环境中安全使用。
        return f"Result: {eval(expression)}"
    except Exception as e:
        return f"Error: {str(e)}"

calc_tool = Tool(
    name="calculate",
    description="Performs mathematical calculations",
    func=calculate
)
```

现在，我们使用该工具实例化智能体并运行它。

```python
# 创建带有计算器工具的 ReAct 智能体
agent_with_tool = ReActAgent(
    name="MathAgent",
    llm_func=mock_llm_react,
    tools=[calc_tool],
    max_iterations=5
)

# 运行需要该工具的智能体，设定目标
goal = "What is 15 * 7?"
result = agent_with_tool.run(goal)

# 显示完整的思考-行动-观察循环
print("REACT LOOP:")
for i, step in enumerate(result.steps, 1):
    print(f"\n[Step {i}]")
    if step.thought:
        print(f"  Thought: {step.thought}")
    if step.action:
        # 智能体框架自动调用工具
        print(f"  Action: {step.action}({step.action_input})")
    if step.observation:
        # 观察是工具的输出
        print(f"  Observation: {step.observation}")

print("\n--------------------")
print(f"Final Answer: {result.output}")
```

当您运行此代码时，您将看到一个多步骤过程：

1. **第一循环：** 智能体接收目标，调用 LLM，并获得 `Thought` 和 `Action` (`calculate`)。智能体框架找到 `calculate` 工具，用 `Action Input` (`15 * 7`) 执行它，并捕获其输出 (`Result: 105`)。此输出成为 `Observation`。
2. **第二循环：** 智能体获取第一循环的 `Observation` 并将其添加到历史记录。然后它再次调用 LLM，使用更新的上下文 (context)。这次，LLM 看到结果并产生 `Thought` 和 `Final Answer`。
3. **完成：** 智能体检测到 `Final Answer` 并停止循环，返回最终结果。

此示例阐明了完整的 ReAct 循环。通过为智能体提供推理 (inference)引擎和一组工具，您可以构建能够自主解决多步骤问题的系统。下一节将说明如何正确定义和管理各种工具，以扩展智能体的能力。

## 参考资料

- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2023)
  Publisher: OpenAI
  阐释了大型语言模型如何通过生成调用函数的结构化输出来与外部工具交互，这是代理的核心机制。
- [ReAct Prompting](https://proceedings.neurips.cc/paper_files/paper/2022/file/6069925f4a7c8c88686d1ee55b550247-Paper-Conference.pdf) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Jiang, Karthik Narasimhan, Zhicong Li (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS) 2022; Publisher: NeurIPS; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  详细介绍了ReAct提示技术，为如何构建提示以使大型语言模型展现推理和行动能力提供了实用指导。

---

[上一节](02-ReAct%20%E6%80%9D%E7%BB%B4%E4%B8%8E%E8%A1%8C%E5%8A%A8%E6%A8%A1%E5%BC%8F.md) · [下一节](04-%E5%AE%9A%E4%B9%89%E5%92%8C%E4%BD%BF%E7%94%A8%E5%B7%A5%E5%85%B7.md)
