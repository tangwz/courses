# ReAct 思维与行动模式

来源：[原文](https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents/react-pattern-for-agents)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为构建能思考并与其环境互动的智能体，我们需要一种结构化的方式来连接大型语言模型（LLM）的“思考”过程与外部行动。ReAct 模式，即**推**理与**行**动（**Re**asoning and **Act**ing）的缩写，为此提供了一个有效的框架。它将思维、行动和观察的循环系统化，使智能体能够分解问题、执行任务，并依据结果来指导其后续步骤。

主要思路是促使大型语言模型不仅生成行动，还要生成其背后的原因。这个过程在一个循环中展开：大型语言模型首先生成关于如何达成目标的**思维**，然后是要采取的**行动**（比如使用工具），最后等待**观察**（该行动的结果）。这个观察结果被反馈回提示词 (prompt)中，用于下一个循环，使得智能体能够调整方向、收集更多信息，或者判断已有足够信息给出最终答案。

这个循环可以被可视化为一个持续的循环，重复执行直到智能体达成其目标：


$$
\text{目标} \rightarrow \text{思维} \rightarrow \text{行动} \rightarrow \text{观察} \rightarrow \dots
$$


`agent` 模块提供了一个 `ReActAgent` 类，负责组织协调整个过程。要使用它，我们需要两个主要组件：一个作为推理 (inference)引擎的大型语言模型，以及一组定义智能体可执行行动的工具。

### ReAct 循环的工作原理

我们来分析 ReAct 循环的每一步，以及它如何影响智能体的表现。ReAct 智能体中使用的大型语言模型被提示生成一种特定的文本格式，将智能体的内部独白与选定的行动区分开来。

1. **思维**：大型语言模型分析用户的目标以及之前的行动和观察历史。然后它制定下一步的计划。这在大型语言模型的输出中表示为“思维”，例如：`Thought: 我需要计算半径为 5 的圆的面积。我应该使用计算器工具。`
2. **行动**：基于其思维，大型语言模型选择一个工具并指定其输入。这表示为“行动”和“行动输入”，例如：`Action: calculator` 和 `Action Input: 3.14 * 5**2`。
3. **观察**：智能体框架接收行动，使用提供的输入执行相应的工具，并获取其输出。这个结果被格式化为“观察”，并在下一轮中反馈给大型语言模型。例如：`Observation: Result: 78.5`。
4. **重复**：大型语言模型接收到这个新的观察结果，并再次开始循环。它可能会生成一个新的思维，例如 `Thought: 我现在已经计算出面积了。我可以提供最终答案了。`，这会导致一个特殊的“最终答案”行动，从而结束循环。

这种结构使智能体的推理 (inference)过程透明且易于调试，您可以检查导致其最终输出的思维序列。

### 构建 ReAct 智能体

`kerb.agent.patterns` 中的 `ReActAgent` 类是该模式的一个即用型实现。它需要一个调用大型语言模型的函数和智能体可以使用的工具列表。

为了演示，我们将从一个模拟大型语言模型函数开始，它模拟真实大型语言模型会产生的结构化输出。这有助于说明智能体的工作机制，而无需配置实时 API 连接。

```python
from kerb.agent.patterns import ReActAgent
from kerb.agent import Tool

def mock_llm_react(prompt: str) -> str:
    """一个以 ReAct 格式响应的模拟大型语言模型。"""
    # 这模拟了大型语言模型进行计算的推理过程。
    if "calculate" in prompt.lower() or "15 * 7" in prompt.lower():
        return """Thought: The user wants to multiply 15 by 7. I should use the calculator tool for this.
Action: calculate
Action Input: 15 * 7"""
    # 这模拟了大型语言模型收到计算结果后的响应。
    elif "105" in prompt:
        return """Thought: I have the result of the calculation. I can now provide the final answer to the user.
Final Answer: The result of 15 * 7 is 105."""
    else:
        return "Thought: I am not sure what to do.\nFinal Answer: I cannot solve this problem."

def calculate(expression: str) -> str:
    """一个简单的计算器工具，用于评估数学表达式。"""
    try:
        # 我们为这个简单的数学例子使用安全的 eval 函数。
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# 定义智能体的工具
calc_tool = Tool(
    name="calculate",
    description="A calculator for mathematical expressions. Use this for any math-related questions.",
    func=calculate,
    parameters={"expression": {"type": "string", "description": "A valid mathematical expression"}}
)

# 初始化智能体
agent = ReActAgent(
    name="MathAgent",
    llm_func=mock_llm_react,
    tools=[calc_tool],
    max_iterations=5,
    verbose=True # 设置为 True 以观察循环的执行
)
```

在此设置中，`Tool` 对象作用很大。`description` 字段不仅用于文档说明；它还是大型语言模型决定哪个工具适用于特定任务的依据。一个清晰且有描述性的 `description` 是智能体正常运行的必要条件。

### 运行智能体并观察循环

配置好智能体后，我们可以使用 `run` 方法给它一个目标。智能体随后会执行 ReAct 循环，直到它达到最终答案或达到其迭代限制。

```python
# 智能体的目标
goal = "What is 15 * 7?"
result = agent.run(goal)

# 检查最终输出和所采取的步骤
print("\nFINAL RESULT:")
print(result.output)

print("\nEXECUTION TRACE:")
for i, step in enumerate(result.steps, 1):
    print(f"\n--- Step {i} ---")
    if step.thought:
        print(f"Thought: {step.thought}")
    if step.action:
        print(f"Action: {step.action}({step.action_input})")
    if step.observation:
        print(f"Observation: {step.observation}")
```

运行这段代码将生成一个执行追踪，清晰地显示智能体循序渐进的过程。

**第一次循环迭代：**

- 智能体接收到目标“15 \* 7 是多少？”。
- `llm_func` 被调用，并传入包含此目标的提示词 (prompt)。
- 大型语言模型（我们的模拟模型）返回 `Thought: ... Action: calculate ... Action Input: 15 * 7`。
- `ReActAgent` 解析此内容，识别出 `calculate` 工具，并以 `"15 * 7"` 执行它。
- `calculate` 函数返回 `"Result: 105"`。这成为观察结果。

**第二次循环迭代：**

- `llm_func` 再次被调用，但这次提示词中包含之前的思维、行动和新的观察结果：`"Observation: Result: 105"`。
- 大型语言模型看到结果并返回 `Thought: ... Final Answer: The result of 15 * 7 is 105.`。
- 智能体识别出 `Final Answer` 关键词，停止循环，并设置最终输出。

下图显示了 ReAct 智能体中的信息流。智能体的大型语言模型核心不断循环进行推理 (inference)和行动，直到得出结论。

> ReAct 模式结合了大型语言模型的推理能力与外部工具的实际执行能力。

通过以这种方式组织智能体行动，ReAct 创建了一个更强大、更易于理解的系统。您可以追溯其“思维链”来了解它是如何得出结论的，这使其成为构建可靠且可审计的自主智能体的优秀模式。

## 参考资料

- [Agents. LangChain Documentation.](https://python.langchain.com/v0.2/docs/concepts/#agents) — LangChain (2024)
  Publisher: LangChain
  流行框架的官方文档，提供了代理模式（包括ReAct）的实际实现，展示了如何将LLM与工具集成以创建自主代理。

---

[上一节](01-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E6%99%BA%E8%83%BD%E4%BD%93%E7%AE%80%E4%BB%8B.md) · [下一节](03-%E6%9E%84%E5%BB%BA%20ReAct%20%E6%99%BA%E8%83%BD%E4%BD%93.md)
