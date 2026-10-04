---
course: "getting-started-with-llm-toolkit"
chapter: "developing-autonomous-agents"
lesson: "defining-and-using-tools"
sourceId: 7785
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-8-developing-autonomous-agents/defining-and-using-tools"
title: "定义和使用工具"
description: "为你的智能体配备工具，例如搜索引擎或计算器，以便与外部环境交互。"
order: 4
plots: []
sourceHash: "4f64cbe2b48b9c8dc4cfe89d66a9f9b8f5027c4e72bd95014ac72aa7fcd6687e"
sourceCorrections: []
---

尽管大型语言模型（LLM）是一个强大的推理 (inference)引擎，但它也存在明显的局限性。LLM本身是一个封闭系统；它无法从网络获取实时信息、进行精确的数学运算、执行代码或与外部应用程序接口（API）交互。为克服这些局限并构建真正有能力的智能体，智能体都配备了**工具**。

工具是一个函数，智能体可以决定调用它来执行特定动作。这可以是一个简单的计算器，也可以是与公司内部数据库的复杂整合。通过为智能体提供一套工具，你就赋予了它与外部环境交互、收集信息以及执行动作以达成目标的能力。LLM的角色从试图*知道*答案转变为*知道如何找到*答案。

### 工具的构成

在 Kerb 工具包中，工具不只是一段 Python 函数。它是一个结构化对象，为大型语言模型提供所有必要信息，使其能有效理解并运用。每个工具都包含几个主要构成部分：

- **名称：** 工具的独特、描述性名称（例如，`calculator`、`web_search`）。大型语言模型使用此名称来指明它希望使用的工具。
- **说明：** 清晰、自然的语言说明，描述该工具的作用、何时使用以及其输入和输出。这可以说是最主要的组成部分，因为大型语言模型很大程度上依赖于此说明来做出推理 (inference)判断。
- **函数：** 当智能体决定使用工具时，实际会运行的 Python 函数。
- **参数 (parameter)：** 一个定义函数接受的参数的模式，包括它们的名称、类型和说明。这使大型语言模型能够正确构建其`动作输入`（Action Input）。

### 创建你的第一个工具

定义工具主要有两种方式：直接使用 `Tool` 类，或使用 `create_tool` 辅助函数。我们先从 `Tool` 类开始，构建一个简单的计算器。

首先，定义将执行工作的 Python 函数：

```python
def calculate(expression: str) -> str:
    """一个简单的计算器工具。"""
    try:
        # 为简化示例使用 eval()，但在实际应用中应采用更安全的方法
        result = eval(expression)
        return f"结果: {result}"
    except Exception as e:
        return f"错误: {str(e)}"
```

接下来，将此函数包装进一个 `Tool` 对象中。这为智能体的 LLM 提供了使用该工具所需的元数据。

```python
from kerb.agent import Tool

calc_tool = Tool(
    name="calculator",
    description="对给定的表达式执行数学计算。对于任何与数学相关的问题，请使用此工具。",
    func=calculate,
    parameters={
        "expression": {
            "type": "string",
            "description": "要评估的数学表达式（例如，'15 * 7'）。"
        }
    }
)
```

请注意 `description` 是如何为大型语言模型编写的。它清楚地说明了工具的用途（“执行数学计算”）并提供了何时使用它的指引（“处理任何数学相关问题时使用”）。`parameters` 字典定义了 `calculate` 函数期望的输入 `expression`。

### 使用 `create_tool` 辅助函数

为了更简洁地定义工具，你可以使用 `create_tool` 函数。它以略微简单的语法实现了相同的结果。让我们定义另一个工具，这次用于模拟网页查询。

```python
from kerb.agent import create_tool

def search(query: str) -> str:
    """一个模拟查询工具。"""
    # 在实际应用中，这里会调用一个查询 API
    if "python" in query.lower():
        return "Python 是一种高级编程语言。"
    return f"未找到 '{query}' 的结果"

search_tool = create_tool(
    name="search",
    description="查询给定主题的信息。使用此工具查找最新信息。",
    func=search,
    parameters={
        "query": {
            "type": "string",
            "description": "要查询的主题或问题。"
        }
    }
)
```

`Tool(...)` 和 `create_tool(...)` 都生成可传递给智能体的同类型对象。选择最符合你编程风格的方式。

### 为智能体配备工具

一旦你定义好工具，就需要让智能体了解它们。你可以通过将 `Tool` 对象列表传递给智能体的构造函数来完成此操作。

```python
from kerb.agent.patterns import ReActAgent

# 一个用于演示的模拟大型语言模型函数
def mock_llm_react(prompt: str) -> str:
    if "calculate 15 * 7" in prompt:
        return """思考：我需要计算 15 乘以 7。我应该使用计算器工具。
动作：计算器
动作输入：15 * 7"""
    elif "105" in prompt:
        return """思考：我已经得到计算结果。现在我可以提供最终答案了。
最终答案：15 * 7 的结果是 105。"""
    else:
        return """思考：我将对此进行思考。 \n最终答案：我不确定该做什么。"""

# 创建一个智能体，并为其提供我们定义的工具
agent = ReActAgent(
    name="MathAgent",
    llm_func=mock_llm_react,
    tools=[calc_tool, search_tool],
    max_iterations=5
)

print(f"已创建带工具的智能体：{[t.name for t in agent.tools]}")
```

`ReActAgent` 初始化时，它会将所提供工具的名称和说明整合到其内部系统提示中。这使得大型语言模型在每个推理 (inference)步骤中都能看到可用的工具，并判断其中是否有工具能帮助达成用户的目标。

### 工具使用循环的运行过程

智能体配备工具后，让我们追踪一个目标（例如“15 \* 7 是多少？”）的完整执行循环。

```python
goal = "15 * 7 是多少？"
result = agent.run(goal)

# 显示 ReAct 循环
for i, step in enumerate(result.steps, 1):
    print(f"\n[步骤 {i}]")
    if step.thought:
        print(f"思考: {step.thought}")
    if step.action:
        # 智能体框架将“calculator”解析为动作
        print(f"动作: {step.action}({step.action_input})")
    if step.observation:
        # 观察结果是 calculate() 函数的输出
        print(f"观察: {step.observation}")

print(f"\n最终答案: {result.output}")
```

此交互过程如下展开：

1. **目标：** 智能体收到目标：“15 \* 7 是多少？”。
2. **思考：** `llm_func` 收到包含目标以及 `calculator` 和 `search` 工具定义的提示。大型语言模型判断 `calculator` 工具合适，并制定计划。它输出其思考：“我需要计算 15 乘以 7。我应该使用计算器工具。”
3. **动作：** 大型语言模型随后输出一个结构化动作命令：`动作：计算器`，其 `动作输入` 为 `15 * 7`。
4. **执行：** 智能体框架解析此输出。它确定 `calculator` 为要运行的工具，`"15 * 7"` 为输入。然后它调用我们的 Python 函数：`calculate("15 * 7")`。
5. **观察：** `calculate` 函数执行 `eval("15 * 7")`，返回 `105`。智能体将此返回值作为该步骤的观察结果。
6. **下一次循环：** 观察结果（“结果：105”）在下一个提示中反馈给大型语言模型。大型语言模型看到结果并构思其下一步思考：“我已经得到计算结果。”
7. **最终答案：** 收集到必要信息后，大型语言模型提供最终答案：“15 \* 7 的结果是 105。”智能体看到“最终答案”标签并终止循环，返回结果。

这种思考、动作和观察的循环是基本机制，使得智能体能够将其能力远远超过大型语言模型本身的固有知识。

### 工具设计的良好实践

你设计工具的方式对智能体的表现有显著影响。

- **撰写描述性说明：** 工具的 `description` 是你与大型语言模型沟通的主要途径。撰写清晰、明确的说明，解释工具的作用、适用场景以及任何主要限制。例如，对于一个 `search` 工具，一个好的说明可以是“查询公共网站上的实时信息。最适合用于近期事件、新闻和一般知识问题。”
- **倾向于简单、原子化工具：** 最好拥有多个简单、单一用途的工具，而不是一个做多项事务的复杂工具。智能体能更容易地判断如何组合 `search_web` 和 `read_file`，而不是使用一个单一的 `research_and_summarize_topic` 工具。
- **定义清晰的参数 (parameter)：** 为你的参数使用描述性名称，并为每个参数提供清晰的说明。如果参数有特定限制（例如，可能值的枚举），请在说明中包含这些信息。
- **优雅地处理错误：** 你的底层 Python 函数应预见并处理潜在错误。如果工具失败（例如，API 调用超时或收到无效输入），它应返回一条提供信息的错误消息。此消息成为智能体的观察结果，为其提供了恢复、重试或尝试不同方案的机会。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，结合了语言模型中的推理和行动，是工具使用型代理的基础。
- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI Developer Documentation
  关于如何将外部工具和函数与OpenAI大型语言模型集成的官方指南，提供了实用的设计原则。
- [A Survey on Large Language Model Based Autonomous Agents](https://arxiv.org/abs/2308.11432) — Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, Ji-Rong Wen (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2308.11432](https://doi.org/10.48550/arXiv.2308.11432)
  全面综述了基于大型语言模型的自主代理的架构、组件和挑战，包括工具集成。
- [Agents](https://python.langchain.com/docs/concepts/#agents) — LangChain (2024)
  Publisher: LangChain Documentation
  解释了LangChain框架中代理的概念和实现，包括如何有效定义和使用工具。
