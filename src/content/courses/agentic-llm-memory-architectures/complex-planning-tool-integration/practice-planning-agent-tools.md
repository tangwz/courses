---
course: "agentic-llm-memory-architectures"
chapter: "complex-planning-tool-integration"
lesson: "practice-planning-agent-tools"
sourceId: 4503
sourceUrl: "https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-4-complex-planning-tool-integration/practice-planning-agent-tools"
title: "实践：搭建一个使用工具的多步骤规划智能体"
description: "构建一个能够进行复杂规划并使用多个外部工具的智能体。"
order: 7
plots: []
sourceHash: "e6ed29b7bce824de117f6f437a23e6717111072a3df5bd3ec6da74d5205ea1d2"
sourceCorrections: []
---

实践练习涵盖多步骤规划和工具整合。您将搭建一个基础智能体，使其能够分解任务，使用外部工具（一个简单的搜索功能和一个计算器），并按顺序执行动作以完成一个任务。本练习假定您熟悉 Python 编程以及与 LLM API 的交互。

我们的目标是构建一个能够回答需要信息检索和计算的问题的智能体，例如“1992年和2000年夏季奥运会主办城市之间的人口大致差异是多少？”

### 1. 定义工具

有效使用工具始于 LLM 能够理解的清晰定义。每个工具都需要一个名称、一个说明其用途和使用时机的描述，以及预期的输入格式。

让我们用 Python 定义两个简单的工具：

```python
import re

# 一个真实搜索API的占位符
def simple_search(query: str) -> str:
    """
    一个简单的搜索工具。
    使用此工具查找特定实体、事件或事实的信息。
    输入应为一个简洁的搜索查询字符串。
    查询示例：“日本首都”，“1992年夏季奥运会主办城市”
    """
    print(f"执行搜索：{query}")
    # 在实际场景中，这将调用一个搜索API（例如，谷歌搜索，必应）。
    # 本例中我们将使用硬编码的响应。
    query = query.lower()
    if "1992 summer olympics host city" in query:
        return "Barcelona"
    elif "2000 summer olympics host city" in query:
        return "Sydney"
    elif "population of barcelona" in query:
        return "Approximately 1.6 million"
    elif "population of sydney" in query:
        return "Approximately 5.3 million"
    else:
        return "未找到信息。"

def simple_calculator(expression: str) -> str:
    """
    一个简单的计算器工具。
    使用此工具执行算术计算。
    输入必须是有效的数学表达式字符串（例如，'5.3 - 1.6'）。
    它处理加法（+）、减法（-）、乘法（*）和除法（/）。
    """
    print(f"执行计算：{expression}")
    try:
        # 基本安全：只允许数字、运算符和空格。
        if not re.match(r"^[0-9\.\s\+\-\*\/\(\)]+$", expression):
            return "错误：表达式中包含无效字符。"

        # 评估表达式。注意：此处使用 eval() 是为了简化，
        # 但可能不安全。在生产环境中请使用更安全的数学表达式解析器。
        result = eval(expression)
        return f"{result:.2f}" # 格式化为两位小数
    except Exception as e:
        return f"错误：计算失败。{str(e)}"

# 将工具存储在字典中以便于查找
tools = {
    "Search": simple_search,
    "Calculator": simple_calculator
}

# 为LLM提示生成工具描述
tool_descriptions = ""
for name, func in tools.items():
    tool_descriptions += f"- {name}: {func.__doc__.strip()}\n"

print("用于提示的工具描述：\n", tool_descriptions)
```

`tool_descriptions` 字符串非常重要。它将作为提示的一部分，告知 LLM 可用的功能。

### 2. 设计智能体循环

我们将实现 ReAct（推理 (inference) + 行动）模式的一个变体。智能体将以循环方式运作，对下一步进行推理，选择一个行动（要么使用工具，要么给出最终答案），并观察结果。

核心逻辑如下：

1. **初始化：** 从用户的目标和空的历史记录开始。
2. **推理/规划：** 将目标和历史记录发送给 LLM。提示它逐步思考，决定下一步行动，或提供最终答案。提示中必须包含工具描述。
3. **解析行动：** 从 LLM 的响应中提取预期的行动。这可能是：
   - `Action: ToolName[Input]`（例如，`Action: Search[1992年夏季奥运会主办城市]`）
   - `Final Answer: [Answer]`
4. **执行行动：**
   - 如果是工具行动，验证工具名称和输入。执行对应的函数。
   - 如果是最终答案，则终止循环。
5. **观察：** 记录行动结果（工具输出或错误消息）。
6. **更新历史记录：** 将推理、行动和观察添加到历史记录中。
7. **重复：** 返回步骤2，除非给出了最终答案或达到步数限制。

### 3. 实现智能体

让我们勾勒出这个智能体循环的 Python 代码。我们需要一个与 LLM 交互的函数（此处由占位符 `call_llm` 表示）。

```python
# 您的LLM API调用函数的占位符
# 假设它接收一个提示字符串并返回LLM的文本响应。
def call_llm(prompt: str) -> str:
    # 替换为实际的OpenAI、Anthropic、Gemini等API调用
    print("\n--- 正在向LLM发送提示 ---")
    # 如果提示过长，则截断以显示
    print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
    print("--- 提示结束 ---")

    # 用于示例任务的模拟LLM行为的虚拟响应
    # 这个序列模拟了推理和行动的产生
    if "Initial Goal" in prompt:
      response = """
      思考：用户想知道1992年和2000年奥运会主办城市之间的人口差异。
      我需要找到1992年的主办城市，然后找到它的人口。
      然后找到2000年的主办城市，然后找到它的人口。
      最后，计算差异。
      步骤1：找到1992年的主办城市。
      Action: Search[1992 Summer Olympics host city]
      """
    elif "Observation: Barcelona" in prompt:
      response = """
      思考：好的，1992年的主办城市是巴塞罗那。现在我需要它的人口。
      Action: Search[population of Barcelona]
      """
    elif "Observation: Approximately 1.6 million" in prompt:
      response = """
      思考：巴塞罗那的人口大约是160万。现在找到2000年的主办城市。
      Action: Search[2000 Summer Olympics host city]
      """
    elif "Observation: Sydney" in prompt:
      response = """
      思考：2000年的主办城市是悉尼。现在我需要它的人口。
      Action: Search[population of Sydney]
      """
    elif "Observation: Approximately 5.3 million" in prompt:
      response = """
      思考：悉尼的人口大约是530万。现在我有了两个城市的人口（悉尼530万，巴塞罗那160万）。我需要计算差异。
      Action: Calculator[5.3 - 1.6]
      """
    elif "Observation: 3.70" in prompt:
       response = """
       思考：计算器返回了3.70。这表示370万。我有了最终答案。
       最终答案：悉尼（2000年主办城市，约530万）和巴塞罗那（1992年主办城市，约160万）之间的人口大致差异是370万。
       """
    else:
        response = "最终答案：遇到了意外情况。"

    print("\n--- LLM 响应 ---")
    print(response)
    print("--- 响应结束 ---")
    return response.strip()

def parse_llm_output(response: str) -> tuple[str, str, str]:
    """解析LLM响应以查找思考、行动和最终答案。"""
    thought_match = re.search(r"Thought:(.*)", response, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*(\w+)\s*\[(.*)\]", response, re.DOTALL | re.IGNORECASE)
    final_answer_match = re.search(r"Final Answer:(.*)", response, re.DOTALL | re.IGNORECASE)

    thought = thought_match.group(1).strip() if thought_match else ""

    if action_match:
        tool_name = action_match.group(1).strip()
        tool_input = action_match.group(2).strip()
        return thought, tool_name, tool_input
    elif final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        # 通过对工具名称/输入返回None来指示最终答案
        return thought, "Final Answer", final_answer 
    else:
        # 如果没有具体的行动或最终答案，则假定它是推理的一部分或意外情况
        return thought, "No Action", "" 

def run_agent(initial_goal: str, max_steps: int = 10):
    """运行多步骤规划智能体。"""
    history = f"初始目标：{initial_goal}\n"

    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        prompt = f"""
您是一名专业助手，旨在通过规划步骤和使用可用工具来回答问题。
逐步思考以分解目标。
您可以使用以下工具：
{tool_descriptions}
请使用以下格式：
思考：[您的推理过程]
Action: [ToolName][Input]

或者，如果您有最终答案：
思考：[您的推理过程]
最终答案：[最终答案]

当前目标：{initial_goal}

对话历史：
{history}

轮到您了：
"""

        llm_response = call_llm(prompt)
        thought, tool_name, tool_input_or_answer = parse_llm_output(llm_response)

        history += f"Thought: {thought}\n"

        if tool_name == "Final Answer":
            final_answer = tool_input_or_answer
            print(f"\n收到最终答案：{final_answer}")
            history += f"Final Answer: {final_answer}\n"
            return final_answer, history

        elif tool_name == "No Action":
             print("智能体判断无需行动或输出格式不符合预期。")
             # 可能会添加备用逻辑或请求澄清
             history += "Observation: 智能体提供了推理但没有具体的行动或最终答案。\n"
             # 对于本例，如果卡住我们就停止
             return "智能体停止：没有明确的下一步行动。", history

        elif tool_name in tools:
            print(f"行动：使用工具 '{tool_name}'，输入为 '{tool_input_or_answer}'")
            history += f"Action: {tool_name}[{tool_input_or_answer}]\n"

            try:
                tool_function = tools[tool_name]
                observation = tool_function(tool_input_or_answer)
                print(f"观察：{observation}")
                history += f"观察：{observation}\n"
            except Exception as e:
                print(f"执行工具 {tool_name} 时出错：{e}")
                history += f"观察：执行工具 {tool_name} 时出错：{str(e)}\n"
        else:
            print(f"错误：请求了未知工具 '{tool_name}'。")
            history += f"观察：尝试使用未知工具 '{tool_name}'。\n"

        if step == max_steps - 1:
            print("已达到最大步数。")
            return "智能体停止：达到最大步数。", history

    return "智能体意外停止。", history

# --- 运行示例 ---
goal = "1992年和2000年夏季奥运会主办城市之间的人口大致差异是多少？"
final_answer, execution_history = run_agent(goal)

print("\n--- 执行历史 ---")
print(execution_history)
```

### 4. 示例追踪与可视化

使用示例目标运行代码会产生一系列交互。智能体首先使用`搜索`工具查找主办城市（巴塞罗那、悉尼），然后再次使用`搜索`来获取其人口，最后使用`计算器`工具来查找差异。

我们可以可视化规划的执行流程：

> 此图描绘了智能体为完成请求而规划的工具使用顺序。每个方框表示一个行动，通常涉及一次工具调用，最终导向最终答案。

### 5. 错误处理与自我纠正

我们简单的实现包含了基本的错误报告（例如，“未找到信息”，“错误：计算失败”）。在一个更复杂的智能体中，这些错误观察结果会在下一步被反馈到 LLM 的上下文 (context)中。

如果 `Search[巴塞罗那人口]` 失败。历史记录将包含 `观察：未找到信息。`。LLM 看到此情况，理想情况下应调整其计划。它可能：

1. **重试：** 尝试稍微不同的查询，例如，`Search[巴塞罗那城市人口]`。
2. **寻求替代方案：** 如果有可用方式，尝试通过其他途径查找信息。
3. **报告失败：** 得出结论无法找到所需信息，并在最终答案中报告此情况。

实现自我纠正需要仔细的提示工程 (prompt engineering)，可能需要向 LLM 提供明确的指令，说明如何处理错误或模糊的工具输出。像反思（智能体根据观察结果评判自己的计划或输出）这样的方法也可以融入。

### 6. 进一步发展

本示例提供了一个基础结构。专家级系统通常会纳入更高级的方法：

- **分层规划：** 将主要目标分解为子目标，每个子目标可能需要自己的子计划和工具使用。这对于非常复杂的任务来说是必不可少的。
- **动态工具选择：** 不仅使用 LLM 格式化工具输入，而且根据当前的子任务从更大的工具集中*选择*最合适的工具。
- **复杂的错误处理：** 实现带回退的重试逻辑、解析特定的错误代码，甚至调用调试工具或子智能体。
- **状态管理：** 明确管理智能体的内部状态，可能使用第3章讨论的结构化格式或内存组件。
- **结构化输出：** 使用支持受限解码或函数调用的 LLM 来获取更可靠的 JSON 工具行动输出，从而减少解析错误。

“LangChain、LlamaIndex 或 AutoGen 等框架为构建此类智能体、管理提示、工具定义、解析和执行循环提供了更高级别的抽象。然而，理解其背后的原理（如本例所示的实践）对于调试、优化和为复杂应用定制智能体行为很重要。”

## 参考资料

- [Function calling](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  官方文档，详细说明如何提示LLM调用外部函数，作为工具集成的实用方法。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  探讨了带有记忆和自我反思的LLM驱动代理的创建，为代理系统提供了更广阔的视角。
