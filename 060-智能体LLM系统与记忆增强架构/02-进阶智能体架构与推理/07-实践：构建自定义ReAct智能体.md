# 实践：构建自定义ReAct智能体

来源：[原文](https://apxml.com/zh/courses/agentic-llm-memory-architectures/chapter-2-advanced-agent-architectures-reasoning/practice-custom-react-agent)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建ReAct智能体需要理解其核心控制流程和交互模式。手动构建这个智能体，能够让您对操作机制、提示词 (prompt)设计的复杂性以及编排思-行-观察循环所面临的难点有全面的理解，这些内容通常在更高级的框架中被抽象化。

我们的目标不是复刻一个功能全面的库，而是实现ReAct的核心逻辑。我们假设您可以使用LLM API（如OpenAI的GPT模型、Anthropic的Claude或自托管模型），并熟练掌握Python以进行API调用和处理响应。

### 自定义ReAct智能体的核心构成部分

实现一个ReAct智能体需要几个相互连接的构成部分：

1. **LLM接口：** 一个函数或类，用于处理与您所选大型语言模型的通信。它接收一个格式化的提示字符串，并返回模型的文本生成结果。API调用的错误处理（如超时、速率限制）对于实现一个可用系统非常重要。
2. **工具集：** 智能体可使用的一系列工具。每种工具需要：
   - 一个独有的名称（例如，`search`，`calculator`）。
   - 清晰的描述，解释其功能、预期的输入格式和输出。该描述有助于LLM明确何时及如何使用该工具。
   - 实际的实现（例如，一个Python函数），用于执行工具的逻辑。
3. **提示模板：** ReAct智能体的核心。此模板规范LLM的输入，引导其推理 (inference)过程。它通常包含：
   - 最初的问题或任务。
   - 定义ReAct格式（思、行、观察）的指令。
   - 可用工具的描述。
   - 一个“暂存区”，智能体在此记录其先前步骤的思、行和观察序列。
   - 当前步骤思考过程的占位符。
4. **响应解析器：** 用于解析LLM输出字符串的逻辑。它需要可靠地识别和提取：
   - `Thought`：智能体的下一步推理。
   - `Action`：要使用的工具及其输入（例如，`Action: search[query: recent advancements in LLM agents]`）。
   - `Final Answer`：当智能体认为任务完成时的最终回复。处理LLM输出格式的变化和潜在的解析失败，需要周全的设计（例如，使用正则表达式，或者在LLM支持的情况下，请求结构化输出如JSON）。
5. **执行循环：** 主控制流程，迭代地编排ReAct过程：
   - 使用模板、当前问题、工具描述和累积的暂存区内容来格式化提示。
   - 将提示发送给LLM。
   - 解析LLM的响应。
   - 如果检测到`Final Answer`，则终止并返回答案。
   - 如果解析出`Action`：
     - 识别工具名称和输入。
     - 使用提供的输入执行相应的工具函数。处理潜在的执行错误。
     - 将工具的输出（或错误消息）格式化为`Observation`。
     - 将`Thought`、`Action`和`Observation`追加到暂存区。
     - 继续下一次迭代。
   - 设置终止条件（例如，最大步骤数）以防止无限循环。

### ReAct循环流程示例

该交互遵循一种明确的模式，通常可视化为一个循环。

> ReAct智能体通过格式化提示、调用LLM、解析响应、根据行动可能执行工具、将结果格式化为观察，并循环直到获得最终答案，从而在其暂存区中迭代地构建上下文 (context)。

### Python简化实现概述

我们来概述Python伪代码中的核心循环结构。这主要侧重于流程，而非具体的API或解析细节。

```python
import re # 用于基本解析示例

# 假设llm_call(prompt)存在并返回LLM文本响应
# 假设tools = { "tool_name": {"description": "...", "function": callable} } 存在

def execute_react_agent(question, tools, llm_call, max_steps=10):
    """ 执行ReAct智能体循环 """

    scratchpad = "" # 存储思-行-观察历史
    tool_descriptions = "\n".join([f"- {name}: {details['description']}" for name, details in tools.items()])

    for step in range(max_steps):
        # 1. 格式化提示
        prompt = f"""你是一个使用ReAct框架来回答问题的助手。
可用工具：
{tool_descriptions}

请使用以下格式：

思: 你的推理步骤。
行: 要采取的行动，应该是[{', '.join(tools.keys())}]中的一个或'Final Answer'。工具请使用Action: tool_name[input]格式。
观察: 行动的结果。
... (此思/行/观察循环重复)

问题: {question}
{scratchpad}思:""" # 提示LLM以思考开始

        # 2. LLM调用
        response = llm_call(prompt).strip()
        # 立即将LLM的思考过程追加到暂存区
        scratchpad += f"思: {response}\n"
        print(f"--- 步骤 {step+1} ---")
        print(f"思: {response}")

        # 3. 解析响应 (简化示例)
        action_match = re.search(r"Action: (.*?)(?:\[(.*?)\])?$", response, re.MULTILINE)
        final_answer_match = re.search(r"Final Answer: (.*)", response, re.MULTILINE | re.DOTALL)

        if final_answer_match:
            # 4a. 检测到最终答案
            final_answer = final_answer_match.group(1).strip()
            print(f"最终答案: {final_answer}")
            return final_answer

        if action_match:
            action_name = action_match.group(1).strip()
            action_input = action_match.group(2).strip() if action_match.group(2) else ""
            scratchpad += f"Action: {action_name}[{action_input}]\n"
            print(f"行: {action_name}[{action_input}]")

            if action_name in tools:
                # 5. 执行工具
                try:
                    tool_function = tools[action_name]["function"]
                    observation = tool_function(action_input)
                except Exception as e:
                    observation = f"执行工具 {action_name} 出错：{e}"

                # 6. 格式化观察并更新暂存区
                observation_str = str(observation) # 确保它是字符串
                scratchpad += f"观察: {observation_str}\n"
                print(f"观察: {observation_str}")
            else:
                scratchpad += "Observation: 指定了未知工具。\n"
                print("指定了未知工具。")
        else:
            # 处理LLM未输出有效Action或Final Answer的情况
            scratchpad += "Observation: 响应格式无效。停止。\n"
            print("响应格式无效。停止。")
            return "智能体因响应格式无效而失败。"

    return "智能体在达到最大步骤数后停止。"

# 示例用法（需要定义llm_call和tools）
# result = execute_react_agent("2加2是多少？", my_tools, my_llm_call)
# print(f"\n最终结果: {result}")
```

### 实现考量

- **提示工程 (prompt engineering)：** 您的提示模板的结构和措辞非常重要。关于所需格式（思、行、观察）的清晰指令、简洁的工具描述以及有效的示例（如果使用少量样本提示），都会显著影响性能。迭代优化通常是必需的。
- **解析可靠性：** LLM输出可能不一致。仅依赖简单的字符串分割或正则表达式可能不够稳定。如果可行，考虑让LLM输出结构化数据（如JSON），或实现更精密的带错误校正的解析逻辑。
- **工具输入/输出：** 确保您的函数预期的工具输入与LLM可能生成的内容相符。同样，将工具输出清晰地格式化，以便LLM在`Observation`步骤中理解。在工具内优雅地处理错误并在观察中报告它们，这一点很重要。
- **上下文 (context)管理：** 随着`暂存区`的增长，它会占用上下文窗口空间并增加API成本。对于长期运行的任务，可采用策略，例如对暂存区早期部分进行概括，或使用更高级的内存技术（第3章会介绍）。
- **终止条件：** 在检测到`Final Answer:`和达到最大步骤数后，考虑其他终止标准，例如重复行动、特定错误模式，或者如果您的解析器或LLM提供，则考虑置信度得分。

构建这个自定义智能体，即使是使用简化组件，也能阐明构建自主系统中的基本挑战和固有的设计选择。它在使用或扩展更复杂的智能体框架之前，提供了扎实的基础。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izzet Akman, Penporn Koanantakool, Chen Sun, Kai Wei, Jaehong Choe, Aakanksha Chowdary, Derek Chow, Jiantao Jiao, Parker Barnes, Quoc V. Le, Ed H. Chi, Jeff Dean, and Yuan Cao (2022)
  Journal: International Conference on Learning Representations (ICLR 2023); DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了ReAct框架，详细阐述了LLM智能体的思维-行动-观察循环。
- [Function calling and tool use](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI
  解释了如何通过描述函数和接收结构化输出来使LLM调用外部工具，这对于实现行动步骤很重要。
- [Prompt Engineering Guide](https://www.kaggle.com/whitepaper-prompt-engineering) — Lee Boonstra (2024)
  Publisher: Google; Pages: 68-69
  提供了构建有效提示的基本和高级技术，以指导LLM行为，包括复杂任务和工具使用的说明。
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (2022)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2022); DOI: [10.48550/arXiv.2201.11903](https://doi.org/10.48550/arXiv.2201.11903)
  介绍了思维链提示，这是一种通过展示中间思考来使LLM执行多步骤推理的技术，是ReAct框架的前身。

---

[上一节](06-%E6%8E%A8%E7%90%86%E6%9E%B6%E6%9E%84%E7%9A%84%E6%AF%94%E8%BE%83%E5%88%86%E6%9E%90.md) · [下一节](../03-%E8%AE%BE%E8%AE%A1%E5%A4%8D%E6%9D%82%E7%9A%84%E8%AE%B0%E5%BF%86%E7%B3%BB%E7%BB%9F/01-%E8%AE%B0%E5%BF%86%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E4%BD%9C%E7%94%A8.md)
