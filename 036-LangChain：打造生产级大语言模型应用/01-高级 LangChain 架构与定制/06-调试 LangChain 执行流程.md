# 调试 LangChain 执行流程

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-1-advanced-langchain-architecture/debugging-execution-flow)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当你利用 LangChain 的高级功能（例如 LangChain 表达式语言 (LCEL)、异步操作和自定义组件）构建更精巧的应用时，明确执行流程变得愈发重要。让 LangChain 功能强大的分层抽象有时会掩盖链或代理运行期间发生的具体情况。因此，高效调试必不可少，它不仅用于修正错误，更用于真正弄懂并优化应用的表现。本文将介绍专门用于追踪和分析 LangChain 执行流程的方法和工具。

### 调试 LLM 应用的挑战

与传统软件相比，用 LangChain 构建的应用在调试时面临独有的挑战：

1. **抽象层：** 链和代理包含多个步骤，常常调用 LLM、检索器、解析器和工具。要找出问题在此序列何处出现，需能看清每一步的情况。
2. **中间数据：** 组件之间传递的数据（发送给 LLM 的提示词 (prompt)、LLM 的原始输出、解析结果、检索到的文档）常被隐藏。检查这些中间数据对于问题判断非常要紧。
3. **LLM 的非确定性：** LLM 的输出即使输入相同也可能不同（因温度设置或模型更新），使重现问题变得更难。调试通常注重确保 LLM 的*输入*及其输出的*解析*都正确。
4. **复杂的控制流程：** 特别是代理，其执行路径依据 LLM 的决策而动态变化，使得简单的线性调试很不容易。

### LangChain 内置调试辅助工具

LangChain 包含内置机制，可在执行期间提高可见度。

#### 全局详细设置

获取更多情况最简单的办法是通过 globals 模块启用全局详细设置：

```python
from langchain.globals import set_verbose
set_verbose(True)

# 现在，当你运行链或代理时，详细日志会打印到标准输出
# 例如，显示发送给 LLM 的准确提示、LLM 的原始响应等。
```

虽然易于启用，但详细日志会产生大量输出，特别是对于涉及多次 LLM 调用或工具使用的精巧链或代理。它对初步察看有益，但对于有目标的调试可能会变得难以应对。

#### 全局调试设置

一个更规整的替代方案是全局调试设置：

```python
from langchain.globals import set_debug
set_debug(True)

# 运行组件现在会打印出更规整的调试信息，
# 通常包含时间信息和更清楚的步骤划分。
```

与详细模式相比，将调试设置为 true 通常会提供更清晰、可能带颜色编码的输出，使其稍微更容易追踪执行路径。然而，与详细设置一样，它全局生效，仍可能生成过多信息。

### LangSmith：可观测性平台

对于认真的开发和生产监控，LangSmith 是推荐工具。虽然第 5 章详细介绍了 LangSmith 用于评估和监控，但其追踪功能在开发期间进行调试时是不可或缺的。

要使用 LangSmith，你通常需要设置环境变量：

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="..."
# 可选：设置项目名称
# export LANGCHAIN_PROJECT="我的高级应用"
```

配置完成后，LangChain 自动将执行追踪记录到 LangSmith。网页界面允许你进行如下操作：

- **可视化运行情况：** 查看链或代理的完整执行图。
- **检查输入/输出：** 点击任何节点（LLM 调用、工具执行、检索器查询）以查看其接收到的确切输入和产生的输出。
- **分析延迟：** 找出哪些步骤耗时最多。
- **筛选和搜索：** 根据输入、输出或错误快速找到特定运行。

LangSmith 将调试从解读原始日志转变为交互式查看规整的追踪信息，大大加快了找出根本原因的过程。

### 使用回调函数获取定制化信息

为了精细控制和程序化访问执行事件，LangChain 的回调系统功能强大。回调函数允许你介入 LangChain 生命周期中的各个阶段（例如，`on_llm_start`、`on_chain_end`、`on_agent_action`）。

你通过继承 `BaseCallbackHandler` 并覆盖你想要拦截事件对应的方法来实行自定义回调处理器。

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.agents import AgentAction
from typing import Any, Dict, List

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        print(f"LLM Start: Sending {len(prompts)} prompts.")
        # 将提示记录到自定义位置，也许只在它们符合特定条件时才记录

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print("LLM End:")
        # 记录 token 使用量或成本估算
        if response.llm_output:
             print(f"  Tokens Used: {response.llm_output.get('token_usage')}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        print(f"Chain Start: Input keys: {list(inputs.keys())}")

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        print(f"Agent Action: Tool={action.tool}, Input={action.tool_input}")
        # 在此处添加工具输入的验证逻辑

# --- 用法 ---
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate

# llm = ChatOpenAI(temperature=0) # 假设 OPENAI_API_KEY 已设置
# prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
# chain = prompt | llm

# handler = MyCustomHandler()
# # 回调函数通过 LCEL 中的 config 字典传递
# result = chain.invoke({"topic": "debugging"}, config={"callbacks": [handler]})
# print(f"\n最终结果:\n{result.content}")
```

回调函数提供精准控制。你可以在恰好需要的时候和地方记录特定数据片段、触发外部警报、实行自定义验证或收集精细的性能指标，而不依赖全局设置或外部平台。LangChain 也提供 `StdOutCallbackHandler` 等标准处理器，它模仿了部分全局详细行为，但可以有选择地应用。

### 可视化 LCEL 结构

使用 LangChain 表达式语言 (LCEL) 构建精巧链时，了解其结构本身就可以是调试的一个步骤。LCEL 对象有帮助可视化其组成的办法：

````python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 定义一个简单的 LCEL 链
prompt = ChatPromptTemplate.from_template("Explain {topic} simply.")
model = ChatOpenAI() # 假设 OPENAI_API_KEY 已设置
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# 获取图的可打印表示
# print(chain.get_graph().draw_ascii())

# 或者生成一个 Graphviz 图（需要安装 Graphviz 库）
try:
    graph_viz_object = chain.get_graph().draw_graphviz()
    # 你可以将其保存到文件或在合适的坏境（例如 Jupyter）中显示
    # graph_viz_object.render("chain_structure", format="png", view=True) # 保存并打开图片
    graph_dot_string = graph_viz_object.source
    print("Graphviz 定义:\n")
    # 显示 Graphviz DOT 字符串以供表示
    print(f'```graphviz\ndigraph G {{\ncompound=true;\n\t"{chain.first.id}" [label="ChatPromptTemplate" shape=box style=rounded];\n\t"{chain.middle[0].id}" [label="ChatOpenAI" shape=box style=rounded];\n\t"{chain.last.id}" [label="StrOutputParser" shape=box style=rounded];\n\t"{chain.first.id}" -> "{chain.middle[0].id}" [ltail=lc_0 lhead=lc_1];\n\t"{chain.middle[0].id}" -> "{chain.last.id}" [ltail=lc_1 lhead=lc_2];\n}} \n```')

except ImportError:
    print("未找到 Graphviz 库。请通过 'pip install graphviz' 安装。")
    print("改为 ASCII 表示：")
    print(chain.get_graph().draw_ascii())
````

> 一个简单 LCEL 链结构的可视化：提示词 (prompt) -> 模型 -> 解析器。

可视化图有助于确认组件按预期连接，这在进一步研究精巧 LCEL 组合的运行时调试之前尤其有用。

### 传统调试方法

别忘了标准 Python 调试技术。使用自定义组件（封装为 `RunnableLambda` 或自定义子类的函数、类）时，使用 `print` 语句、日志记录或 Python 调试器（如 `pdb` 或 IDE 的调试器）通常是检查自定义代码*内部*变量和控制流程最直接的途径。

### 整合所有方法：调试策略

1. **从 LangSmith 开始：** 对于任何非简单应用，尽早设置 LangSmith 追踪。它提供执行流程的最佳概览。
2. **重现问题：** 尝试用特定输入可靠地重现错误或非预期行为。
3. **追踪执行：** 使用 LangSmith 查看运行追踪。查看导致故障或输出不正确的每一步的输入和输出。
4. **隔离组件：** 如果特定组件（例如，特定提示词 (prompt)、工具调用、解析器）似乎有问题，尝试使用从追踪中发现的问题输入单独运行它。
5. **使用回调函数获取具体信息：** 如果 LangSmith 没有提供关于特定的内部事件或计算的足够细节，实行自定义回调处理器来记录或检查该特定数据片段。
6. **调试自定义代码：** 如果问题出在你整合的自定义函数或类中，请在该代码内部使用标准 Python 调试工具（`print`、`logging`、`pdb`）。
7. **检查组件配置：** 仔细检查每个组件的配置（例如，LLM 参数 (parameter)、提示模板变量、检索器设置）。

掌握这些调试方法非常基本，可帮助你超越简单的原型。随着你构建越来越精巧的链和代理，有效追踪、检查和诊断执行流程的能力将是创建可靠且高性能的生产应用不可或缺的。

## 参考资料

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/concepts/lcel/) — LangChain Developers (2024)
  Publisher: LangChain
  全面概述 LCEL，详细介绍了其组合方法和用于可视化链结构的工具，有助于调试。
- [Tracing in LangSmith](https://docs.smith.langchain.com/tracing) — LangSmith Team (2024)
  Publisher: LangChain
  LangSmith 追踪功能的官方文档，对 LangChain 应用程序的交互式调试、性能分析和监控至关重要。

---

[上一节](05-%E7%AE%A1%E7%90%86%E5%A4%8D%E6%9D%82%E9%93%BE%E4%B8%AD%E7%9A%84%E7%8A%B6%E6%80%81.md) · [下一节](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E8%87%AA%E5%AE%9A%E4%B9%89%E9%93%BE%E7%BB%84%E4%BB%B6.md)
