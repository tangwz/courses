---
course: "building-llm-apps-with-langchain"
chapter: "developing-autonomous-agents"
lesson: "practice-web-searching-agent"
sourceId: 7440
sourceUrl: "https://apxml.com/zh/courses/building-llm-apps-with-langchain/chapter-6-developing-autonomous-agents/practice-web-searching-agent"
title: "实操：一个网络搜索代理"
description: "一个动手操作，构建一个能通过搜寻网络并归纳信息来回答问题的代理。"
order: 6
plots: []
sourceHash: "02dc2e27f5c358502d7a9aa58d73c92701b37308f808d3bfac9387ee42e8adc6"
sourceCorrections: []
---

### 前提条件与环境配置

在我们开始之前，请确保您的环境已准备好。我们将使用 Tavily Search API 作为我们的网络搜索工具，因为它与 LangChain 集成得很好，并且专注于提供清晰、适合 AI 处理的搜索结果。

首先，安装所需的 Python 包。我们包含 `langgraph`，因为它是 LangChain 生态系统中构建代理工作流的现代标准。

```bash
pip install langchain langchain-openai tavily-python langgraph
```

接下来，您需要为 OpenAI（用于 LLM）和 Tavily（用于搜索工具）配置 API 密钥。您可以从他们的网站获取 Tavily API 密钥。将这些设置为环境变量，以实现安全访问。

```bash
export OPENAI_API_KEY="your-openai-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

环境准备就绪后，我们就可以开始组合代理的组件了。

### 1. 定义工具

代理的能力由它可用的工具来决定。为了我们的目的，一个工具就足够了：一个网络搜索器。LangChain 为 Tavily Search API 提供了便捷的集成。

让我们导入并初始化这个工具。我们将搜索结果数量限制为三个，以保持提供给 LLM 的上下文 (context)简洁。

```python
from langchain_community.tools.tavily_search import TavilySearchResults

# 初始化工具。我们可以配置它返回结果的数量。
# 更少的结果速度更快，并且使用的 token 也更少。
tools = [TavilySearchResults(max_results=3)]
```

这个 `tools` 列表将提供给我们的代理，使 `TavilySearchResults` 函数可供其调用。

### 2. 初始化 LLM

代理的核心是一个能够判断使用哪个工具的 LLM。对于需要遵循指令和使用函数定义的任务，更先进的模型表现更好。我们将使用 OpenAI 的 `gpt-4o` 模型。

```python
from langchain_openai import ChatOpenAI

# 初始化 LLM。温度设置为 0 会促使输出更具确定性和事实性。
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

在下一步中，我们将把这个 LLM 实例直接传递给代理的构造函数。

### 3. 创建代理

为了构建代理，我们将使用 `langgraph`，它使我们能够创建稳定、有状态的代理工作流。`create_react_agent` 函数构建了一个图，该图作为我们代理的运行时。它管理着调用 LLM、解释决策（思考）、执行所选工具（行动）并将输出反馈给模型（观察）的循环。

下面的图表说明了这个执行循环。

> 代理图管理着 LLM 推理 (inference)、使用工具采取行动、观察结果并将其反馈到推理过程中的循环。

现在，让我们用代码创建代理。

```python
from langgraph.prebuilt import create_react_agent

# create_react_agent 函数将 LLM 绑定到工具，并设置
# 管理交互循环的执行图。
agent_executor = create_react_agent(llm, tools)
```

### 4. 运行代理

所有组件组装完毕后，我们的代理就可以使用了。让我们问一个问题，这个问题需要从两个不同方面：技术和金融，搜索信息。

代理期望接收一个包含消息列表的字典作为输入。我们将问题格式化为用户消息。

```python
question = "What was the stock price of NVIDIA (NVDA) at the time of the announcement of the Blackwell GPU architecture?"

# 用用户输入调用代理
response = agent_executor.invoke({"messages": [("user", question)]})

print("\nFinal Answer:")
# 最终答案包含在返回历史的最后一条消息中
print(response["messages"][-1].content)
```

当代理运行时，它会在内部执行必要的步骤。其推理 (inference)过程的追踪如下所示：

```text
思考：用户询问 NVIDIA (NVDA) 在 Blackwell GPU 架构发布时的股价。我需要找到两部分信息：
1.  Blackwell GPU 架构的发布日期。
2.  NVDA 在该日期的股价。
我将首先搜索发布日期。

行动：
{
  "tool": "tavily_search_results",
  "tool_input": "NVIDIA Blackwell GPU architecture announcement date"
}
观察：NVIDIA 在 2024 年 3 月 18 日的 GTC 2024 主题演讲中正式发布了 Blackwell 架构。

思考：我已找到发布日期：2024 年 3 月 18 日。现在我需要找到 NVDA 在该日期或前后时间的股价。我可以再次搜索此信息。

行动：
{
  "tool": "tavily_search_results",
  "tool_input": "NVDA stock price on March 18, 2024"
}
观察：2024 年 3 月 18 日，NVIDIA 公司 (NVDA) 股票收盘价为 884.55 美元。

思考：我已找到两部分信息。
1. Blackwell GPU 架构于 2024 年 3 月 18 日发布。
2. 当天 NVDA 的收盘价为 884.55 美元。
现在我可以向用户提供最终答案了。

最终答案：NVIDIA 于 2024 年 3 月 18 日发布了 Blackwell GPU 架构。当天，NVDA 的收盘价为 884.55 美元。
```

这个序列展示了代理的能力。它正确地将复杂的查询分解为子问题，按顺序执行其搜索工具以收集必要的事实，然后将这些事实综合成一个完整准确的答案。您现在构建了一个简单但有效的自主系统，能够与外部环境交互以实现一个目标。这种基本模式是构建更复杂代理的依据，这些代理可以与数据库、API 和其他软件系统交互。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao (2023)
  Journal: ICLR; DOI: [10.48550/arXiv.2210.03629](https://doi.org/10.48550/arXiv.2210.03629)
  介绍了 ReAct 框架，它结合了推理和行动，使大型语言模型能够通过生成口头推理轨迹和特定任务行动来执行复杂任务。此框架与文中描述的智能体循环直接相关。
- [LangChain Agents Documentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEIwo-5MHghjnJCgJwnll1DRqFmRldfB2QmRCnHXaOzcLscH_yU29lV91-cQIdBag_1dcE0_qPN3w82BXNXRS4IoYIJJvqoY-F5LaNoEWOCD7e4m2Gwt7kyybVAYBiA46YMrgi1IR4Vd73FF_d9hy2VWw==) — LangChain (2024)
  LangChain 框架中构建和使用智能体的官方指南，涵盖智能体概念、工具和执行器等组件以及集成示例。
- [Tools (Function calling)](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI
  指导如何通过定义模型可调用的函数，使 GPT-4o 等大型语言模型与外部工具交互，与智能体的行动生成直接相关。
