---
course: "prompt-engineering-llm-application-development"
chapter: "building-applications-llm-frameworks"
lesson: "practice-develop-agentic-application"
sourceId: 3054
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-llm-application-development/chapter-5-building-applications-llm-frameworks/practice-develop-agentic-application"
title: "动手实践：开发一个智能体应用"
description: "动手实践，使用LLM框架构建一个运用工具的简单智能体。"
order: 7
plots: []
sourceHash: "335a3b0a192f069ac79a2f4d7d61a13736e8f420d039170eeae021de7546ec66"
sourceCorrections: []
---

LLM框架的核心组成部分包括模型、提示模板、解析器、链、记忆，以及能够使用工具的智能体。智能体可以利用LLM进行推理 (inference)并与外部工具交互。此处构建一个使用LangChain的简单智能体，该智能体运用LLM进行推理并与外部工具（本例中是网络搜索引擎）进行交互。

本练习说明了智能体如何获取外部信息源来回答问题。

### 前提条件与设置

在开始之前，请确保您已安装所需的库。您还需要一个您所选LLM提供商的API密钥（本例中我们将使用OpenAI）。

1. **安装库：**

   ```bash
   pip install langchain langchain_openai duckduckgo-search
   ```

   - `langchain`: LangChain核心库。
   - `langchain_openai`: 提供OpenAI模型的集成。
   - `duckduckgo-search`: 用于执行DuckDuckGo搜索的库，LangChain可以将其作为一个工具使用。
2. **设置API密钥：** 确保您的OpenAI API密钥作为环境变量可用。您可以在终端会话中设置，或使用`.env`文件（这需要安装`python-dotenv`：`pip install python-dotenv`）。

   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

   或者，在您的项目目录中创建`.env`文件：

   ```
   OPENAI_API_KEY='your-api-key-here'
   ```

   并在您的Python脚本中使用以下代码加载：

   ```python
   # import os
   # from dotenv import load_dotenv
   # load_dotenv()
   # openai_api_key = os.getenv("OPENAI_API_KEY")
   ```

   *注意：* 务必安全地处理API密钥。避免将它们直接硬编码到您的源代码中，特别是如果您打算共享或进行版本控制。使用环境变量是一种标准做法。

### 构建搜索智能体

让我们创建一个Python脚本来构建和运行我们的智能体。

```python
import os
from langchain_openai import ChatOpenAI
from langchain import hub # Hub用于预制提示
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools

# 确保您的OPENAI_API_KEY环境变量已设置

# 1. 初始化LLM
# 我们使用ChatOpenAI，因为它能够使用OpenAI的“工具”功能（函数调用）
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# 2. 加载工具
# LangChain为常用工具提供便捷的加载器。
# “ddg-search”使用DuckDuckGo搜索引擎。
tools = load_tools(["ddg-search"])
# print(f"Loaded tools: {[tool.name for tool in tools]}")
# 预期输出：已加载工具：['duckduckgo_search']

# 3. 获取智能体提示模板
# 我们从LangChain Hub拉取一个为OpenAI工具智能体优化的预构建提示。
# 此提示包含输入、工具和中间步骤（agent_scratchpad）的占位符。
prompt = hub.pull("hwchase17/openai-tools-agent")
# 您可以像这样检查提示模板：
# print(prompt.pretty_repr())

# 4. 创建智能体
# create_openai_tools_agent函数将LLM、工具和提示绑定在一起。
# 它使用OpenAI的函数调用能力来确定使用哪个工具。
agent = create_openai_tools_agent(llm, tools, prompt)

# 5. 创建智能体执行器
# AgentExecutor会循环运行智能体。
# 它调用智能体，确定动作（使用哪个工具以及提供什么输入），
# 执行工具，获取结果（观察），然后将其传回给智能体，
# 直到智能体决定它有了最终答案。
# 'verbose=True'会显示智能体的思考过程。
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 运行智能体
# 让我们问一个需要最新外部信息的问题。
print("\n运行智能体查询1:")
response1 = agent_executor.invoke({
    "input": "巴西现在的首都和之前的首都是什么？"
})
print("\n最终答案 (查询1):")
print(response1['output'])

# 示例2：一个需要计算或最新信息的问题
print("\n运行智能体查询2:")
response2 = agent_executor.invoke({
    "input": "2023年温布尔登网球公开赛男子单打决赛的冠军是谁？"
})
print("\n最终答案 (查询2):")
print(response2['output'])
```

### 理解执行流程

当您运行此脚本时，在`AgentExecutor`中设置`verbose=True`会提供丰富的信息。您将看到类似于以下的输出（细节可能有所不同）：

```text
> 进入新的AgentExecutor链...

调用: `duckduckgo_search` 使用 `{'query': '巴西当前首都和之前首都'}`
工具输出
巴西利亚是巴西的联邦首都和联邦区政府所在地。这座城市位于巴西高原的巴西中西部地区。它由朱塞利诺·库比契克总统于1960年4月21日建立，作为新的国家首都。里约热内卢从1763年到1960年，近两个世纪以来一直是巴西的首都，直到政府机构迁至新建的巴西利亚。萨尔瓦多是巴西的第一个首都，从1549年到1763年。巴西曾有三个首都：萨尔瓦多、里约热内卢和巴西利亚。
巴西现在的首都是巴西利亚。
巴西之前的首都是里约热内卢（1763年至1960年）。在里约热内卢之前，第一个首都是萨尔瓦多（1549年至1763年）。

> 链执行完毕。

最终答案 (查询1):
巴西现在的首都是巴西利亚。之前的首都是里约热内卢。

运行智能体查询2:

> 进入新的AgentExecutor链...

调用: `duckduckgo_search` 使用 `{'query': "2023年温布尔登男子单打冠军"}`
工具输出
西班牙选手卡洛斯·阿尔卡拉斯在2023年温布尔登网球锦标赛男子单打决赛中击败塞尔维亚选手诺瓦克·德约科维奇后，手捧冠军奖杯。比赛于2023年7月16日在伦敦西南部的温布尔登全英草地网球俱乐部举行，是锦标赛的第十四天。卡洛斯·阿尔卡拉斯在决赛中以1-6, 7-6(8-6), 6-1, 3-6, 6-4击败诺瓦克·德约科维奇，赢得了2023年温布尔登网球锦标赛的男子单打冠军。这是他继2022年美网之后的第二个大满贯冠军。阿尔卡拉斯成为继曼努埃尔·桑塔纳和拉斐尔·纳达尔之后，第三位赢得温布尔登冠军的西班牙男子选手。2023年温布尔登男子决赛结果：卡洛斯·阿尔卡拉斯在五盘激战中击败诺瓦克·德约科维奇。更新于2023年7月16日星期日21:10。卡洛斯·阿尔卡拉斯结束了诺瓦克·德约科维奇在温布尔登的统治……卡洛斯·阿尔卡拉斯赢得了2023年温布尔登男子单打冠军。

> 链执行完毕。

最终答案 (查询2):
卡洛斯·阿尔卡拉斯在2023年温布尔登网球公开赛男子单打决赛中击败诺瓦克·德约科维奇，获得冠军。
```

让我们分解一下所发生的过程：

1. **接收输入：** `AgentExecutor`收到了包含用户问题的输入字典。
2. **智能体被调用：** 执行器将输入、工具描述和一个空的“暂存区”（用于中间步骤）传递给智能体（LLM + 提示）。
3. **推理 (inference)与工具选择：** LLM在`hwchase17/openai-tools-agent`提示及其函数调用训练的引导下，确定了：
   - 它无法直接知道答案。
   - `duckduckgo_search`工具可以找到所需信息。
   - 它为搜索工具（例如，“巴西当前首都和之前首都”）生成了合适的查询。
4. **工具执行：** `AgentExecutor`识别了选定的工具（`duckduckgo_search`）以及LLM生成的输入。它执行了搜索。
5. **接收观察结果：** DuckDuckGo的搜索结果作为“观察结果”返回给了`AgentExecutor`。
6. **合成答案：** 执行器将观察结果传回给智能体。LLM现在拥有了上下文 (context)（原始问题）和新信息（搜索结果）。它综合这些信息来生成最终的、面向用户的答案。
7. **最终输出：** `AgentExecutor`识别到LLM提供了最终答案（而非另一次工具调用），并返回了结果。

### 进一步尝试

既然您已经有了一个运行中的基本智能体，可以尝试以下修改：

- **不同问题：** 提出需要通过搜索获取不同类型信息的问题。尝试多跳问题（例如，“电影《奥本海默》的导演是谁？他还导演过哪些电影？”）。
- **添加更多工具：** 使用`load_tools(["ddg-search", "llm-math"], llm=llm)`加载`llm-math`工具，并提出涉及计算的问题（例如，“东京近似人口的平方根是多少？”）。观察智能体如何在搜索和计算器之间进行选择。
- **查看提示：** 打印`prompt`对象（`print(prompt.pretty_repr())`）以查看智能体使用的结构和指令。理解`input`、`tools`、`tool_names`和`agent_scratchpad`的作用。
- **更换LLM：** 尝试更换为其他模型，例如`gpt-4`（如果可用），通过更改`ChatOpenAI(model="...")`实现。观察推理 (inference)或工具使用是否发生变化。请注意潜在的成本差异。

此次动手实践提供了一个关于智能体如何在LangChain等框架中运行的理解。通过结合LLM推理和外部工具，您可以构建能够执行需要访问和处理最新或专业信息的复杂应用程序。

## 参考资料

- [LangChain: Agents](https://python.langchain.com/docs/concepts/#agents) — LangChain (2024)
  Publisher: LangChain
  LangChain中智能体的通用介绍，解释了它们的架构、类型以及如何将大型语言模型与工具结合。
- [OpenAI API: Tools](https://platform.openai.com/docs/guides/function-calling) — OpenAI (2024)
  Publisher: OpenAI
  官方文档，详细说明了如何将自定义工具和函数与OpenAI模型集成，这是智能体与工具交互的核心机制。
- [LangChain Hub](https://python.langchain.com/docs/langchain_api/langchain_hub/) — LangChain (2023)
  Publisher: LangChain Inc.
  LangChain Hub的文档，这是一个社区贡献的提示、链和智能体的仓库，演示了如何使用预构建的组件。
