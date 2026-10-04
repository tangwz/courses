---
course: "intro-llm-agents"
chapter: "equipping-agents-with-tools"
lesson: "hands-on-equipping-agent-with-search"
sourceId: 6593
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-4-equipping-agents-with-tools/hands-on-equipping-agent-with-search"
title: "动手实践：为你的智能体添加搜索功能"
description: "一项将网页搜索工具集成到你的LLM智能体中的动手活动。"
order: 8
plots: []
sourceHash: "0e20c4546f07e571de0e3ac17a4022d5de0317df108a2305536ab6bccb2a362a"
sourceCorrections: []
---

现在你已经了解了工具对扩展智能体能力的重要性，以及如何定义一个像计算器这样的简单工具，是时候亲自动手处理一个更动态的能力了：为你的智能体添加网页搜索功能。这将使你的智能体能够获取不在其初始训练数据中的信息，例如时事或非常具体的知识。

我们将引导你集成一个搜索工具。你将学习如何：

1. 设置一个函数来调用外部搜索API。
2. 有效地描述这个工具，以便你的LLM智能体能理解其用途和使用方法。
3. 修改智能体的提示，以便在适当时鼓励使用搜索工具。
4. 观察智能体使用该工具回答以前无法回答的问题。

让我们从构建搜索功能开始。

### 设置你的搜索工具

为了执行网页搜索，我们的智能体需要一种与搜索引擎交互的方式。这通常通过API（应用程序编程接口）完成。许多服务提供搜索API，例如SerpApi、Google Custom Search API或Bing Search API。本次练习中，我们将概述一种通用方法。你需要注册其中一个服务并获取API密钥。大多数都提供免费套餐，足以用于学习目的。

**1. 安装必要的库**

如果你要与Web API交互，你可能需要Python的`requests`库来发送HTTP请求。如果你没有安装它，请打开你的终端或命令行并运行：

```bash
pip install requests
```

**2. 创建搜索函数**

让我们编写一个Python函数，它接受一个搜索查询，调用外部搜索API，并返回结果的摘要。下面是一个模板。你需要将`"YOUR_API_KEY"`和`API_ENDPOINT`替换为你选择的搜索API提供商的详细信息。API响应的结构也会有所不同，因此你需要调整如何提取和格式化`results`。

```python
import requests
import json # 用于解析API响应

# 替换为你的实际API密钥和端点
SEARCH_API_KEY = "YOUR_SEARCH_API_KEY"
SEARCH_API_ENDPOINT = "https://api.example-search.com/search" # 替换为实际的API端点

def web_search(query: str) -> str:
    """
    使用外部API对给定查询执行网页搜索，并返回格式化的搜索结果字符串。
    """
    if SEARCH_API_KEY == "YOUR_SEARCH_API_KEY":
        return "Search API key not配置。请设置它以使用搜索工具。"

    params = {
        "q": query,
        "api_key": SEARCH_API_KEY,
        # 添加你的搜索API所需的任何其他参数
        # 例如，'num_results': 3
    }

    try:
        response = requests.get(SEARCH_API_ENDPOINT, params=params)
        response.raise_for_status() # 对不良响应（4XX或5XX）引发HTTPError

        search_data = response.json() # 假设API返回JSON

        # 处理搜索数据。这部分高度依赖于你的API响应格式。
        # 例如，你可能会提取标题和摘要。
        snippets = []
        if "organic_results" in search_data: # SerpApi类结构的示例
            for result in search_data["organic_results"][:3]: # 获取前3个结果
                snippet = f"Title: {result.get('title', 'N/A')}\nSnippet: {result.get('snippet', 'N/A')}\nLink: {result.get('link', 'N/A')}"
                snippets.append(snippet)
        elif "webPages" in search_data and "value" in search_data["webPages"]: # Bing类结构的示例
             for result in search_data["webPages"]["value"][:3]:
                snippet = f"Title: {result.get('name', 'N/A')}\nSnippet: {result.get('snippet', 'N/A')}\nLink: {result.get('url', 'N/A')}"
                snippets.append(snippet)
        else:
            # 如果结构未知或不同，则为备用或更通用的解析
            # 这可能只是返回JSON的字符串表示
            # 或尝试查找常用键。为简单起见，我们将指示是否未找到结果。
            return "未找到预期格式的特定结果。原始数据可能可用。"

        if not snippets:
            return "未找到搜索结果。"

        return "\n\n".join(snippets)

    except requests.exceptions.RequestException as e:
        return f"搜索过程中出错: {e}"
    except json.JSONDecodeError:
        return "错误：无法从API解码搜索结果。"

# 示例用法（可选，用于直接测试函数）
# if __name__ == "__main__":
#     test_query = "LLM智能体最新进展"
#     search_results = web_search(test_query)
#     print(f"'{test_query}'的搜索结果:\n{search_results}")
```

**重要提示**：请记住替换占位符为你的实际API凭据，并调整结果解析逻辑以匹配你选择的特定API。请仔细阅读你选择的搜索API的文档。

### 为智能体描述工具

现在我们有了`web_search`函数，我们需要将它告知我们的LLM智能体。正如在“提示智能体使用工具”中讨论的，LLM需要为每个工具提供清晰的名称和描述，以理解其作用和使用时机。

让我们为搜索工具定义一个描述：

```python
search_tool_description = {
    "name": "WebSearch",
    "description": "使用此工具查找互联网上的当前信息，回答有关近期事件的问题，或获取任何主题的最新详情。输入应为清晰的搜索查询。",
    "function": web_search, # 指向我们的Python函数
    "input_schema": { # 描述该工具的预期输入
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "要在互联网上查找的搜索查询。"
            }
        },
        "required": ["query"]
    }
}
```

这个字典提供了：

- 一个`name`：一个简短、可识别的工具名称（例如，`WebSearch`）。
- 一个`description`：清晰解释工具的作用以及智能体何时应考虑使用它。这对于LLM的决策过程非常重要。
- 函数本身：以便智能体框架可以调用它。
- 一个`input_schema`：这告诉LLM（和智能体框架）该工具期望哪些参数 (parameter)。这里，它期望一个名为`query`的字符串参数。

你的智能体框架将使用这些信息。当LLM决定使用“WebSearch”时，框架将知道调用`web_search`函数，并将尝试从LLM生成的参数中提取`query`参数。

### 将工具集成到你的智能体中

如何集成此工具取决于你可能正在使用的特定智能体框架或你自定义智能体的结构。对于本次实践，我们假设你有一个涉及以下步骤的基本智能体循环：

1. 获取用户请求。
2. 将此请求（和工具描述）呈现给LLM。
3. LLM决定是直接响应还是使用工具。
4. 如果选择了工具，智能体将使用LLM提供的参数 (parameter)执行工具。
5. 然后将工具的输出反馈给LLM以生成最终响应。

这是一个你的智能体可能如何使用的简化流程。假设你有一个核心函数`run_agent_turn(user_request, available_tools)`：

```python
# 这是一个简化表示。你的实际智能体逻辑可能更复杂。
# 假设有一个LLM交互函数，例如`get_llm_decision(prompt, tools_descriptions)` 
# 它返回直接答案或工具调用指令。

def get_llm_decision(prompt_text, tools_list_for_llm):
    """
    你的LLM调用的占位符。
    这个函数会将提示和工具描述发送给LLM。
    LLM将响应以下任一情况：
    1. 直接答案。
    2. 一个指示工具调用的JSON对象，例如：
       {'tool_name': 'WebSearch', 'tool_input': {'query': '当前AI趋势'}}
    """
    # 在实际场景中，这涉及对LLM的API调用。
    # 在本例中，我们将根据关键词模拟LLM的决策。
    print(f"\n[LLM输入提示]:\n{prompt_text}\n") # 显示LLM看到的内容

    if "weather in Paris" in prompt_text.lower() and "WebSearch" in str(tools_list_for_llm):
        print("[LLM模拟决策]: 为'巴黎当前天气'使用WebSearch")
        return {"tool_name": "WebSearch", "tool_input": {"query": "current weather in Paris"}}
    elif "latest news on AI" in prompt_text.lower() and "WebSearch" in str(tools_list_for_llm):
        print("[LLM模拟决策]: 为'AI最新消息'使用WebSearch")
        return {"tool_name": "WebSearch", "tool_input": {"query": "latest news on AI"}}
    else:
        print("[LLM模拟决策]: 直接回答（或者我不知道）。")
        return {"answer": "在此模拟中，我只能在决定使用工具处理特定查询（如天气或新闻）时进行回答。"}

# 我们为智能体准备的可用工具列表
available_tools = {
    "WebSearch": search_tool_description 
    # 你可以在此处添加其他工具，例如上一节中的计算器
}

def run_agent_turn(user_request: str):
    print(f"用户: {user_request}")

    # 为LLM准备工具描述列表
    tools_for_llm_prompt = []
    for tool_name, tool_details in available_tools.items():
        tools_for_llm_prompt.append({
            "name": tool_details["name"],
            "description": tool_details["description"],
            "parameters": tool_details["input_schema"] # LLM需要知道要生成哪些参数
        })

    # 为LLM构造提示
    # 这是一个非常基本的提示。实际的智能体提示更复杂。
    prompt = f"""你是一个有用的助手。你可以使用以下工具：
{json.dumps(tools_for_llm_prompt, indent=2)}

用户查询: "{user_request}"

根据用户查询和可用工具，决定是否应使用工具。
如果需要，请回复一个JSON对象，指定工具名称及其输入。
例如：{{"tool_name": "WebSearch", "tool_input": {{"query": "你的搜索查询"}}}}
如果不需要工具，或者你可以直接回答，请提供答案。
"""

    llm_response = get_llm_decision(prompt, tools_for_llm_prompt)

    if "tool_name" in llm_response:
        tool_name = llm_response["tool_name"]
        tool_input_args = llm_response["tool_input"]

        if tool_name in available_tools:
            selected_tool = available_tools[tool_name]
            tool_function = selected_tool["function"]

            # 在这里，我们假设LLM提供的参数与函数的需求匹配。
            # 对于`web_search`，它期望一个'query'参数。
            query_arg = tool_input_args.get("query")
            if query_arg is None:
                print(f"智能体: LLM决定使用{tool_name}但未提供'query'。")
                return

            print(f"智能体: 使用工具'{tool_name}'，查询: '{query_arg}'")
            tool_output = tool_function(query_arg)
            print(f"工具'{tool_name}'输出:\n{tool_output}")

            # 现在，我们通常会将此输出反馈给LLM进行最终的综合。
            # 为简化此分步操作，我们只打印它。
            # 在完整的ReAct循环中，第二次LLM调用的提示将是：
            # final_prompt = f"{prompt}\n观察: {tool_output}\n思考: 现在我将使用这些信息来回答用户。\n最终答案:"
            # final_answer_from_llm = get_llm_decision(final_prompt, []) # 最终答案不需要工具
            # print(f"智能体（基于工具的最终答案）: {final_answer_from_llm.get('answer', '无法处理工具输出。')}")
            print(f"智能体: 基于搜索，这是我发现的: {tool_output}")

        else:
            print(f"智能体: LLM尝试使用未知工具: {tool_name}")
    elif "answer" in llm_response:
        print(f"智能体: {llm_response['answer']}")
    else:
        print("智能体: 我不确定如何进行。")

# 让我们试一试！
# 首先，确保web_search函数中的SEARCH_API_KEY已设置， 
# 否则它将返回“搜索API密钥未配置。”
run_agent_turn("AI的最新消息是什么？")
print("\n" + "="*50 + "\n")
run_agent_turn("给我讲个笑话。") # 这个查询在我们的模拟中可能不会触发搜索工具
```

**图表：智能体工具使用流程**

为了可视化智能体、LLM和工具如何交互，请考虑以下图表：

> 这个图表描述了当用户请求引导智能体使用网页搜索工具时，操作的序列。智能体核心协调用户、LLM和工具之间的交互，工具进而与外部API通信。

### 运行和测试你的增强型智能体

要全面测试此功能：

1. **替换占位符**：确保你已使用有效的API密钥和端点配置了`web_search`函数。
2. **LLM集成**：提供的`get_llm_decision`函数是一个*模拟*。在实际的智能体中，此函数会向LLM（如OpenAI的GPT模型、Anthropic的Claude，或通过Ollama等接口在本地运行的模型）进行API调用。你需要将用户查询、系统消息和可用工具的JSON描述传递给LLM，然后它会决定是否调用工具。
3. **测试查询**：
   - 尝试询问需要当前信息的问题：“今天伦敦天气如何？”或“谁赢得了最新的大型网球赛事？”
   - 询问LLM可能知道但可以验证或更新的问题：“法国的首都是哪里？”（智能体可能会直接回答，或者仍然选择搜索，这取决于其提示。）
   - 询问不需要搜索的问题：“给我讲一个关于勇敢骑士的故事。”

观察你的智能体（或你模拟的LLM决策逻辑）如何决定是否使用`WebSearch`工具。如果它使用了该工具，请检查输出。

### 故障排除与注意事项

- **API密钥错误**：确保你的API密钥正确且未超出配额。
- **API响应解析**：搜索API返回的JSON响应结构各不相同。你需要在测试期间在`web_search`函数内部`print(response.json())`，以理解其结构并相应调整你的解析逻辑（`snippets`的创建）。
- **LLM不使用工具**：
  - **提示**：你给LLM的主要提示需要清楚地说明工具可用并鼓励使用它们。工具描述本身必须非常清晰。
  - **工具描述**：如果`search_tool_description`中的`description`模糊不清，或者没有清楚地传达何时使用该工具，LLM可能会忽略它或错误地使用它。
  - **LLM“惰性”**：有时，LLM可能会尝试从其预训练 (pre-training)知识中回答，即使工具能提供更好的答案。更高级的提示技术可以帮助缓解这种情况。
- **过于急于使用工具**：反之，LLM可能尝试对所有事情都使用搜索工具。微调 (fine-tuning)提示和工具描述有助于平衡这种情况。
- **`web_search`中的错误处理**：示例函数具有基本的错误处理。工具应优雅地处理网络问题、超时和意外的API响应。

这个动手实践为使你的LLM智能体更有能力并与信息关联提供了一个基础性的一步。通过为它们配备搜索等工具，你显著扩展了它们能执行的任务范围以及响应的质量。当你构建更复杂的智能体时，你会发现自己正在创建和集成各种工具以适应不同需求。

## 参考资料

- [Tools](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF7P614T1eG1FwvtPTGvH05nTAyZsqGmTMQmftvpubWVAibePIlGYlyEgH3iXJ6f3BTcwwFLevK8a8xg8Y_B-edCbAlG18nEtd8Km2gWtjIZb2c0x42LgJB4Yyw7leZvvUzjU12hLs4y4g6gW5b_fV9pTw=) — LangChain (2024)
  广泛使用的 LLM 代理框架的官方文档，解释了如何定义、描述并将自定义工具（例如用于外部搜索 API 的工具）集成到代理工作流中。
- [Tools & Agents](https://docs.llamaindex.ai/en/stable/module_guides/tool_use/) — LlamaIndex (2024)
  本文档提供了在 LlamaIndex 框架内定义和使用工具的说明，这对于为大型语言模型代理提供网络搜索等外部功能很重要。
- [Custom Search JSON API](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQE83vfioN2mPvv_Ox5nzOxPmnlBD9pUamsQGhwwIYW6eIzVgPDV_0pcPb9Yoq1pcVyiFN16EIuhwGWC2Ndoov4ObgCh6R5anEhIFIsyjN47dU18Db9-ygigxgMAGA_FLR80hm-eQ5nGag==) — Google Developers (2024)
  Google Custom Search JSON API 的官方文档，涵盖了 API 密钥设置、端点使用、请求参数和响应结构的实践方面，用于将网络搜索集成到应用程序中。
