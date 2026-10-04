---
course: "prompt-engineering-agentic-workflows"
chapter: "managing-agent-memory-prompts"
lesson: "hands-on-agent-dynamic-information"
sourceId: 6748
sourceUrl: "https://apxml.com/zh/courses/prompt-engineering-agentic-workflows/chapter-5-managing-agent-memory-prompts/hands-on-agent-dynamic-information"
title: "动手实践：用动态信息流提示代理"
description: "为需要处理并响应持续更新信息的代理制作提示。"
order: 8
plots: []
sourceHash: "dacccfcb493d4b828b134636c089926f4b191bb8c4467a716febdfae11970fff"
sourceCorrections: []
---

代理经常遇到这样的情况：信息并非静止不变，而是以持续流动的形式抵达。新闻推送、市场数据、传感器读数或社交媒体更新都是动态信息流的例子。有效提示代理处理、理解并根据此类不断变化的数据行动是一个常见难题。这个动手练习展示了如何设计提示，以帮助代理在处理一系列传入信息时管理其工作记忆并保持上下文 (context)。我们将着重介绍能让代理反复更新其理解、概括新输入并持续关注其任务的方法。

### 场景：实时事件追踪器

假设我们需要一个AI代理来监控短文本片段流（例如简化的新闻标题或社交媒体帖子），并维护与特定主题（例如“本地可再生能源项目的进展”）相关的事件的持续概括。代理需要接收每一条新信息，将其与已学到的内容结合，并简洁地更新其概括。

这种方法的要点在于构建提示，以便将代理当前的“状态”或“记忆”从一次交互带到下一次。代理处理一条信息后的输出成为处理下一条信息的输入的一部分。

### 初始提示设计

让我们为事件追踪代理设计一个初始提示。这个提示将设定代理的角色、目标，并为其当前理解（即概括）和新信息提供占位符。

```text
你是一个AI助手，负责监控并概括关于“本地可再生能源项目进展”的新闻片段。
你的目标是维护一个简明、最新的主要进展概括。

当前进展概括：
{current_summary}

新信息片段：
"{new_snippet}"

你的任务：
1. 分析“新信息片段”。
2. 如果它与“本地可再生能源项目进展”相关，并提供了尚未涵盖的新信息或重要更新，请将其整合到“当前进展概括”中。
3. 如果它不相关、冗余或只是非常小的更新，你可以表明无需对概括进行重大更改。
4. 提供更新后的概括。如果没有更改，返回现有概括。概括应保持简短，只关注主要点。

更新后的概括：
```

这里，`{current_summary}`最初将是“尚未报告。”或一个空字符串。`{new_snippet}`将是传入的文本片段。代理的响应，特别是“更新后的概括”部分，将成为下一次迭代的`{current_summary}`。

### 模拟信息流

让我们模拟一个信息流，看看代理在我们的提示结构引导下可能会如何处理它。

**迭代 1:**

- `当前概括`： “尚未报告。”
- `新片段`： “奥克维尔镇议会批准市政建筑太阳能电池板安装试点计划。”

**迭代 1 的提示：**

```text
你是一个AI助手，负责监控并概括关于“本地可再生能源项目进展”的新闻片段。
你的目标是维护一个简明、最新的主要进展概括。

当前进展概括：
尚未报告。

新信息片段：
"奥克维尔镇议会批准市政建筑太阳能电池板安装试点计划。"

你的任务：
1. 分析“新信息片段”。
2. 如果它与“本地可再生能源项目进展”相关，并提供了尚未涵盖的新信息或重要更新，请将其整合到“当前进展概括”中。
3. 如果它不相关、冗余或只是非常小的更新，你可以表明无需对概括进行重大更改。
4. 提供更新后的概括。如果没有更改，返回现有概括。概括应保持简短，只关注主要点。

更新后的概括：
```

**代理的可能响应（用于下一次`当前概括`的内容）：**
“奥克维尔镇议会批准了市政建筑的太阳能电池板试点计划。”

**迭代 2:**

- `当前概括`： “奥克维尔镇议会批准了市政建筑的太阳能电池板试点计划。”
- `新片段`： “斯普林菲尔德的新地热发电厂本周开始运营，有望为500户家庭供电。”

**迭代 2 的提示（片段）：**

```text
...
当前进展概括：
奥克维尔镇议会批准了市政建筑的太阳能电池板试点计划。

新信息片段：
"斯普林菲尔德的新地热发电厂本周开始运营，有望为500户家庭供电。"
...
更新后的概括：
```

**代理的可能响应：**
“主要进展：奥克维尔批准了市政建筑的太阳能电池板试点。斯普林菲尔德的新地热发电厂已开始运营，目标是为500户家庭供电。”

注意代理是如何被提示*整合*新信息的。这个过程要求代理进行一种信息凝练，如本章前面讨论的，确保概括保持简洁。

**迭代 3:**

- `当前概括`： “进展：奥克维尔批准了市政建筑的太阳能电池板试点。斯普林菲尔德的新地热发电厂已开始运营，目标是为500户家庭供电。”
- `新片段`： “奥克维尔天气预报：晴，最高气温25°C。”

**迭代 3 的提示（片段）：**

```text
...
当前进展概括：
重要进展：奥克维尔批准了市政建筑的太阳能电池板试点。斯普林菲尔德的新地热发电厂已开始运营，目标是为500户家庭供电。

新信息片段：
"奥克维尔天气预报：晴，最高气温25°C。"
...
更新后的概括：
```

**代理的可能响应：**
“重要进展：奥克维尔批准了市政建筑的太阳能电池板试点。斯普林菲尔德的新地热发电厂已开始运营，目标是为500户家庭供电。（关于奥克维尔天气的新片段与可再生能源进展无关）。”

或者，如果它被设定为在输出的自我反思部分更简洁：
“主要进展：奥克维尔批准了市政建筑的太阳能电池板试点。斯普林菲尔德的新地热发电厂已开始运营，目标是为500户家庭供电。”

这种反复进行的过程，其中代理的记忆（即概括）在提示中明确得到管理，使其能有效处理动态信息流，在它的上下文 (context)窗口限制内。

### 流程可视化

这个过程可以可视化为一个循环：

> 一个用于处理动态信息流的反复进行循环。代理的理解，作为“当前概括”传递，随每个新片段得到改进。

### 循环的Python伪代码

以下是你如何在Python风格的伪代码中实现这一点，假设你有一个`call_llm(prompt_text)`函数与你的语言模型交互：

```python
def process_dynamic_stream(initial_prompt_template, information_snippets):
    current_summary = "None reported yet."
    all_summaries = []

    for snippet_index, snippet in enumerate(information_snippets):
        # 为当前迭代构建提示
        prompt = initial_prompt_template.format(
            current_summary=current_summary,
            new_snippet=snippet
        )

        print(f"\n--- Iteration {snippet_index + 1} ---")
        print(f"输入片段: {snippet}")
        # print(f"发送给LLM的提示:\n{prompt}") # 用于调试

        # 调用LLM
        # 在实际系统中，你需要解析LLM的完整响应
        # 以提取“更新后的概括”部分。
        # 为简化起见，我们假设LLM直接返回更新后的概括
        # 或者我们有一个解析器。
        llm_response_text = call_llm(prompt) # 这是你的LLM调用的占位符

        # 如果LLM遵循指示，这是提取概括的简单方法
        # 这在实践中可能需要更实用。
        if "Updated Summary:" in llm_response_text:
            updated_summary_text = llm_response_text.split("Updated Summary:")[1].strip()
        else:
            # 回退或错误处理
            updated_summary_text = current_summary # 或记录错误

        current_summary = updated_summary_text
        all_summaries.append(current_summary)

        print(f"代理更新后的概括: {current_summary}")

    return all_summaries

# 示例用法：
prompt_template = """你是一个AI助手，负责监控并概括关于“本地可再生能源项目进展”的新闻片段。
你的目标是维护一个简明、最新的主要进展概括。

当前进展概括：
{current_summary}

新信息片段：
"{new_snippet}"

你的任务：
1. 分析“新信息片段”。
2. 如果它与“本地可再生能源项目进展”相关，并提供了尚未涵盖的新信息或重要更新，请将其整合到“当前进展概括”中。
3. 如果它不相关、冗余或只是非常小的更新，你可以表明无需对概括进行重大更改。
4. 提供更新后的概括。如果没有更改，返回现有概括。概括应保持简短，只关注主要点。

更新后的概括：
"""

simulated_snippets = [
    "Oakville town council approves pilot program for solar panel installations on municipal buildings.",
    "The new geothermal plant in Springfield began operations this week, promising to power 500 homes.",
    "当地面包店赢得最佳羊角面包奖。",
    "派恩维尤的研究人员宣布在太阳能电池储能效率方面取得突破。"
]

# 这是一个LLM调用函数
def call_llm(prompt_text):
    # 在实际应用中，此函数将向LLM发出API调用。
    # 对于此伪代码，我们将根据提示模拟响应。
    # 此模拟高度简化。
    if "Oakville town council approves" in prompt_text and "None reported yet" in prompt_text:
        return "Updated Summary: Oakville town council approved a solar panel pilot for municipal buildings."
    elif "geothermal plant in Springfield" in prompt_text:
        if "Oakville town council approved" in prompt_text:
             return "Updated Summary: Oakville approved solar pilot. Springfield's new geothermal plant operational (powers 500 homes)."
        else:
            return "Updated Summary: Springfield's new geothermal plant started operations, aiming to power 500 homes."
    elif "best croissants" in prompt_text:
        # 提取以前的概括，如果片段不相关则返回
        summary_marker = "Current Summary of Developments:\n"
        current_summary_start = prompt_text.find(summary_marker) + len(summary_marker)
        current_summary_end = prompt_text.find("\n\nNew Information Snippet:")
        prev_summary = prompt_text[current_summary_start:current_summary_end].strip()
        return f"Updated Summary: {prev_summary} (关于羊角面包的新片段不相关)。"
    elif "battery storage efficiency" in prompt_text:
        return "Updated Summary: Oakville: solar pilot. Springfield: geothermal plant. Pineview: breakthrough in battery storage for solar."
    return "Updated Summary: 对概括无更改。"


# 运行模拟
# final_summaries = process_dynamic_stream(prompt_template, simulated_snippets)
# print("\n--- 每一步后的最终概括列表 ---")
# for i, summary in enumerate(final_summaries):
#    print(f"概括 {i+1} 后: {summary}")
```

要运行此示例，你需要用实际的语言模型调用替换`call_llm(prompt)`。上述模拟的`call_llm`仅用于说明概括可能如何演变。

### 动态信息流的考量

- **保持任务专注**：提示不断提醒代理其主要目标（“监控和概括关于‘本地可再生能源项目进展’的新闻片段”）。这有助于保持专注，如“在扩展交互中保持任务专注”中所述。
- **信息凝练**：指示“概括应保持简短，只关注主要点”明确要求代理执行概括和凝练。随着片段数量的增加，这一点很重要，确保`current_summary`不超过上下文 (context)窗口限制或变得难以管理。
- **处理不相关/冗余信息**：提示指导代理如何处理不相关或重复的信息，这在数据集中很常见。这涉及到保持信息的一致性。
- **错误传播**：如果代理在一个概括中犯了错误，那个错误将被带入后续的提示中。设计鼓励自我纠正的提示或使用验证步骤可以减轻这种情况。例如，你可以添加一个步骤：“如果你选择放弃新信息，请简要说明你的理由。”
- **近期偏见**：LLM有时可能会对提示中的最新信息给予过多的权重 (weight)。仔细的结构设计并明确要求代理考虑整个`current_summary`会有所帮助。
- **可扩展性**：对于非常长的数据流或非常复杂的状态，这种在提示内管理记忆的方法可能会达到上下文窗口的限制。在那时，整合外部记忆存储或更复杂的概括策略（例如，分层概括）等方法变得必要，这就涉及到了长期记忆管理。

这个动手练习展示了一种基本方法，用于使AI代理处理动态信息。通过深思熟虑地构建提示来管理滚动状态或概括，你可以引导代理在数据不断变化的环境中有效执行。请记住，根据观察到的代理行为反复改进提示设计是开发可靠代理系统的一个标准部分。

## 参考资料

- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2304.03442](https://doi.org/10.48550/arXiv.2304.03442)
  这项研究描述了一种模拟人类行为的AI代理架构，强调了记忆流、反思和规划，这对于代理维护状态并根据动态信息采取行动非常重要。
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, Haofen Wang (2024)
  Journal: arXiv preprint arXiv:2312.10997; DOI: [10.48550/arXiv.2312.10997](https://doi.org/10.48550/arXiv.2312.10997)
  这项调查概述了检索增强生成（RAG）技术，这些技术使大型语言模型能够访问和整合外部信息，解决了上下文窗口的限制，这对于代理的长期记忆和动态数据流至关重要。
