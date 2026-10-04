---
course: "intro-llm-agents"
chapter: "llm-agent-memory"
lesson: "hands-on-adding-contextual-memory-to-agent"
sourceId: 6622
sourceUrl: "https://apxml.com/zh/courses/intro-llm-agents/chapter-6-llm-agent-memory/hands-on-adding-contextual-memory-to-agent"
title: "动手实践：为你的智能体添加情境记忆"
description: "一个实践活动，通过添加对话历史作为短期记忆来增强你的LLM智能体。"
order: 7
plots: []
sourceHash: "94f7ff0ea4afa3eb47dcfd601206a1539e42872db8a72a741eafdcb4374fdd2c"
sourceCorrections: []
---

既然我们已经讨论了记忆对于智能体维持连贯互动的重要性，也了解了短期记忆的理念，现在是时候将其付诸实践了。在此实践环节中，我们将修改一个基础智能体，使其包含一种简单的对话记忆形式。这将使我们的智能体能够“记住”对话中的先前回合，从而产生更自然、更具情境感的回复。

存储互动历史（包括用户输入和智能体回复），并在每次新请求时将此历史信息反馈给大型语言模型（LLM），是实现其掌握之前对话背景的主要方法。这样，LLM就能掌握之前对话的背景。

### 设定场景：一个简单的问候智能体

设想你有一个基础智能体，其主要任务是问候用户，并可能询问他们今天过得怎么样。没有记忆，每次你与它互动，都像是初次见面。

- **用户：** 你好！
- **智能体：** 你好！今天我能帮你什么？
- **用户：** 我很好，我叫Alex。
- **智能体：** 你好！今天我能帮你什么？（立即忘记Alex的名字）

有了记忆，我们希望智能体能记住Alex。

### 我们需要什么

对于本次练习，我们假设你有一种与LLM互动的方式。这可以通过API（如OpenAI、Anthropic或Cohere的）或本地运行的模型来实现。添加记忆的核心思路与你使用的具体LLM无关。

我们的示例将主要使用Python，因为它在AI开发中很常用。你将需要：

1. 一个Python环境。
2. 一种向LLM发送请求的方式（例如，使用`requests`库针对自定义API，或使用官方客户端库如`openai`）。

### 第一步：初始化记忆

实现短期记忆最简单的方式是使用列表。列表中的每个元素可以代表对话的一部分。我们可以存储用户消息和智能体回复的历史记录。

让我们从初始化一个空列表开始，用于保存对话历史：

```python
conversation_history = []
```

### 第二步：构建记忆条目

我们需要一种一致的方式将消息添加到我们的历史记录中。一个好的方法是存储字典，其中每个字典代表一条消息，并指明发送者（“用户”或“助手”/“智能体”）以及消息内容。

例如：
`{"role": "user", "content": "Hello, my name is Alex."}`
`{"role": "assistant", "content": "Nice to meet you, Alex! How are you today?"}`

这种结构很常见，特别是当你使用OpenAI的聊天补全等API时。

### 第三步：更新记忆

每次用户说些什么，我们就将其添加到`conversation_history`中。同样，每次智能体回复时，我们也会将其回复添加到历史记录中。

让我们设想一个函数`get_llm_response(prompt_with_history)`，它接受我们对话的当前状态并返回LLM的下一条消息。

```python
# (假设get_llm_response已在其他地方定义)

def chat_with_agent():
    conversation_history = []
    print("智能体：你好！我是你的友好助手。输入'quit'退出。")

    while True:
        user_input = input("你：")
        if user_input.lower() == 'quit':
            print("智能体：再见！")
            break

        # 将用户消息添加到历史记录
        conversation_history.append({"role": "user", "content": user_input})

        # 准备LLM的输入，包括历史记录
        # 对于某些LLM，你可以直接传递整个历史记录。
        # 对于另一些，你可能需要将其格式化为单个字符串。
        # 我们假设我们的get_llm_response函数处理了这一点。

        # 在实际情况中，发送给LLM的提示会通过结合
        # 系统消息（定义智能体角色/任务）和
        # conversation_history来构建。
        # 为此处简单起见，我们假设get_llm_response只需要历史记录。

        # 核心思路：LLM能看到过去的对话
        agent_response_content = get_llm_response(conversation_history)

        print(f"智能体：{agent_response_content}")

        # 将智能体回复添加到历史记录
        conversation_history.append({"role": "assistant", "content": agent_response_content})

# 你的LLM交互函数的占位符
# 在实际应用中，这会调用LLM API
def get_llm_response(current_history):
    # 这是一个非常简化的模拟。
    # 真实的LLM会使用历史记录生成与上下文相关的回复。
    # 例如，如果提到了“Alex”，它可能会使用这个名字。

    # 根据历史记录模拟LLM行为
    last_user_message = ""
    if current_history and current_history[-1]["role"] == "user":
        last_user_message = current_history[-1]["content"].lower()

    if "my name is" in last_user_message:
        name = last_user_message.split("my name is")[-1].strip().capitalize()
        return f"很高兴认识你，{name}！我还能帮你什么？"
    elif "how are you" in last_user_message:
        return "我很好，谢谢你的询问！"
    elif "weather" in last_user_message:
        # 检查之前是否提到了地点
        location = None
        for message in reversed(current_history[:-1]): # 在过去的消息中查找
            if "i live in" in message["content"].lower():
                location = message["content"].lower().split("i live in")[-1].strip().capitalize()
                break
            if "near" in message["content"].lower(): # 简单检查
                location_parts = message["content"].lower().split("near")
                if len(location_parts) > 1:
                    location = location_parts[-1].strip().capitalize()
                    break
        if location:
            return f"我没有实时天气访问权限，但我记得你提到了{location}。希望那里天气很好！"
        else:
            return "我没有实时天气访问权限。你住在哪里？"
    else:
        return "我在这里和你聊天。你有什么想说的吗？"

# 要运行聊天（在Python环境中）：
# chat_with_agent()
```

### 第四步：LLM的视角——使用历史记录进行提示

重要之处在于调用LLM时`conversation_history`的使用方式。大多数专为聊天设计的现代LLM可以接受消息列表，通常包含“系统”、“用户”和“助手”等角色。

A typical call to an LLM API might look something like this (example):

```python
# 这是get_llm_response内部发生情况的一个简化、说明性示例
# client = YourLLMProviderClient()

# response = client.chat.completions.create(
# model="some-llm-model-name",
# messages=[
# {"role": "system", "content": "你是一个乐于助人的助手。"},
# *conversation_history # 展开用户/助手消息列表
# ]
# )
# agent_response_content = response.choices[0].message.content
```

“系统”消息有助于设定智能体的角色或总体指令。然后，整个`conversation_history`被传递。LLM利用这一系列消息来理解正在进行的对话，并生成相关的下一条回复。

### 第五步：运行与观察

如果你实现`chat_with_agent()`函数（以及一个调用实际LLM的真实`get_llm_response`函数），你可以测试它：

**互动 1：**

```
智能体：你好！我是你的友好助手。输入'quit'退出。
你：你好，我叫Sarah。
智能体：很高兴认识你，Sarah！我还能帮你什么？
```

*(此时，`conversation_history`将包含你的消息和智能体的回复。)*

**互动 2：**

```
你：我告诉你我叫什么名字来着？
智能体：你告诉我你叫Sarah。
```

*(智能体“记住”了，因为“你好，我叫Sarah”是发送给LLM的第二次查询历史的一部分。)*

**互动 3：**

```
你：我住在伦敦。
智能体：我在这里和你聊天。你有什么想说的吗？（或LLM基于此的更具体的回应）
```

*(记忆已更新：`{"role": "user", "content": "I live in London."}`)*

**互动 4：**

```
你：今天天气怎么样？
智能体：我没有实时天气访问权限，但我记得你提到了伦敦。希望那里天气很好！
```

*(智能体使用历史记录中之前提到的地点“伦敦”来提供更具情境感但仍有限的回复。)*

这个例子对`get_llm_response`使用了非常基础的模拟。一个真实的LLM，在提供历史记录的情况下，会更自然、更有效地捕捉这些情境线索。

### 讨论：我们取得了什么成就及其局限性

通过添加这个简单的基于列表的`conversation_history`，我们的智能体现在可以：

- 引用用户在早期回合中提供的信息。
- 避免在信息已给出的情况下提出重复性问题。
- 进行更流畅、听起来更自然的对话。

然而，这种基础的短期记忆形式存在局限性：

1. **上下文 (context)窗口大小：** LLM有最大的上下文窗口，这是它们一次可以处理的文本总量（提示词 (prompt) + 历史记录 + 完成内容）。如果对话变得非常长，`conversation_history`可能会超出此限制。发生这种情况时，你需要采取策略，如截断历史记录（例如，只保留最后N个回合）或总结对话的旧部分。
2. **无长期持久性：** 这种记忆是临时的。如果你重启智能体，`conversation_history`列表就会重置。为了实现跨会话的持久记忆，你需要将历史记录保存到文件或数据库中。
3. **无高级检索：** 智能体并没有从更细致的理解层面来“理解”记忆。它只是被喂入了原始文本。更高级的记忆系统可能涉及对历史记录进行语义搜索或创建结构化摘要等技术。

这个实践练习展示了为LLM智能体提供对话上下文的基本原理。即使是这种简单的实现，也显著提升了智能体进行连贯对话的能力。当你研究更复杂的智能体时，你会遇到更高级的记忆管理技术，但将相关的历史信息反馈给LLM的核心思路仍是一个重要构成部分。

## 参考资料

- [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) — OpenAI (2024)
  为大型语言模型（LLM）构建对话输入提供了实用指南，具体演示了用于维护对话历史的`role`和`content`消息格式。
- [Memory](https://python.langchain.com/docs/modules/memory/) — LangChain (2024)
  详细介绍了各种记忆类型和管理策略，包括超越基本列表记忆的持久化和更复杂的方案，用于构建LLM代理。
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein (2023)
  Journal: arXiv preprint arXiv:2304.03442
  介绍了一种复杂的LLM代理架构，其特点是记忆流（观察、反思、规划），探讨了超越简单对话历史的高级记忆概念，以创建复杂的代理行为。
