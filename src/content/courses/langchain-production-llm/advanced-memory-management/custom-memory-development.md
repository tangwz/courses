---
course: "langchain-production-llm"
chapter: "advanced-memory-management"
lesson: "custom-memory-development"
sourceId: 3880
sourceUrl: "https://apxml.com/zh/courses/langchain-production-llm/chapter-3-advanced-memory-management/custom-memory-development"
title: "自定义内存模块开发"
description: "构建针对特定应用需求定制的内存类。"
order: 4
plots: []
sourceHash: "50bc43cfb4a025c9551a0d6a588b92af693f767b76f4eb7176304ceb7a5a764c"
sourceCorrections: []
---

尽管 LangChain 提供了一套多功能的内置内存模块，但生产应用常遇到标准实现无法完全满足的独特上下文 (context)需求。您可能需要与专有数据库集成，实行领域特定的上下文筛选，实现新颖的总结技术，或者以适合您特定用例的方式管理对话状态。在这种情况下，开发自定义内存模块就很有必要。

构建自定义内存模块让您能够精确定义对话历史如何存储、检索和处理，从而对提供给您的 LLM、链和代理的上下文提供细致的控制。

### 理解内存接口

任何 LangChain 内存模块的核心都必须遵循特定接口。您通常会使用的主要基类是 `BaseChatMemory`，它本身继承自 `BaseMemory`。要创建可用的自定义内存模块，您需要实现几个必要的方法：

1. **`__init__(self, ...)`**: 这是您的构造函数。您将在此处初始化任何必需的资源，例如数据库连接、配置参数 (parameter)（如历史记录长度限制）或内部状态变量。您通常还会调用父类的构造函数 (`super().__init__(...)`)。
2. **`memory_variables(self) -> List[str]`**: 此属性应给出您的内存模块期望注入到提示中的字符串键列表。最常见的键是 `"history"`，但您可以根据自己的逻辑定义其他键。
3. **`load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]`**: 这是获取上下文 (context)的主要方法。它接收一个包含链或代理*当前*输入的字典（不包含内存变量）。您的实现应根据这些输入获取相关的对话历史或状态，并返回一个字典，其中的键与 `memory_variables` 中定义的键匹配，值包含格式化的上下文（例如，一个包含过去消息的字符串）。
4. **`save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None`**: 此方法在链或代理执行*后*调用。它接收原始输入（不包含内存变量）和 LLM 生成的最终输出。您的任务是处理此新的交互轮次，并根据您的自定义逻辑妥善存储（例如，添加到内部列表、保存到数据库或更新实体信息）。
5. **`clear(self) -> None`**: 此方法应重置内存状态，以清除对话历史。

以下是说明交互流程的图示：

> LangChain 执行单元与自定义内存模块核心方法之间的交互流程。

### 设计您的自定义逻辑

自定义内存的效用在于您在 `load_memory_variables` 和 `save_context` 中实现的逻辑。可以考虑以下方面：

- **专业化存储：** 不同于简单的列表或标准数据库，可以将对话轮次存储在图数据库中以追踪实体和关系，或存储在时间序列数据库中以研究交互模式，或存储在专有内部系统中。
- **高级筛选/总结：** 实现逻辑，根据意图检测、情感分析或关键词匹配来筛选掉不相关的轮次，然后再返回上下文 (context)。您可以在 `load_memory_variables` 内部或在 `save_context` 内部定期集成自定义总结模型或算法。
- **上下文塑形：** 以您的提示所需的特定格式返回上下文，例如将其构建为 XML、JSON 或自定义 DSL，而不仅仅是简单的消息字符串。
- **动态上下文窗口：** 不同于固定 `k` 个轮次，可以根据当前输入的复杂性或估算的 token 数量，动态决定加载多少历史记录。
- **外部数据集成：** 在 `load_memory_variables` 步骤中，根据最近对话历史中提到的实体或主题，通过获取外部 API 或知识库中的相关数据来丰富已加载的上下文。

### 实现示例：`RelevantTurnMemory`

让我们描绘一个简单的自定义内存模块，它只存储和检索包含特定关键词的轮次。这是一个基础示例，但它展现了核心实现模式。

```python
import re
from typing import Any, Dict, List, Optional
from langchain.memory import BaseChatMemory
from langchain_core.messages import get_buffer_string

# 假设 relevant_keywords 在其他地方定义，例如 ["项目阿尔法", "预算", "里程碑"]
relevant_keywords = ["project alpha", "budget", "milestone"]

class RelevantTurnMemory(BaseChatMemory):
    """
    只存储和检索输入或输出中包含特定关键词的对话轮次的内存。
    """
    keywords_pattern: re.Pattern
    memory_key: str = "relevant_history"
    input_key: Optional[str] = None
    output_key: Optional[str] = None
    return_messages: bool = False

    def __init__(self, keywords: List[str], return_messages: bool = False,
                 memory_key: str = "relevant_history",
                 input_key: Optional[str] = None, output_key: Optional[str] = None,
                 **kwargs):
        # 将 kwargs (例如 chat_memory) 传递给父类
        super().__init__(**kwargs)
        self.keywords_pattern = re.compile("|".join(map(re.escape, keywords)), re.IGNORECASE)
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key

    @property
    def memory_variables(self) -> List[str]:
        """由 load_memory_variables 返回的键列表。"""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """获取相关的对话历史。"""
        # 因为 save_context 只存储相关的轮次，
        # 所以 self.chat_memory.messages 包含了我们所需的内容。
        if self.return_messages:
            context = self.chat_memory.messages
        else:
            context = get_buffer_string(
                self.chat_memory.messages,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: context}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """如果相关，则将上下文保存到内存。"""
        # 识别正确的输入/输出字符串
        # 如果未指定键，则回退到 "input" 和 "output"
        prompt_input_key = self.input_key or "input"
        prompt_output_key = self.output_key or "output"

        input_str = str(inputs.get(prompt_input_key, ""))
        output_str = str(outputs.get(prompt_output_key, ""))

        is_relevant = bool(self.keywords_pattern.search(input_str)) or \
                      bool(self.keywords_pattern.search(output_str))

        if is_relevant:
            # 直接添加到底层的聊天内存存储中
            self.chat_memory.add_user_message(input_str)
            self.chat_memory.add_ai_message(output_str)

    def clear(self) -> None:
        """清除相关内存。"""
        self.chat_memory.clear()

# --- 如何使用 ---
# from langchain_openai import ChatOpenAI
# from langchain.chains import ConversationChain

# llm = ChatOpenAI(model="gpt-4", temperature=0)
# custom_memory = RelevantTurnMemory(keywords=["billing", "invoice", "payment"])
# conversation = ConversationChain(
#     llm=llm,
#     memory=custom_memory,
#     verbose=True
# )

# # 示例交互
# print(conversation.predict(input="Hello, how are you?"))
# # -> 此轮次将不会被保存。

# print(conversation.predict(input="I have a question about my recent invoice."))
# # -> 此轮次将因关键词“发票”而被保存。

# print(conversation.predict(input="What was the total amount?"))
# # -> 当为本轮次加载内存时，会包含之前相关的轮次。
```

### 集成您的自定义内存

使用您的自定义内存模块很直接。实例化您的自定义类，并将该实例传递给 LangChain `Chain` 或 `AgentExecutor` 的 `memory` 参数 (parameter)：

```python
# 假设 MyCustomDatabaseMemory 的定义与上述类似
# 并在 __init__ 中正确处理数据库连接

# db_connection_string = "your_db_connection_details"
# custom_db_memory = MyCustomDatabaseMemory(connection_string=db_connection_string)

# agent_executor = AgentExecutor.from_agent_and_tools(
#     agent=my_agent,
#     tools=my_tools,
#     memory=custom_db_memory, # 传递自定义内存实例
#     verbose=True
# )

# # 现在 agent_executor 将使用您的自定义内存逻辑
# agent_executor.invoke({"input": "User query..."})
```

### 高级内存设计

- **持久性：** 对于生产环境，您的内存可能需要在单个脚本运行的生命周期中保持持久。在 `save_context` 和 `__init__`（或专用加载方法）中实现保存/加载逻辑，以与数据库、文件或其他持久存储进行交互。确保 I/O 操作的错误处理。
- **异步操作：** 如果您的应用程序使用 LangChain 的异步功能 (`ainvoke`)，您的自定义内存最好实现其异步对应方法：`aload_memory_variables` 和 `asave_context`。这些方法应使用 `async`/`await` 兼容的库（例如 `aiohttp`、`asyncpg`）执行 I/O 操作。
- **状态管理：** 注意状态的管理方式，特别是当内存模块在多个并发请求或用户之间共享时。如果需要，确保线程安全或使用适当的数据库事务机制。精心设计您的存储方案，以隔离不同的对话历史记录（例如，使用会话 ID）。
- **序列化：** 如果您需要在进程之间传递内存状态或存储快照，请确保您的内部状态是可序列化的（例如，使用 Python 的 `pickle` 或在 JSON 格式之间转换）。LangChain 提供的 `BaseMessage` 对象通常是可序列化的。
- **测试：** 彻底对您的自定义内存模块进行单元测试。验证 `load_memory_variables` 在各种条件下返回预期的上下文 (context)，`save_context` 正确存储数据，以及 `clear` 按预期运行。测试诸如空历史记录或最大历史记录限制等边缘情况。

开发自定义内存模块为您的 LangChain 应用程序提供很大程度的控制和定制。通过理解核心接口并精心设计您的存储和检索逻辑，您可以构建精巧的对话系统，能够以与您的应用独特需求完美契合的方式管理上下文。

## 参考资料

- [Custom Memory](https://python.langchain.com/docs/modules/memory/how_to/custom_memory) — LangChain Developers (2025)
  Publisher: LangChain Inc.
  LangChain中实现自定义内存模块的官方指南，详细介绍了所需的接口和方法。
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (2020)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2020); DOI: [10.48550/arXiv.2005.11401](https://doi.org/10.48550/arXiv.2005.11401)
  介绍了检索增强生成（RAG），这是一种整合外部知识和上下文管理的方法，自定义内存模块可以支持此方法。
- [Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/) — Martin Kleppmann (2017)
  Publisher: O'Reilly Media
  关于设计健壮、可扩展和可维护系统的综合指南，涵盖了与生产级内存模块相关的持久化、状态管理和异步操作。
- [Recent Advances in Deep Learning Based Dialogue Systems](https://ieeexplore.ieee.org/document/8966532) — Zhaojun Li, Wenjun Mao (2020)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 42; Pages: 1544-1560; DOI: [10.1109/TPAMI.2020.2968805](https://doi.org/10.1109/TPAMI.2020.2968805)
  一项探讨对话系统各种组件和挑战的调查，包括内存和上下文管理，有助于更广泛地理解自定义内存所解决的问题领域。
