# LangChain 表达式语言 (LCEL) 内部机制

来源：[原文](https://apxml.com/zh/courses/langchain-production-llm/chapter-1-advanced-langchain-architecture/lcel-internals)

[返回章节目录](README.md) · [返回课程目录](../README.md)

管道运算符 `|` 是连接 LangChain 组件的常用方法。然而，构建生产就绪的应用需要对驱动这种组合的 LangChain 表达式语言 (LCEL) 有透彻的理解。LCEL 不仅仅是语法糖；它是一种声明式定义计算图的方式，为组件提供统一接口，并直接提供流式处理、异步操作和并行执行等功能。了解 LCEL 背后的主要原理将帮助您在复杂架构中发挥其最大作用。

### Runnable 协议：一个统一接口

LCEL 的核心是 `Runnable` 协议。任何旨在成为 LCEL 链一部分的组件，无论是提示模板、LLM、输出解析器，还是自定义函数，都遵循此统一接口。这种一致性使得不同组件能够相互连接。`Runnable` 协议定义了几个方法，其中最主要的有：

- `invoke(input)`: 使用给定输入同步执行组件。
- `ainvoke(input)`: 异步执行组件。对于生产环境 Web 服务器中类似 LLM 调用的 I/O 密集型操作非常重要。
- `stream(input)`: 同步执行组件，并在输出块可用时将其流式传回。这对于向用户提供 LLM 的增量响应尤其有用。
- `astream(input)`: 异步执行组件，流式传回输出块。结合了异步执行和流式处理的优势。
- `batch(inputs)`: 在输入列表上同步执行组件，可能包含优化。
- `abatch(inputs)`: 在输入列表上异步执行组件。

通过统一这些方法，LCEL 使得任何 `Runnable` 都能以一致的方式被调用、流式处理或批量处理，无论其内部实现如何。当您实现自定义组件时（本章后续会介绍），遵循 `Runnable` 协议会使它们成为 LangChain 生态系统中的一等公民。

### 组合：链即 RunnableSequence

当您使用管道运算符 (`|`) 连接两个 `Runnable` 组件时，例如 `prompt | model`，LCEL 会在内部构建一个 `RunnableSequence`。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 假设已设置 OPENAI_API_KEY
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()

# 这会创建一个 RunnableSequence 实例
chain = prompt | model
```

`RunnableSequence` 本身就是一个 `Runnable`。当您 `invoke` 这个序列时，它会按顺序执行其组成部分：

1. 输入 (`{"topic": "bears"}`) 被传递给第一个元素 (`prompt`) 的 `invoke` 方法。
2. 第一个元素（一个格式化的 `PromptValue`）的输出被传递给第二个元素 (`model`) 的 `invoke` 方法。
3. 第二个元素（一个 `AIMessage`）的输出成为序列的最终输出。

这种顺序执行同样适用于 `stream`、`astream`、`batch` 和 `abatch`。对于流式处理，LCEL 负责在序列中传递数据块。如果某个组件本身不支持流式处理（例如简单的提示模板），LCEL 会智能地将其完整输出传递到下游，从而使得后续支持流式处理的组件（如 LLM）能按预期运行。

### 使用 RunnableParallel 进行并行执行

LCEL 也支持 `Runnable` 的并行执行。这通常通过字典语法实现，该语法会创建一个 `RunnableParallel` 实例。

```python
from langchain_core.runnables import RunnableParallel

# 示例设置（请替换为实际的 runnable）
runnable1 = ChatPromptTemplate.from_template("Runnable 1: {input}") | model
runnable2 = ChatPromptTemplate.from_template("Runnable 2: {input}") | model

# 使用 RunnableParallel 定义并行执行
parallel_chain = RunnableParallel(
    output1=runnable1,
    output2=runnable2,
)

# 注意：在链中，您也可以使用字典字面量，LCEL 会
# 自动将其转换为 RunnableParallel：
# chain = {"output1": runnable1, "output2": runnable2} | some_next_step

# 调用此方法可能会并行运行 runnable1 和 runnable2
# 输入会被传递给 *所有* runnable
output = parallel_chain.invoke({"input": "parallel processing"})

# 输出将是一个字典：
# {'output1': AIMessage(...), 'output2': AIMessage(...)}
```

当一个 `RunnableParallel`（或链中使用的字典字面量）被调用时，它会将 *相同* 的输入传递给其包含的每个 `Runnable`。然后，它会尝试并发执行这些 `Runnable`，尤其是在使用 `ainvoke` 或 `abatch` 等异步方法时。最终输出是一个字典，其中的键对应于定义中提供的键，值则是相应 `Runnable` 的输出。

> LCEL 中顺序执行 (`RunnableSequence`) 和并行执行 (`RunnableParallel`) 模式的数据流比较。

理解这一区别对于设计高效的链条很有帮助。当一个步骤的输出是下一个步骤的输入时，使用 `RunnableSequence` (`|`)。当多个步骤可以独立且可能并发地处理相同的输入时，使用 `RunnableParallel` (`{}`)。

### 配置和流式处理机制

LCEL 在任何 `Runnable` 上提供 `.with_config()` 方法，用于传递回调、递归限制或运行名称等配置选项。这些配置通常会沿顺序或并行执行图传播。

```python
# 传递配置的示例
result = chain.with_config(
    {"run_name": "JokeGenerationRun"}
).invoke({"topic": "robots"})
```

流式处理支持是一个主要特性。当您在 LCEL 链上调用 `.stream()` 或 `.astream()` 时，框架会端到端地协调流式处理过程。如果 LLM 组件流式传输标记 (token)，LCEL 会使得这些标记在到达时即被生成。对于序列，中间的非流式处理步骤会完成其执行，其输出被传递到下一步，然后下一步可能会启动流式处理。为流式处理设计的输出解析器可以增量处理标记块。这种内置的流式处理能力对于用户期望即时反馈的交互式应用很有帮助。

### 对高级使用的意义

了解 LCEL 内部机制并非仅限于理论层面。它直接影响您进行以下工作的能力：

1. **高效调试：** 理解 `Runnable` 接口以及 `RunnableSequence` 和 `RunnableParallel` 的执行方式，有助于追踪数据流并找出错误或意外行为，尤其是在使用 LangSmith 等工具时（第五章会介绍）。
2. **提升表现：** 识别并行执行 (`RunnableParallel`) 的机会或运用异步方法 (`ainvoke`、`astream`、`abatch`)，使您能够构建响应更快、更具扩展性的应用（第六章会进一步说明）。
3. **实现自定义组件：** 当您需要内置 LangChain 组件之外的功能时，实现 `Runnable` 协议会使得您的自定义逻辑可以融入 LCEL 链中，从而继承流式处理和配置管理等功能。
4. **构建复杂逻辑：** 将顺序执行和并行执行与配置结合，能够在声明式且易于管理的框架中生成精密的、多步骤的推理 (inference)过程或代理行为。

LCEL 是复杂 LangChain 应用的架构支柱。通过把握其核心原则、`Runnable` 协议、通过 `RunnableSequence` 进行顺序组合、通过 `RunnableParallel` 进行并行执行、配置传播以及对流式处理和异步操作的内在支持，您将更有能力去设计、构建和调试成熟的 LLM 系统。

## 参考资料

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Developers (2024)
  Publisher: LangChain
  LCEL 的主要官方文档，详细介绍了其组件、组合方法以及流式传输和异步执行等核心功能。
- [Runnable Protocol in LangChain Core](https://api.python.langchain.com/en/latest/core/langchain_core.runnables.base.Runnable.html) — LangChain Developers (2024)
  Publisher: LangChain
  提供了 `Runnable` 协议的官方 API 参考，概述了其方法（如 `invoke`、`ainvoke`）以及它们如何构成 LCEL 组件的标准接口。
- [Asynchronous I/O (asyncio)](https://docs.python.org/3/library/asyncio.html) — Python Software Foundation (2024)
  Publisher: Python Software Foundation
  Python `asyncio` 模块的官方文档，为 LCEL 的异步方法提供了并发和异步操作的基础。

---

[下一节](02-%E5%BC%82%E6%AD%A5%E6%93%8D%E4%BD%9C%E4%B8%8E%E5%B9%B6%E5%8F%91.md)
