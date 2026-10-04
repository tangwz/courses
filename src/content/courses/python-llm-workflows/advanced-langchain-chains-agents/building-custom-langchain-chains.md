---
course: "python-llm-workflows"
chapter: "advanced-langchain-chains-agents"
lesson: "building-custom-langchain-chains"
sourceId: 5154
sourceUrl: "https://apxml.com/zh/courses/python-llm-workflows/chapter-5-advanced-langchain-chains-agents/building-custom-langchain-chains"
title: "构建自定义链"
description: "根据特定工作流需求开发自己的自定义链逻辑。"
order: 2
plots: []
sourceHash: "5f624c46ee9ae62fc20b46b5c24ae7b38ef3f1d0be2f66b97e9babab29b5050c"
sourceCorrections: []
---

尽管LangChain提供了诸如`LLMChain`和`SequentialChain`等几种有用的预设链，但你经常会遇到应用程序需要更符合需求的系列操作或连接组件的特定逻辑的情况。此时，构建自定义链就变得必要。你可能需要：

- 在LLM调用之间整合自定义数据转换步骤。
- 根据中间结果实现条件逻辑。
- 以特定方式组合来自多个组件的输出。
- 创建带有非线性数据流的链。

在现代LangChain中，构建自定义链最灵活和推荐的方式是使用**LangChain表达式语言（LCEL）**。LCEL提供了一种声明式方法，通过反映数据流的语法将不同组件组合起来。

### 使用LangChain表达式语言（LCEL）进行组合

LCEL围绕`Runnable`协议构建。大多数LangChain组件，例如LLM封装器、提示模板和输出解析器，都实现了此协议。LCEL中的基本操作是通过管道（`|`）将组件连接起来。当你将组件A通过管道输入到组件B（`A | B`）时，A的输出会自动作为B的输入传递。

让我们用一个建立在第四章原理之上的简单例子来说明。假设我们想要一个链，它接收一个主题，生成一个关于该主题的技术问题，然后使用另一次LLM调用对生成的那个问题提供简洁的回答。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 假设OPENAI_API_KEY已在环境变量中设置
# 初始化模型
model = ChatOpenAI(model="gpt-3.5-turbo")
qa_model = ChatOpenAI(model="gpt-4") # 使用可能更强大的模型进行问答

# 1. 生成问题的提示
prompt_generate_question = ChatPromptTemplate.from_template(
    "生成一个关于主题的简洁技术问题：{topic}"
)

# 2. 回答生成问题的提示
prompt_answer_question = ChatPromptTemplate.from_template(
    "请对以下问题提供简明清晰的回答：\n"
    "问题：{generated_question}\n"
    "回答："
)

# 3. 使用LCEL定义自定义链
#    第二个提示需要原始主题和生成的问题。
#    RunnablePassthrough会将输入{'topic': '...'}传递下去。
#    RunnableParallel会执行分支并将结果收集到字典中。

chain = (
    {"generated_question": prompt_generate_question | model | StrOutputParser(),
     "topic": RunnablePassthrough()} # 传递原始主题
    | prompt_answer_question
    | qa_model
    | StrOutputParser()
)

# 链的输入
topic_input = {"topic": "vector databases"}

# 调用链
result = chain.invoke(topic_input)

print(f"Topic: {topic_input['topic']}")
# 示例输出（会有所不同）：
# 生成的问题（中间结果，非链直接输出）：向量数据库用于相似性搜索的主要机制是什么？
print(f"Generated Answer:\n{result}")
# 示例输出（会有所不同）：
# 生成的回答：
# 向量数据库用于相似性搜索的主要机制是近似最近邻（ANN）算法。这些算法能高效地在高维空间中找到与给定查询向量最接近（最相似）的向量，而无需将查询与数据库中的每个向量进行比较。
```

在这个例子中：

1. 我们定义了两个独立的提示模板和LLM实例。
2. 我们使用`RunnableParallel`（由字典结构`{...}`表示）来运行两个分支：
   - 第一个分支（`prompt_generate_question | model | StrOutputParser()`）基于输入`topic`生成问题。
   - 第二个分支（`RunnablePassthrough()`）简单地将原始输入字典（包含`topic`）传递下去。
3. 这些并行分支的结果被收集到一个字典中：`{'generated_question': '...', 'topic': '...'}`。
4. 这个字典随后通过管道（`|`）传入`prompt_answer_question`，它使用这两个键。
5. 格式化的提示传递给`qa_model`，其输出由`StrOutputParser`解析。

> 使用LCEL结合问题生成和回答的自定义链的数据流。

### 整合自定义Python函数

LCEL链不限于LangChain组件。你可以整合你自己的Python函数。LangChain会自动将LCEL序列中传递的标准Python函数封装成`RunnableLambda`。

让我们对之前的例子稍作修改。假设在生成问题后，我们想在将其传递给回答提示之前，使用一个Python函数添加一个自定义前缀。

```python
# ... （导入和模型定义保持不变） ...

# 自定义Python函数
def add_question_prefix(question_text):
  print(f"--- Applying custom prefix ---") # 仅作演示
  return f"Technical Query: {question_text}"

# 使用自定义函数定义链
chain_with_custom_func = (
    {"intermediate_question": prompt_generate_question | model | StrOutputParser(),
     "topic": RunnablePassthrough()}
    | RunnableParallel(
        generated_question=lambda x: add_question_prefix(x["intermediate_question"]), # 应用函数
        topic=lambda x: x["topic"] # 传递主题
      )
    | prompt_answer_question # 期望输入'generated_question'和'topic'
    | qa_model
    | StrOutputParser()
)

# 调用修改后的链
result_custom = chain_with_custom_func.invoke(topic_input)

print(f"Topic: {topic_input['topic']}")
# 示例输出（会有所不同）：
# --- Applying custom prefix ---
print(f"Generated Answer (with custom func):\n{result_custom}")
# 示例输出（会有所不同）：
# 生成的回答（带自定义函数）：
# 向量数据库用于相似性搜索的主要机制是近似最近邻（ANN）算法。这些算法能高效地在高维空间中找到与给定查询向量最接近（最相似）的向量，通常使用哈希或基于图的索引等技术，而无需穷尽地与每个条目进行比较。
```

在这里，我们在`RunnableParallel`中使用了一个`lambda`函数来调用`add_question_prefix`函数。LangChain在后台处理将其封装成`RunnableLambda`，使得整合过程很流畅。

### 旧方法：继承`Chain`

在LCEL成为标准之前，创建自定义链的主要方式是继承`langchain.chains.base.Chain`。这包括：

1. 定义`input_keys`（一个预期输入变量名的列表）。
2. 定义`output_keys`（一个链将输出的变量名列表）。
3. 实现`_call`方法，它包含链的核心逻辑，接收输入并返回一个输出字典。

尽管仍然可用，但对于大多数常见的工作流模式而言，继承通常比使用LCEL更冗长且组合性更差。对于需要复杂内部状态管理或其逻辑不易被LCEL轻松建模为顺序或并行流的链，它可能仍然可以考虑。然而，对于构建大多数自定义工作流，LCEL提供了一种更优雅且易于维护的解决方案。

构建自定义链，特别是使用LCEL，使你能够对LLM工作流进行细致控制，从而构建精确满足需求的复杂应用程序。这种可组合性是突破简单提示-响应交互时的一个显著优势。

## 参考资料

- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/) — LangChain Contributors (2024)
  Publisher: LangChain
  LangChain 表达式语言的官方文档，详细介绍了其核心概念、`Runnable` 协议以及组合自定义 LLM 链的方法。
- [Hands-On Large Language Models with LangChain: Build real-world LLM applications with LangChain, OpenAI, and Hugging Face](https://www.packtpub.com/product/hands-on-large-language-models-with-langchain/9781832402120) — Subhash Patankar (2023)
  Publisher: Packt Publishing
  一本使用 LangChain 构建真实世界 LLM 应用的实用指南，涵盖了构建复杂工作流和自定义链的高级技术。
