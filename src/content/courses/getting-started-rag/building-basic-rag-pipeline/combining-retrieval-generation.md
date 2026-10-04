---
course: "getting-started-rag"
chapter: "building-basic-rag-pipeline"
lesson: "combining-retrieval-generation"
sourceId: 3662
sourceUrl: "https://apxml.com/zh/courses/getting-started-rag/chapter-5-building-basic-rag-pipeline/combining-retrieval-generation"
title: "结合检索与生成"
description: "演示如何将检索器和生成器组件连接起来处理查询。"
order: 5
plots: []
sourceHash: "49fc9a5bbfb2e4b94b4add3997e6b462bc4113b141c8df8552d2cf2a1a1e8e76"
sourceCorrections: []
---

将用于获取相关信息的检索器组件与用于生成文本的生成器（LLM）连接起来，构成了RAG流程的主要逻辑。根据查询检索到的信息会直接影响LLM生成的最终输出。

基本流程包括接收用户的输入查询，使用它来检索上下文 (context)，然后将原始查询和检索到的上下文一起以精心设计的提示词 (prompt)形式呈现给LLM。

### RAG流程顺序

我们来图示化信息在结合检索与生成时的典型流动方式：

> RAG系统中的标准数据流。用户查询启动检索，检索到的上下文 (context)被添加到查询中，形成一个增强型提示词 (prompt)，然后由生成器LLM处理。

### 实现连接

其核心是，结合检索与生成涉及一系列函数调用。您通常会创建一个函数或方法来协调这个流程。假设您拥有：

1. 一个`retriever`对象，其中包含一个方法，例如`search(query: str) -> list[str]`，返回相关文本块列表的。
2. 一个`generator`对象（代表LLM接口），其中包含一个方法，例如`generate(prompt: str) -> str`，接受提示词 (prompt)并返回生成文本的。

这种组合逻辑在Python中大致如下所示：

```python
# 假设 'retriever' 和 'generator' 对象已按先前章节所述初始化。

def create_augmented_prompt(query: str, context_chunks: list[str]) -> str:
    """
    格式化提示词字符串，使其包含查询和检索到的上下文。
    """
    # 简单的拼接策略
    context = "\n\n".join(context_chunks)

    # 示例提示词模板
    prompt = f"""请根据以下上下文回答查询。如果上下文不包含答案，请说明。

上下文：
{context}

查询：{query}

答案："""
    return prompt

def execute_rag_pipeline(query: str) -> str:
    """
    运行查询，经过检索和生成步骤。
    """
    # 1. 检索相关上下文
    try:
        retrieved_chunks = retriever.search(query)
        if not retrieved_chunks:
            # 处理未找到相关文本块的情况
            # 选项1：返回特定消息
            # return "我未能找到相关信息来回答您的查询。"
            # 选项2：在没有上下文的情况下继续（回退到标准LLM行为）
            retrieved_chunks = []
            print("警告：未找到相关上下文。") # 或者使用日志记录

    except Exception as e:
        print(f"检索时出错：{e}")
        # 适当处理检索错误
        return "信息检索过程中发生错误。"

    # 2. 构建增强型提示词
    augmented_prompt = create_augmented_prompt(query, retrieved_chunks)

    # 可选：检查提示词长度是否超过模型限制
    # （实现取决于具体的LLM和分词器）
    # if len(tokenizer.encode(augmented_prompt)) > MAX_CONTEXT_LENGTH:
    #     # 处理上下文溢出（例如，截断上下文，使用不同的策略）
    #     print("警告：提示词超出最大长度。可能发生截断。")
    #     # 如有必要，在此添加缩短提示词的逻辑

    # 3. 使用LLM生成回复
    try:
        final_response = generator.generate(augmented_prompt)
    except Exception as e:
        print(f"生成时出错：{e}")
        # 处理生成错误
        return "生成回复时发生错误。"

    return final_response

# 示例用法：
user_query = "与微调相比，使用RAG的主要好处是什么？"
response = execute_rag_pipeline(user_query)
print(f"查询：{user_query}")
print(f"回复：{response}")
```

### 组合步骤中的考量

- **提示词 (prompt)格式：** 正如第4章（“生成组件与增强”）中讨论的，您构建提示词的方式（示例中的`create_augmented_prompt`函数）非常重要。它指导LLM如何使用所提供的上下文 (context)。示例使用了简单的模板，但更复杂的模板可能会指示LLM引用来源或处理冲突信息。
- **上下文处理：** 当检索器返回多个文本块时，您需要一个策略来组合它们。简单的拼接很常见，但请注意LLM的上下文窗口限制。如果组合后的上下文过长，您可能需要截断它，仅选择前k个最相关的文本块，或者使用更高级的摘要或过滤技术。
- **错误处理：** 对检索和生成步骤实施强大的错误处理。网络问题、API错误或数据处理问题都可能发生。决定在这种情况下流程应如何运作（例如，返回错误消息，回退到非增强型生成）。
- **框架抽象：** 虽然本节说明了手动组合逻辑，请记住，像LangChain或LlamaIndex这样的框架（在`overview-rag-frameworks`章节中提到）通常提供更高层次的抽象，例如“链”（Chains）或“查询引擎”（Query Engines）。这些抽象封装了检索-增强-生成序列，通过自动处理提示词格式、组件连接以及有时上下文管理，大大简化了实现。然而，理解这里展示的底层序列对于调试和定制很重要。

通过连接检索器和生成器，您为外部知识告知LLM的输出创造了途径，从独立的组件转变为一个功能性的RAG系统。下一节将着重讨论如何通过这个组装好的流程运行查询。

## 参考资料

- [Prompt engineering](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api) — OpenAI (2024)
  Publisher: OpenAI
  提供构建大型语言模型有效提示的指南，包括如何提供外部信息以提升生成质量的策略。
- [Retrieval Augmented Generation (RAG)](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQH7U8-F2s36plp0PqtPBEjxZJfJqDN8U-aiuexPld9dEY4J_jVnw1H6f89tSW2LKM3QPQbXLvo0p1ttlaXO79JiopqCr7JpNQzwBoTUVWuZMC8CuinDcNvtYsaFm5ETBmG032OfZbjay5Ws) — LangChain (2024)
  Publisher: LangChain
  官方文档，说明如何使用 LangChain 框架实现 RAG 流水线，包括检索器和生成器的集成。
