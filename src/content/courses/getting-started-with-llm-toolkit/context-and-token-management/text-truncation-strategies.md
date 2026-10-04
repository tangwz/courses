---
course: "getting-started-with-llm-toolkit"
chapter: "context-and-token-management"
lesson: "text-truncation-strategies"
sourceId: 7761
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-3-context-and-token-management/text-truncation-strategies"
title: "文本截断的策略"
description: "了解将文本截断以适应模型上下文窗口的不同方法。"
order: 3
plots: []
sourceHash: "4501076d076dd8f6b81958bded2bb2e31af665889b2be7457cb5379b938bb374"
sourceCorrections: []
---

确保提示不超出大型语言模型的上下文 (context)窗口以避免错误，这是一项核心工作。文本截断是实现这一点的最直接方法，它涉及到缩短内容以适应特定的令牌限制。尽管操作简单，这项技术却是管理大型语言模型输入的基本组成部分，尤其是在处理长篇文档或大量对话历史时。

截断操作的主要权衡是信息丢失。通过裁剪文本的一部分，您有可能移除模型所需的上下文信息。因此，选择正确的截断策略对于保持应用程序输出的质量影响很大。

### 保留文本开头

最常见的截断策略是保留文本的开头并裁剪末尾。这通常是默认做法，因为引言、摘要和开篇段落通常包含最核心、概括性的信息。

`truncate_to_token_limit` 函数提供了一种简单的方法来实现这一点。您需要指定文本、最大令牌数量以及您正在使用的分词 (tokenization)器 (tokenizer)。

```python
from kerb.tokenizer import truncate_to_token_limit, count_tokens, Tokenizer

long_text = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a long piece of text that needs to be truncated to fit within "
    "a specific token limit. We want to preserve the beginning of the text "
    "because it usually contains the most important information in many contexts."
)

original_tokens = count_tokens(long_text, tokenizer=Tokenizer.CL100K_BASE)
print(f"Original tokens: {original_tokens}")

# 截断至最多20个令牌
truncated_text = truncate_to_token_limit(
    long_text,
    max_tokens=20,
    tokenizer=Tokenizer.CL100K_BASE
)

truncated_tokens = count_tokens(truncated_text, tokenizer=Tokenizer.CL100K_BASE)
print(f"Truncated text: '{truncated_text}'")
print(f"Truncated tokens: {truncated_tokens}")
```

正如您所看到的，该函数会缩短文本以符合 `max_tokens` 限制，默认添加省略号 `...` 以表明内容已被移除。结果文本的令牌数量保证小于或等于指定限制。

### 保留文本末尾

有时，最核心的信息位于文档的末尾。例如，在对话日志中，最新消息最相关；在错误日志中，最后几行常包含问题的根本原因。在这些情况下，您会希望从文本的开头进行截断，保留末尾。

您可以通过将 `preserve_end` 参数 (parameter)设置为 `True` 来实现此目的。

```python
log_entry = (
    "2024-10-15 14:30:22 INFO Processing started for batch_id=12345 "
    "with 150 items. System load: 45%. Memory usage: 2.3GB. "
    "Previous batches completed successfully. "
    "ERROR: Failed to process item_id=67890 due to invalid format. "
    "Status: FAILED. Error code: E404."
)

original_tokens = count_tokens(log_entry, tokenizer=Tokenizer.CL100K_BASE)
print(f"Original log tokens: {original_tokens}")

# 截断至25个令牌，保留末尾
truncated_log = truncate_to_token_limit(
    log_entry,
    max_tokens=25,
    tokenizer=Tokenizer.CL100K_BASE,
    preserve_end=True
)

truncated_tokens = count_tokens(truncated_log, tokenizer=Tokenizer.CL100K_BASE)
print(f"Truncated log: '{truncated_log}'")
print(f"Truncated tokens: {truncated_tokens}")
```

这种方法正确地保留了日志末尾的核心错误消息，为模型提供了最有用的上下文 (context)信息，而默认截断则会完全丢失这些信息。

### 自定义截断指示器

默认的省略号 `...` 是一个明确的截断信号，但您可以根据不同需求进行定制。例如，在面向用户的应用程序中，使用 `[...内容已截断...]` 这样更具描述性的指示器可能更好。在代码摘要工具中，您可能会使用 `# ... 代码其余部分` 这样的注释。

`ellipsis` 参数 (parameter)允许您定义一个自定义字符串作为截断标记 (token)。

```python
documentation = (
    "This function takes a list of integers as input and returns the sum. "
    "The implementation uses a simple loop to iterate through all elements. "
    "Time complexity is O(n) where n is the length of the input list."
)

# 使用自定义指示器截断
truncated_docs = truncate_to_token_limit(
    documentation,
    max_tokens=15,
    tokenizer=Tokenizer.CL100K_BASE,
    ellipsis=" [truncated for brevity]"
)

print(f"Original text: '{documentation}'")
print(f"Custom ellipsis: '{truncated_docs}'")
```

请记住，`ellipsis` 字符串本身会占用令牌，函数会考虑到这一点，以确保最终输出遵守 `max_tokens` 限制。

### 简单截断的局限性

尽管有效且易于实施，简单截断有一个主要缺点：它不加区分。它不理解自己正在删除的内容。无论您保留开头还是末尾，文档中间的核心信息都会丢失。

例如，如果您的文档包含引言、中间的核心发现和结论，简单截断会迫使您在引言和结论之间做出选择，而该发现总是被丢弃。

更高级的技术可以解决这个问题。例如，一些系统实现了“中间移除”式截断，它保留文档的开头和结尾，同时移除中间内容。另一种方法是使用大型语言模型来总结文本，这是一种智能的、上下文 (context)感知截断。这些方法更复杂，但在处理结构化文档时可以带来更好的结果。在关于数据准备和检索的后续章节中，我们将讨论分块（chunking），它提供了另一种处理大段文本的方式，即将它们分解成更小、连贯的片段，而不是简单地裁剪。

## 参考资料

- [How to get tokens](https://platform.openai.com/docs/guides/tokenization) — OpenAI (2024)
  Publisher: OpenAI
  提供LLM如何对文本进行分词的基础知识，解释上下文窗口限制和代币计数，这使得截断策略成为必要。
