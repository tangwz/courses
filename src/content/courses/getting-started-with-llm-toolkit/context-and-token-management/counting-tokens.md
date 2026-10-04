---
course: "getting-started-with-llm-toolkit"
chapter: "context-and-token-management"
lesson: "counting-tokens"
sourceId: 7760
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-llm-toolkit/chapter-3-context-and-token-management/counting-tokens"
title: "使用分词器统计Token数量"
description: "使用分词器模块准确统计给定文本在不同模型下的Token数量。"
order: 2
plots: []
sourceHash: "199967e2c574311faa1ca30ef87e27588087dea90bd0e3f584dd7a054e7de5df"
sourceCorrections: []
---

要有效管理大型语言模型（LLM）的上下文 (context)窗口，首先你需要一个精确的方法来衡量输入的大小。我们通常以词或字符来思考，但大型语言模型（LLM）使用的单位不同：**Token**。Token是词语的一部分，它可以是一个完整的词、一个子词 (subword)，甚至只是一个字符或标点符号。例如，单词“tokenization”可能会被拆分成两个Token：“token”和“ization”。

理解并准确统计Token对于构建可靠的大型语言模型（LLM）应用来说是基础。这能帮助你避免超出模型的上下文限制（否则会导致API错误），并有助于你管理成本，因为成本与输入和输出Token的数量直接相关。

`tokenizer`模块提供了`count_tokens`函数，这是一个计算任何给定文本Token数量的便捷功能。

### 使用count\_tokens

获取Token数量最直接的方法是将你的文本传递给`count_tokens`函数。指定要使用的分词 (tokenization)器 (tokenizer)是必需的，因为不同的模型和模型系列使用不同的分词方案。`Tokenizer`枚举类型提供了访问常用分词器的方法。对于像GPT-4和GPT-3.5-Turbo这样的现代OpenAI模型，`CL100K_BASE`是正确的选择。

```python
from kerb.tokenizer import count_tokens, Tokenizer

text = "This is a sample sentence for token counting."

token_count = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)

print(f"文本: '{text}'")
print(f"字符数: {len(text)}")
print(f"Token数量: {token_count}")
```

请注意，Token数量与词数不同。Token与字符或词的这种比例会根据文本的结构、语言和复杂程度而变化。

### 选择正确的分词 (tokenization)器 (tokenizer)

为你的目标模型使用正确的分词器对准确性有很大影响。不匹配可能导致错误的计数，引发意料之外的错误或更高的成本。`Tokenizer`枚举类型包含针对不同模型系列的多个选项。例如，`P50K_BASE`被较旧的OpenAI模型和一些专门用于代码的模型使用。

我们来比较一下使用不同分词器时同一文本的Token数量。

```python
from kerb.tokenizer import count_tokens, Tokenizer

text = "Large language models have revolutionized AI."

# 适用于GPT-4, GPT-3.5-Turbo
tokens_cl100k = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
print(f"CL100K_BASE 计数: {tokens_cl100k}")

# 适用于较旧的代码模型
tokens_p50k = count_tokens(text, tokenizer=Tokenizer.P50K_BASE)
print(f"P50K_BASE 计数: {tokens_p50k}")
```

即使是这个短句，Token数量也不同。虽然差异看起来可能不大，但对于大型文档来说，它可能会累积到数百或数千个Token，影响性能和成本。通常来说，始终将分词器与你正在使用的模型相匹配。

### 快速近似估算

官方分词 (tokenization)器 (tokenizer)提供完美准确性，但需要加载特定的编码文件。在需要快速、轻量级估算且不依赖外部组件的场景中，你可以使用近似方法。该库为此提供了几种启发式方法。一个常见的方法是`CHAR_4`，它根据一条规则来估算Token：对于典型的英文文本，一个Token大约是四个字符。

```python
from kerb.tokenizer import count_tokens, Tokenizer

text = "This is a long sentence used for demonstrating the trade-offs between accurate tokenization and fast approximation methods for quick checks."

# 准确计数
accurate_count = count_tokens(text, tokenizer=Tokenizer.CL100K_BASE)
print(f"准确计数 (CL100K_BASE): {accurate_count}")

# 快速近似估算
approx_count = count_tokens(text, tokenizer=Tokenizer.CHAR_4)
print(f"近似计数 (CHAR_4): {approx_count}")
```

近似方法对于客户端验证或需要粗略估算的吞吐量 (throughput)高的系统很有用，但在API调用之前的最终验证中，始终建议使用模型特定的分词器。

### 检查API调用

我们将其应用于一个情景。在向大型语言模型（LLM）发送请求之前，你可以检查你组合后的提示是否符合模型的上下文 (context)窗口限制。这个预防性措施使你的应用更可靠。

假设你正在构建一个使用具有4096 Token上下文窗口的模型开发的聊天机器人。你决定为模型的响应保留1000个Token，为你的输入（包括系统提示和用户消息）留下3096个Token。

```python
from kerb.tokenizer import count_tokens, Tokenizer

# 应用设置
CONTEXT_LIMIT = 4096
RESERVED_FOR_COMPLETION = 1000
INPUT_TOKEN_BUDGET = CONTEXT_LIMIT - RESERVED_FOR_COMPLETION

# 提示
system_prompt = "You are a helpful assistant that explains complex topics simply."
user_input = "Can you provide a detailed explanation of Retrieval-Augmented Generation (RAG) and how it helps reduce hallucinations in large language models?"

# 计算提示各部分的Token数量
system_tokens = count_tokens(system_prompt, tokenizer=Tokenizer.CL100K_BASE)
user_tokens = count_tokens(user_input, tokenizer=Tokenizer.CL100K_BASE)
total_input_tokens = system_tokens + user_tokens

print(f"系统提示Token: {system_tokens}")
print(f"用户输入Token: {user_tokens}")
print(f"总输入Token: {total_input_tokens}")
print(f"输入Token预算: {INPUT_TOKEN_BUDGET}")

if total_input_tokens > INPUT_TOKEN_BUDGET:
    print("\n警告：输入超出可用的Token预算！")
else:
    print("\n好的：输入符合可用的Token预算。")
```

通过执行此检查，你可以决定是继续进行API调用，还是先应用截断策略，我们将在下一节中介绍此策略。这种验证是构建可靠且经济高效的大型语言模型（LLM）应用的组成部分。

## 参考资料

- [tiktoken (GitHub repository)](https://github.com/openai/tiktoken) — OpenAI (2024)
  OpenAI的官方分词器库，为GPT-4和GPT-3.5-Turbo等模型提供编码，包括`cl100k_base`和`p50k_base`编码。
- [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/) — Rico Sennrich, Barry Haddow, Alexandra Birch (2016)
  Journal: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers); Publisher: Association for Computational Linguistics; Volume: 1; Pages: 1715–1725; DOI: [10.18653/v1/P16-1162](https://doi.org/10.18653/v1/P16-1162)
  引入字节对编码（BPE）的原始论文，这是一种将单词分割成子词单元的方法，是许多现代分词器的基础。
- [Hugging Face \`tokenizers\` library](https://huggingface.co/docs/tokenizers/index) — Hugging Face (2024)
  Publisher: Hugging Face
  关于变压器模型中使用的各种分词器（包括BPE和WordPiece）原理和实现的通用资源。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://aclanthology.org/D18-2012/) — Taku Kudo, John Richardson (2018)
  Journal: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations; Publisher: Association for Computational Linguistics; Pages: 66-71; DOI: [10.18653/v1/D18-2012](https://doi.org/10.18653/v1/D18-2012)
  描述了SentencePiece，一种在许多变压器模型中使用的语言无关的子词分词方法。
