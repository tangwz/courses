---
course: "fine-tuning-small-language-model"
chapter: "data-preparation-and-formatting"
lesson: "formatting-prompts-specific-architectures"
sourceId: 8505
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-small-language-model/chapter-2-data-preparation-and-formatting/formatting-prompts-specific-architectures"
title: "针对特定架构格式化提示词"
description: "调整数据集格式，以匹配不同模型架构所需的特定提示词模板。"
order: 4
plots: []
sourceHash: "cd13ea52527b0d321817a55264dfaab4fe8ff6ec67c1397d0eeb58f1717c3236"
sourceCorrections: []
---

在为小语言模型准备数据集时，仅仅提供纯文本或通用的 JSON 是不够的。基座模型是在非结构化文本上训练的，但经过指令微调 (fine-tuning)的模型则需要高度特定的对话结构。这种结构由特殊的词表标记（token）组成，用于分隔交互中的不同部分。如果使用模型最初训练时未采用的格式进行微调，损失函数 (loss function)将难以收敛，导致生成的文本质量大幅下降。

提示词 (prompt)模板充当了交互的结构蓝图。它们告诉模型谁在说话、系统规则是什么，以及什么时候该生成回复。标准的交互通常包括系统消息、用户提示词和预期的助手回答。为了区分这些角色，各种架构依赖于在最初预训练 (pre-training)阶段添加到其词表中的特殊控制标记。

> 将原始数据集交互映射到特定架构的提示词结构。

### 常见的架构格式

随着时间推移，不同的研究团队开发了不同的格式规范。在流水线中更换基座模型时，理解这些差异非常必要。

**Alpaca 格式**
Alpaca 格式出现在指令微调 (fine-tuning)的早期阶段。它依赖于标准的类 Markdown 标题，而不是专门的词表标记 (token)。虽然出现较早，但它目前仍常用于许多开源权重 (weight)模型。其形式通常如下：

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Write a Python script to add two numbers.
### Response:
```

**ChatML 格式**
聊天标记语言（ChatML）被 Qwen 和多种 Mistral 变体等现代模型所采用。它通过引入特殊标记来明确定义角色边界，从而降低模型幻觉 (hallucination)出用户输入的可能性。

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
```

**Llama 格式**
Meta 为其 Llama 系列实现了一套独特的结构。对于 Llama 2，它使用严格的方括号标签，如 `[INST]` 和 `<<SYS>>`。

```text
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>
Hello! [/INST]
```

### 通过编程方式应用模板

手动为这些格式拼接字符串很容易出错。漏掉一个空格或序列结束标记 (token)都会严重影响训练效果。Hugging Face Transformers 库提供了一个内置方法 `apply_chat_template`，它直接绑定在分词 (tokenization)器（tokenizer）上。

无需编写自定义的字符串处理脚本，你只需要将数据组织成字典列表即可。每个字典包含一个特定的角色和相应的文本内容。

```python
messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Explain backpropagation."}
]
```

当你将此列表传递给 `tokenizer.apply_chat_template(messages, tokenize=False)` 时，分词器会自动将通用字典映射到特定模型所需的精确字符串格式。设置 `tokenize=False` 参数 (parameter)可以确保输出保持为人类可读的字符串，而不是张量索引数组。建议在初始数据准备阶段这样做，以便在将数据传递给神经网络 (neural network)之前视觉检查格式是否正确。

为了准备监督式微调 (fine-tuning)的数据，助手的回复也必须包含在消息列表中。模型会学习生成紧跟在助手提示标记之后的精确文本序列。

```python
training_messages = [
    {"role": "system", "content": "You are a helpful AI."},
    {"role": "user", "content": "Explain backpropagation."},
    {"role": "assistant", "content": "Backpropagation is an algorithm used to train neural networks..."}
]

formatted_prompt = tokenizer.apply_chat_template(training_messages, tokenize=False)
```

这种方法实现了原始数据集与特定模型架构的完全解耦。你可以使用格式为标准 JSON 角色的数据集，今天微调 Alpaca 风格的模型，明天换成 ChatML 模型，而无需改写任何数据处理代码。

### 格式化对数学计算的影响

在为微调 (fine-tuning)格式化提示词 (prompt)时，排列方式会直接影响损失计算。如果整个序列长度为 $T$，单个标记 (token)为 $y_1, y_2, ..., y_T$，我们只希望模型根据生成的助手回复来更新权重 (weight)。对于系统和用户提示词标记，损失函数 (loss function)会被严格遮蔽（masked）。

如果提示词模板格式不正确，注意力掩码（attention mask）就会对齐 (alignment)错误。交叉熵损失函数随后会根据系统或用户提示词标记计算误差。这会导致模型更新权重去预测用户的问题而不是自己的回答，从而严重损害对话性能。

### 处理序列结束标记 (token)

在文本生成过程中，模型需要一个特定的信号来停止输出。这由序列结束（EOS）标记控制。在为训练格式化提示词 (prompt)时，必须确保 EOS 标记附加在助手回复的最末尾。

如果遗漏了这个标记，微调 (fine-tuning)后的模型在推理 (inference)过程中可能会无休止地啰嗦，因为它从未学过停止的统计概率。如果列表中的最后一个字典属于助手，`apply_chat_template` 方法会自动处理 EOS 标记的添加。但是，如果你不得不为自定义架构手动构建模板，则必须显式地将 `tokenizer.eos_token` 附加到最终的训练字符串中。

## 参考资料

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2307.09288](https://doi.org/10.48550/arXiv.2307.09288)
  该技术报告详细介绍了 Llama 2 架构，包括用于指令微调的具体提示词结构和特殊标记。
- [Chat Templating](https://huggingface.co/docs/transformers/main/en/chat_templating) — Hugging Face (2024)
  Publisher: Hugging Face
  apply_chat_template 方法的官方文档，说明了如何通过编程方式为不同的模型架构格式化对话数据。
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto (2023)
  Publisher: Stanford Center for Research on Foundation Models (CRFM)
  最初的项目文档，该项目推广了用于开源模型微调的 Alpaca 指令格式。
- [Training language models to follow instructions with human feedback](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) — Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan J. Lowe (2022)
  Journal: Advances in Neural Information Processing Systems; Volume: 35; Pages: 27730-27744
  指令微调的基础研究，确立了模型训练中系统、用户和助手角色的范式。
