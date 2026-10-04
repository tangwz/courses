# SFT数据格式（提示词，回应）

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/sft-data-formatting)

[返回章节目录](README.md) · [返回课程目录](../README.md)

为了在监督微调 (fine-tuning)（SFT）过程中让模型有效学习，高质量的指令-回应数据集需要进行适当的格式化。SFT的目的是在给定特定输入（提示词 (prompt)）时，教会模型生成期望的输出（回应）。适当的格式化可确保模型理解任务结构，并将其学习重心放在生成正确的回应上。

### 基本提示词 (prompt)-回应结构

最简单地说，每个SFT示例包含两部分：

1. **提示词：** 提供给模型的输入，包含指令、背景信息或问题。
2. **回应：** 模型应生成的期望输出或答案。

模型在给定`提示词`后，会按顺序预测`回应`的标记 (token)。以一个直接的问答示例来说：

- **提示词：** "问题：马来西亚的首都是哪里？\n答案："
- **回应：** " 吉隆坡"

训练期间，模型会处理提示词并学习将其与目标回应关联起来。用于提示词的具体文本会因任务和期望的交互风格而有很大不同。它可能包含明确的指令、示例（在少量样本情况下），或对话历史。

### 针对不同任务类型的格式化

结构需要根据你希望模型学习的具体行为来调整。

#### 指令遵循

对于模型应遵循明确命令的任务，提示词 (prompt)会清楚地说明指令，通常后跟输入数据。

````text
# 示例 1：摘要
提示词: "总结以下文章：\n\n[文章文本在此...]\n\n总结："
回应: "[文章的简明摘要]"

# 示例 2：代码生成
提示词: "编写一个计算数字阶乘的Python函数。\n```python\n"
回应: "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n```"
````

#### 对话

为了训练对话代理，格式必须呈现对话的往复特性。这通常涉及使用特殊标记 (token)或分隔符来区分用户和助手的发言。

```text
# 使用特殊角色标记的对话示例格式
提示词: "<|USER|> 你好，能解释一下光合作用吗？\n<|ASSISTANT|>"
回应: " 光合作用是植物、藻类和蓝细菌将光能转化为化学能的过程..."

# 多轮对话示例格式
提示词: "<|USER|> 伦敦天气怎么样？\n<|ASSISTANT|> 伦敦现在多云，15°C。\n<|USER|> 明天呢？\n<|ASSISTANT|>"
回应: " 伦敦明天的预报是局部多云，最高气温18°C。"
```

使用`<|USER|>`和`<|ASSISTANT|>`等不同标记有助于模型学习对话结构并识别轮到谁发言。

#### 思维链推理 (inference)

对于需要推理的任务，回应本身可能包含得出最终答案的中间步骤。这会教导模型*如何*得到答案。

```text
提示词: "问题：约翰有5个苹果。他又买了3箱，每箱有4个苹果。他总共有多少个苹果？\n答案："
回应: " 约翰最初有5个苹果。他买了3箱 * 4个苹果/箱 = 12个苹果。总共，他有 5 + 12 = 17个苹果。最终答案是17。"
```

### 特殊标记 (token)和分隔符

为了帮助模型在连接的输入序列中清楚地区分提示词 (prompt)和回应，通常会使用特殊标记。这些可以是模型分词 (tokenization)器 (tokenizer)中预定义的标记（如 `[SEP]`、`</s>`、`<|endoftext|>`），也可以是专门为SFT添加的自定义标记（如 `<|PROMPT|>`、`<|COMPLETION|>`、`<|END_OF_TURN|>`）。

考虑一个使用通用标记的简化示例：

```python
import torch
from transformers import AutoTokenizer

# 假设分词器已加载
tokenizer = AutoTokenizer.from_pretrained("gpt2") # 示例分词器

# 如果需要，添加特殊标记（首先检查它们是否存在）
special_tokens_dict = {'sep_token': '<|SEP|>', 'pad_token': '<|PAD|>'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# 如果添加了新标记，请记住调整模型嵌入层的大小

prompt = "Translate to French: Hello world"
completion = " Bonjour le monde"

# 使用分隔符进行简单格式化
formatted_text = (
    f"{prompt}{tokenizer.sep_token}"
    f"{completion}{tokenizer.eos_token}"
)

# 对格式化文本进行分词
tokenized_input = tokenizer(formatted_text, return_tensors="pt")

print("Formatted Text:", formatted_text)
print("Token IDs:", tokenized_input['input_ids'])

# 输出可能如下所示（标记ID取决于具体分词器）：
# Formatted Text: Translate to French: Hello world<|SEP|>
# Bonjour le monde<|endoftext|>
# Token IDs: tensor([[ 14685,    284,  10607,    35,   995,  11858, 50257,
# 40195,   259,   813, 50256]])
```

分隔符的选择会影响模型在训练和推理 (inference)时如何划分输入。在整个数据集中一致地应用这些标记是很重要的。

### 损失计算的掩码处理

SFT训练的一个重要方面是确保损失*只*在回应标记 (token)上计算。模型应学习预测期望的输出，而不是预测它已作为输入收到的提示词 (prompt)本身。这通常通过使用标签掩码或修改注意力掩码来实现。

为模型准备批次数据时，输入ID将包含连接的提示词和回应标记。标签（损失函数 (loss function)的目标）通常是输入ID的右移版本。我们需要告知损失函数（例如CrossEntropyLoss）忽略为提示词标记计算的损失。

```python
import torch
import torch.nn.functional as F

# 假设 tokenized_input 包含提示词 + 分隔符 + 回应 + eos
# input_ids = tokenized_input['input_ids'] # 形状：[batch_size, sequence_length]

# 示例：单个序列的ID
# 提示词: "Q: Why sky blue? <|SEP|>" -> ID [10, 20, 30, 40, 50] (长度 5)
# 回应: " Scattering <|EOS|>" -> ID [60, 70, 80] (长度 3)
# 连接后: [10, 20, 30, 40, 50, 60, 70, 80] (长度 8)
input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])

# 标签通常是 input_ids 向右移动的版本
# 模型在每个位置预测*下一个*标记
# 标签 = [10, 20, 30, 40, 50, 60, 70, 80] -> 移动后
# -> [20, 30, 40, 50, 60, 70, 80, <PAD>]
# 或更常见的是: [-100, -100, -100, -100, -100, 60, 70, 80]
# 其中 -100 是 ignore_index
labels = torch.tensor([[-100, -100, -100, -100, -100, 60, 70, 80]]) # 掩盖提示词标记

# 假设 model_output 的形状为 [batch_size, sequence_length, vocab_size]
# 示例虚拟输出 logits
vocab_size = 100
sequence_length = input_ids.shape[1]
model_output_logits = torch.randn(1, sequence_length, vocab_size)

# 计算损失
# 为 CrossEntropyLoss 重塑：需要 (N, C) 和 (N)
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
# -100 是默认的忽略索引
loss = loss_fct(
    model_output_logits.view(-1, vocab_size),
    labels.view(-1)
)

print("Calculated Loss (only on completion tokens):", loss.item())
```

在此片段中，将与提示词标记对应的标签值设置为`-100`（PyTorch的CrossEntropyLoss的默认`ignore_index`）可确保这些位置不参与损失计算或梯度更新。只有模型对回应标记（`[60, 70, 80]`）的预测会受到惩罚。

### 连接和一致性

通常，提示词 (prompt)和回应会连接成一个单一序列，并输入给模型，通常像前面所示那样由一个特殊标记 (token)分隔。模型随后会处理整个序列。

> SFT输入序列的常见结构。提示词和回应被连接起来，通常带有分隔符和序列结束标记。损失通常只在回应部分计算。

在SFT数据集中所有示例中保持格式的绝对一致性是重要的。不一致地使用空格、换行符或特殊标记会使模型感到困惑，并阻碍其学习期望的模式。选择一种格式并统一应用。

### 数据结构示例

SFT数据集通常以JSON Lines (`.jsonl`) 等格式存储，其中每行是一个代表一个示例的JSON对象。

```json
{
    "prompt": "对情感进行分类：'这部电影太棒了！'\n情感：",
    "completion": " 积极"
}
{
    "prompt": "写一首关于月亮的短诗。\n诗歌：",
    "completion": " 银盘悬于夜幕，\n轻柔投下暗影，柔和而明亮"
}
{
    "prompt": "<|USER|> 水的沸点是多少摄氏度？\n<|ASSISTANT|>",
    "completion": " 水的沸点是100摄氏度"
}
```

或者，结构化格式可能会分开指令、输入和输出：

```json
{
    "instruction": "将以下英文文本翻译成西班牙语。",
    "input": "今天天气很好。",
    "output": " Hace buen tiempo hoy."
}
```

所选择的结构应清晰地映射到分词 (tokenization)和训练时使用的提示词 (prompt)-回应格式。

### 处理序列长度

一个实际的考量是模型支持的最大序列长度。如果连接的提示词 (prompt)和回应超过此限制，你需要一个截断策略。常见的方法包括：

- **截断回应：** 优先保留完整的提示词，尤其是当它包含重要背景信息或指令时，并截断回应的末尾。这可能会丢失一些目标信息。
- **截断提示词：** 保留完整的回应，并截断提示词的开头或结尾。这可能会丢失提示词中的重要背景信息。
- **丢弃示例：** 丢弃过长的示例。这是最简单的方法，但会减小数据集大小。

最佳策略取决于具体任务以及提示词与回应内容的相对重要性。

通过仔细格式化你的SFT数据，清晰区分提示词和回应，并确保一致性，你为模型提供了它所需的结构化输入，以便有效学习指令遵循和有益对话等对齐 (alignment)行为。

## 参考资料

- [Fine-tune a pretrained model with the SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face `trl`库中`SFTTrainer`的官方文档，提供了监督微调的数据准备、特殊标记和损失掩码的实用指导。
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) — Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2109.01652](https://doi.org/10.48550/arXiv.2109.01652)
  介绍了指令微调，这是SFT的核心概念，展示了如何使用多样化的任务格式（提示和完成）来提高模型的泛化能力和零样本性能。

---

[上一节](03-%E6%9E%84%E5%BB%BA%E9%AB%98%E8%B4%A8%E9%87%8F%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E9%9B%86.md) · [下一节](05-SFT%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B%E4%B8%8E%E8%B6%85%E5%8F%82%E6%95%B0.md)
