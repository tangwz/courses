# 监督微调（SFT）的数据格式化

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-2-data-preparation-fine-tuning/formatting-sft-data)

[返回章节目录](README.md) · [返回课程目录](../README.md)

监督微调 (fine-tuning)（SFT）是一种常用方法，用于使预训练 (pre-training)的大型语言模型（LLM）适应遵循指令或在特定应用方向上发挥作用。其核心思想简单明了：向模型提供期望的输入-输出行为示例，并使用标准监督学习 (supervised learning)训练它复制该行为。然而，SFT 的有效性在很大程度上取决于您如何组织和格式化训练数据。模型需要明确的信号来理解输入中哪一部分是提示（或指令），以及哪一部分是它应学习生成的预期回复。

SFT 本质上要求数据以（提示，补全）对的形式呈现。模型被训练为在给定相应`提示`时生成`补全`。您如何定义这些提示和补全，以及如何将它们组合起来，非常重要。

### SFT 的常见数据结构

虽然存在各种自定义格式，但大多数 SFT 数据集都采用几种常见结构，通常存储为 JSON Lines (JSONL) 等格式，其中每一行都是一个独立的 JSON 对象，代表一个训练示例。

1. **简单提示-补全对：**
   这是最基础的格式，适用于输入是单段文本、输出是其延续或转换的任务。

   ```json
   {"prompt": "翻译成法语: 'Hello, world!'", "completion": "Bonjour le monde!"}
   {"prompt": "总结以下文本: [长文本输入]...", "completion": "[简洁摘要]..."}
   ```

   训练期间，`prompt` 和 `completion` 通常会被连接起来，有时会带有一个分隔符标记 (token)，模型学习预测属于 `completion` 的标记。
2. **指令遵循格式：**
   为了明确训练模型遵循指令，数据集通常会进一步细分输入，将指令本身与它应操作的任何特定输入数据分开。这有助于模型更好地泛化到新指令。

   ```json
   {"instruction": "识别主要情感。", "input": "这部电影太棒了！", "output": "积极"}
   {"instruction": "写一个关于勇敢骑士的短篇故事。", "input": "", "output": "吉迪恩爵士调整了头盔，山谷中回荡着巨龙的咆哮声……"}
   {"instruction": "提取电子邮件地址。", "input": "请通过 info@example.com 或 support@example.org 联系我们。", "output": "info@example.com, support@example.org"}
   ```

   为模型准备这些数据时，这些字段通常使用预定义模板组合成单个提示字符串。例如：

   ```
   以下是描述任务的指令，以及提供更多背景信息的输入。请编写一个适当的回复来完成请求。

   ### 指令：
   {instruction}

   ### 输入：
   {input}

   ### 回复：
   {output}
   ```

   模型随后被训练生成`### 回复:`后面的文本。具体模板结构可以不同，但数据集内部的一致性很重要。
3. **聊天和对话格式：**
   用于训练聊天模型或助手时，数据需要表示多轮对话。这通常被结构为回合列表，每个回合都有指定角色（例如，`user`、`assistant`、`system`）。

   ````json
   {"messages": [
     {"role": "system", "content": "你是一个乐于助人的助手。"},
     {"role": "user", "content": "法国的首都是哪里？"},
     {"role": "assistant", "content": "法国的首都是巴黎。"}
   ]}
   {"messages": [
     {"role": "user", "content": "编写一个计算阶乘的 Python 函数。"},
     {"role": "assistant", "content": "```python\ndef factorial(n):\n  if n == 0:\n    return 1\n  else:\n    return n * factorial(n-1)\n```"}
   ]}
   ````

   为 SFT 处理这种格式需要将对话历史序列化为单个线性序列。这通常需要特殊标记来划分回合和角色。例如，模型可能期望格式化的输入，例如：`<|im_start|>system\nYou are...<|im_end|>\n<|im_start|>user\nWhat is...?<|im_end|>\n<|im_start|>assistant\nThe capital is...<|im_end|>`. 模型随后被训练只预测对应`assistant`消息的标记。

### 特殊标记 (token)的作用

许多 LLM 依赖特殊标记在预训练 (pre-training)和微调 (fine-tuning)期间有效组织输入序列。这些标记作为分隔符，指示指令、用户输入、模型回复或对话回合之间的边界。示例包括：

- `<s>` 和 `</s>`：序列开始 (BOS) 和序列结束 (EOS) 标记。
- `[INST]` 和 `[/INST]`：由 Llama 2-Chat 等模型使用，用于封装用户指令。
- `<|im_start|>` 和 `<|im_end|>:` 由 ChatML 等模型（OpenAI 模型常用并被其他地方采用）使用，标记消息的开始和结束，常与角色标识符（`system`、`user`、`assistant`）配对。
- `<|user|>`, `<|assistant|>`, `<|endoftext|>`, `[SEP]`, `[CLS]`: 不同模型使用的各种其他标记。

**使用基础模型期望的特定模板和特殊标记来格式化您的 SFT 数据非常重要。** 使用错误格式或遗漏必要的特殊标记可能导致性能明显下降，因为模型无法正确解释其接收到的输入结构。务必查阅您正在微调的基础 LLM 的文档或模型卡。

### 损失计算的掩码处理

SFT 的一个基本方面是确保模型只学习预测目标补全或回复，而不是它被给予的提示或指令文本。如果模型被训练预测整个连接序列（提示 + 补全），损失计算将包含预测提示标记 (token)时产生的错误。这是不理想的；我们希望模型的梯度完全基于它生成*期望输出*的能力。

这通过**损失掩码**实现。在训练过程中，计算交叉熵损失时，对应提示标记的损失值被忽略。PyTorch 等框架中的常见做法是为对应提示标记的标签（目标标记 ID）分配一个特殊的忽略索引（例如 -100）。损失函数 (loss function)随后会自动跳过这些位置。

考虑一个简化示例：
提示：`翻译：你好`
补全：`Bonjour`

`翻译：你好 Bonjour`
标记化 ID（示例）：`[101, 8991, 102, 156 Hello, 205 Bonjour, 103]` （索引仅供说明）

用于损失计算的目标标签（带掩码）：`[-100, -100, -100, -100, 205, 103]`

在这里，`-100` 表示不应为对应“翻译:”、“你好”以及任何初始分隔符/特殊标记的损失进行计算。损失仅基于模型对“Bonjour”标记和随后的结束标记的预测进行计算。

> 流程图说明原始指令/输入/输出数据如何被转换为格式化字符串，进行标记化，以及标签如何被掩码以确保模型在监督微调 (fine-tuning)期间只从目标回复标记中学习。

### 为训练管道准备数据

在典型的 SFT 管道中，使用 Hugging Face 的 `transformers` 和 `datasets` 等库：

1. **加载数据：** 将数据集（通常是 JSONL 或类似格式）加载到 `Dataset` 对象中。
2. **格式化：** 对每个示例应用函数，根据目标模型的要求，将其组织成最终的提示字符串，包含指令、输入、输出和必要的特殊标记 (token)。
3. **标记化：** 使用模型专用的标记器将格式化字符串转换为标记 ID。确保标记器在填充和截断方面配置正确。值得注意的是，如果格式化步骤中未手动包含，标记器通常会自动添加模型专用的 BOS/EOS 标记。请注意这一点以避免重复。
4. **掩码：** 实现逻辑，通过复制标记化的 `input_ids` 并将对应提示部分的 ID 替换为忽略索引（例如 -100）来创建 `labels` 数组。
5. **批处理：** 数据加载器将批量处理这些示例，通常会在每个批次内将它们填充到相同长度。

仔细关注这些格式化步骤对于成功的 SFT 非常重要。格式的一致性、特殊标记的正确使用以及适当的损失掩码是有效使 LLM 适应特定任务和指令遵循行为的重要构成部分。

## 参考资料

- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) — Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, Quoc V. Le (2022)
  Journal: arXiv preprint arXiv:2109.01652; DOI: [10.48550/arXiv.2109.01652](https://doi.org/10.48550/arXiv.2109.01652)
  一篇基础论文，介绍了指令微调，展示了在指令格式数据集集合上进行微调如何提高零样本泛化能力，并影响监督微调的数据格式。
- [Fine-tuning](https://platform.openai.com/docs/guides/fine-tuning/fine-tuning) — OpenAI (2024)
  Publisher: OpenAI
  官方指南，介绍如何准备用于微调OpenAI模型的数据，包括聊天格式输入的MESSAGES要求和示例数据结构。

---

[上一节](02-%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E9%9B%86%E7%9A%84%E8%8E%B7%E5%8F%96%E4%B8%8E%E6%9E%84%E5%BB%BA.md) · [下一节](04-%E5%9F%9F%E9%80%82%E5%BA%94%E6%95%B0%E6%8D%AE%E8%A6%81%E6%B1%82.md)
