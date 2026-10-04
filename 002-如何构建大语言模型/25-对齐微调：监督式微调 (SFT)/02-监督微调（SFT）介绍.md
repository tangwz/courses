# 监督微调（SFT）介绍

来源：[原文](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-25-supervised-fine-tuning-alignment/introduction-sft)

[返回章节目录](README.md) · [返回课程目录](../README.md)

预训练 (pre-training)语言模型尽管其表现出色的能力，但通常缺乏可靠地遵循用户指令或遵守期望行为准则所需的特定调整。它们通过海量文本数据训练，以预测序列中的下一个词元 (token)，但这个目标不能直接转化为人类期望中定义的帮助性、诚实性或无害性。监督微调 (fine-tuning)（SFT）是一种旨在弥补这一差距的技术，它通过明确地教导模型如何以更受偏好的方式回应提示。

SFT 借助由精选输入提示及其对应期望输出组成的数据集，来调整预训练的大语言模型 (LLM)。可以将其视为直接为模型提供它*应该*如何表现的示例。模型不再从网络规模文本的隐含模式中学习，而是从优秀回复的明确示范中学习。这个过程包括在这些监督示例上进一步训练预训练模型，通常采用交叉熵等标准序列到序列损失函数 (loss function)。

### SFT 的运行原理

其核心是，SFT 通过使模型生成输出与微调 (fine-tuning)数据集中提供的目标输出之间的差异最小化，从而优化模型的参数 (parameter)。这个过程通常遵循以下步骤：

1. **从预训练 (pre-training)大语言模型 (LLM)开始：** 选择一个已经进行过广泛预训练的基础大语言模型。这个模型提供了基础知识和语言理解能力。
2. **准备指令数据集：** 整理或生成一个包含（提示，期望回复）对的数据集。这个数据集的质量和多样性显著影响SFT的结果。示例可能包含问题与有帮助的答案配对、指令与正确执行的任务配对，或对话轮次与适当的后续内容配对。
3. **格式化数据：** 每对数据通常被格式化为一个单独的序列，常使用特殊词元 (token)来划分提示和回复部分。例如：`<|prompt|> 马来西亚的首都在哪里？<|response|> 马来西亚的首都是吉隆坡。<|endoftext|>`。
4. **微调模型：** 在这个格式化的数据集上训练预训练模型。标准训练目标是预测下一个词元，但重要的是，损失通常*只*针对序列中属于`desired_response`部分的词元计算。提示词 (prompt)元作为上下文 (context)，但不直接参与损失计算或梯度更新。

这种有针对性的损失计算非常重要。我们希望模型学习*如何根据提示生成回复*，而不是简单地预测提示词元本身（这部分它在预训练期间已经学习过）。

考虑目标函数。在预训练期间，模型最大化整个文本语料库的似然，即 $P(	ext{文本})$。在SFT中，模型学习一个条件概率：给定特定提示，它最大化期望回复的似然，即 $P(	ext{回复} | 	ext{提示})$。这种转变使模型专注于根据指令输入生成适当的输出。

### SFT 流程图示

我们可以用图示呈现单个SFT训练步骤中信息的基本流向：

> SFT 过程的简化示意图，显示了如何使用数据集中的提示和期望回复来计算损失并更新预训练 (pre-training)大语言模型 (LLM)的权重 (weight)。

### 损失掩码

为了在实际操作中，使用 PyTorch 等框架实现有针对性的损失计算，我们通常会创建一个损失掩码。这个掩码确保只有与期望回复对应的词元 (token)才参与损失计算。

这是一个 PyTorch 代码片段，说明了这一点：

```python
import torch
import torch.nn.functional as F

# 假设：
# - logits: 模型输出的对数几率 [batch_size, sequence_length, vocab_size]
# - labels: 目标词元ID [batch_size, sequence_length]
# - prompt_lengths: 批次中每个项目的提示部分长度 [batch_size]
# - IGNORE_INDEX: 损失函数会忽略的特殊索引（例如 -100）

# 假设 IGNORE_INDEX 全局定义，例如：
IGNORE_INDEX = -100

def calculate_sft_loss(logits, labels, prompt_lengths):
    """仅针对回复词元计算交叉熵损失。"""
    batch_size, sequence_length, vocab_size = logits.shape

    # 为下一个词元预测平移对数几率和标签
    # 预测词元 i 的对数几率位于索引 i-1
    # 词元 i 的标签位于索引 i
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # 创建一个损失掩码
    # 初始化掩码为 1（计算损失）
    loss_mask = torch.ones_like(shift_labels, dtype=torch.bool)

    # 将提示词元的掩码设置为 0（忽略损失）
    for i in range(batch_size):
        # 提示长度包含初始词元，
        # 因此在平移序列中，掩码范围直到 prompt_length - 1
        prompt_end_index = prompt_lengths[i] - 1
        if prompt_end_index > 0: # 确保存在要掩码的提示部分
             loss_mask[i, :prompt_end_index] = 0

    # 应用掩码：掩码为 0 的地方，将标签设置为 IGNORE_INDEX
    masked_labels = shift_labels.masked_fill(~loss_mask, IGNORE_INDEX)

    # 展平序列维度以计算损失
    shift_logits = shift_logits.view(-1, vocab_size)
    masked_labels = masked_labels.view(-1)

    # 计算交叉熵损失，忽略 IGNORE_INDEX
    loss = F.cross_entropy(shift_logits,
                           masked_labels,
                           ignore_index=IGNORE_INDEX)

    return loss

# --- 示例用法 ---
# batch_size = 2
# sequence_length = 10
# vocab_size = 1000
# prompt_lengths = torch.tensor([3, 5]) # 每个项目的提示长度

# dummy_logits = torch.randn(batch_size,
#                            sequence_length,
#                            vocab_size,
#                            requires_grad=True)
# dummy_labels = torch.randint(0, vocab_size, (batch_size, sequence_length))

# sft_loss = calculate_sft_loss(dummy_logits, dummy_labels, prompt_lengths)
# print(f"Calculated SFT Loss: {sft_loss.item()}")
# sft_loss.backward() # 计算梯度
```

> 这个代码片段概述了如何掩盖损失计算，确保在SFT期间只有回复词元影响模型更新。

### SFT 的目的与目标

监督微调 (fine-tuning)有几个重要的对齐 (alignment)目标：

- **指令遵循：** 教导模型理解并执行自然语言提示中的命令或回答问题。
- **格式遵守：** 训练模型生成特定格式的输出（例如 JSON、Markdown、代码块、特定对话风格）。
- **增强可控性：** 使模型行为更可预测，并与用户在特定任务上的意图保持一致。
- **初步安全与帮助性：** 通过提供期望交互的示例（例如，拒绝有害请求、提供礼貌回复），引入基本的安全约束和有帮助的对话模式。

虽然 SFT 在教导模型*何种*回复是基于示例所期望的方面是有效的，但它本身不能完美地捕获人类偏好。它教导模型模仿所提供回复的风格和内容。对于更复杂的对齐目标，例如判断多个合理回复之间的相对质量，或优化“帮助性”等特质，SFT 之后通常会采用来自人类反馈的强化学习 (reinforcement learning)（RLHF）等技术，我们将在下一章讨论。SFT 提供了一个基础，使模型具备遵循指令的基本能力，之后再使用基于偏好的方法进行进一步的优化。

## 参考资料

- [Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky and James H. Martin (2025)
  这本全面的教材提供了语言模型、序列到序列模型和各种训练目标的基础知识，包括交叉熵损失，这些知识有助于理解监督微调的机制。

---

[上一节](01-%E5%A4%A7%E5%9E%8B%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%AF%B9%E9%BD%90%E7%9A%84%E7%9B%AE%E6%A0%87.md) · [下一节](03-%E6%9E%84%E5%BB%BA%E9%AB%98%E8%B4%A8%E9%87%8F%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E9%9B%86.md)
