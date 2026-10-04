---
course: "how-to-build-a-large-language-model"
chapter: "intrinsic-evaluation-metrics"
lesson: "concept-language-model-evaluation"
sourceId: 6055
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-21-intrinsic-evaluation-metrics/concept-language-model-evaluation"
title: "语言模型评估方法"
description: "阐述评估训练后的语言模型表现的各类方法。"
order: 1
plots: []
sourceHash: "9412232c7869c52fa5bd296e5a6d881414a393500691dd32b2ecbac7f5dda057"
sourceCorrections: []
---

投入大量计算资源并精心整理海量数据集来预训练 (pre-training)大语言模型 (LLM)后，随之而来的紧迫问题是：它的效果如何？仅仅完成训练过程并不能保证得到一个有用或有效的模型。我们需要严格的方法来评估其表现，了解其能力，并找出其不足。评估大语言模型是一个多方面的过程，因为“好”本身在不同情境下有不同含义，可以是从原始预测准确性，到对话中的实用性，或在特定下游应用中的表现。

用于大语言模型评估的技术通常分为几大类：

1. **内部评估：** 这种方法侧重于依据模型的主要训练目标进行评估，通常是预测下一个词元 (token)或填充被遮蔽的词元。它衡量模型在训练数据本身中学习统计模式和分布的程度。如章引言所述，困惑度是最突出的内部评估指标。这些评估直接在模型输出（概率）上进行，使用一个保留的测试集，不涉及外部任务。
2. **外部评估：** 这种方法评估模型在预训练阶段并未明确训练过的特定下游任务上的表现。例子包括问答、文本摘要、情感分析或代码生成。这通常涉及对预训练模型在任务专有数据集上进行微调 (fine-tuning)，然后使用任务专有指标（例如，F1分数、BLEU分数、准确率）来衡量其表现。外部评估提供衡量模型实用性和迁移学习 (transfer learning)能力的标准。我们将在第22章详细审视这些方法。
3. **人工评估：** 这涉及人工评估模型输出的质量，依据连贯性、相关性、实用性、无害性、事实准确性或指令遵循等标准。尽管常被认为是判断整体质量和对齐 (alignment)效果的黄金标准，但人工评估通常昂贵、耗时、主观且难以扩展。它常用于开发的后期阶段，或用于特定的对齐目标，例如通过人工反馈强化学习 (reinforcement learning)（RLHF）实现的目标，这在第26章有说明。

> 大语言模型评估方法的主要类别。

“本章主要关注内部评估。尽管与评估任务表现相比，它可能显得有限，但内部评估在大语言模型开发周期中扮演着重要角色。其主要优点包括：”

- **直接衡量：** 它直接量化 (quantization)了模型的核心语言模型能力，评估它捕捉其所训练语言的底层概率分布的程度。
- **计算效率高：** 计算困惑度等指标通常比设置和运行多个下游任务评估或协调人工评估快得多且所需资源较少。这有助于在开发过程中快速获得反馈。
- **开发监控：** 内部评估指标对跟踪训练进展极有帮助。在训练期间监控验证集上的困惑度有助于识别收敛情况，诊断不稳定性（如第24章所述），并对学习率或停止标准做出决定。
- **对比分析：** 它提供了一种标准方法，用于在受控条件下比较不同的模型架构、超参数 (parameter) (hyperparameter)或预训练数据变体，前提是比较是在相同的测试集上进行并使用相同的词元化方法。

训练损失与内部评估之间有直接关联。在训练期间，模型通常被优化以最小化交叉熵损失，这在数学上与困惑度相关。在PyTorch中，为评估目的计算此损失涉及在保留数据集上进行前向传播，而不进行反向传播 (backpropagation)。

```python
import torch
import torch.nn.functional as F

# 假设 'model' 是您预训练的大语言模型
# 假设 'eval_dataloader' 提供 input_ids 和 attention_mask 的批次
# 评估循环步骤示例（简化）

model.eval() # 将模型设置为评估模式
total_loss = 0
total_tokens = 0

with torch.no_grad(): # 禁用梯度计算
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone() # 通常，预测下一个词元

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        # 模型输出通常直接包含损失
        # 在内部使用交叉熵计算
        loss = outputs.loss

        # 备选方案：如果模型返回 logits，则手动计算损失
        # logits = outputs.logits
        # 移动 logits 和 labels 以进行下一个词元预测任务
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # 仅计算非填充词元的损失
        # loss_fct = torch.nn.CrossEntropyLoss()
        # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
        #                 shift_labels.view(-1))

        # 累加损失，按批次中的词元数量加权
        # （需要仔细处理填充词元以获得准确的困惑度）
        num_tokens = attention_mask.sum() # 简单计数，需改进
        total_loss += loss.item() * num_tokens # 按词元数量加权损失
        total_tokens += num_tokens

# 平均损失与困惑度相关
average_loss = total_loss / total_tokens
# 困惑度 = exp(平均损失) - 详细信息请见下一节
# print(f"Average Cross-Entropy Loss: {average_loss}")
```

> 简化的 PyTorch 代码片段，说明评估期间的损失计算，与困惑度相关。

然而，承认内部评估的局限性很重要。低困惑度分数并不能自动意味着模型能生成有用、真实、连贯或安全的文本。模型可以通过过拟合 (overfitting)训练数据的统计模式来实现低困惑度，可能学会生成概率高但信息量少的重复或通用序列。此外，正如本章后面会看到的那样，困惑度值对所用的特定词元化方案和评估数据集的性质高度敏感，使得跨不同设置的比较变得困难。

因此，内部评估应被视为全面评估策略的一个必要组成部分。它提供对模型语言模型能力的基本检查，但必须辅以外部评估和可能的人工评估，以全面了解大语言模型的表现及其对特定应用的适用性。我们现在将继续详细定义和分析困惑度。

## 参考资料

- [Speech and Language Processing (3rd ed. draft)](https://web.stanford.edu/~jurafsky/slp3/) — Daniel Jurafsky, James H. Martin (2025)
  对语言建模和困惑度提供了基础而详细的解释，包括其数学基础和在自然语言处理中的应用。
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen (2023)
  Journal: arXiv preprint arXiv:2303.18223; DOI: [10.48550/arXiv.2303.18223](https://doi.org/10.48550/arXiv.2303.18223)
  全面概述了大型语言模型研究，包括对这些模型各种评估方法的结构化讨论。
